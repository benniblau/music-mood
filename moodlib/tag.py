"""Stage 2: apply the ontology to every stale track.

Resumable by construction. Work is selected by `db.stale_where()` rather than
tracked in a progress file, so "what still needs doing" is derived from the data
on every run -- there is no separate state to corrupt, and Ctrl-C costs at most
the batches in flight. Re-running after an interruption simply finds less to do.

Threads make the HTTP calls; only the main thread touches SQLite. That keeps the
concurrency story simple (SQLite connections are not thread-safe) and means a
batch is committed as it lands rather than at the end of a four-hour run.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Sequence

from moodlib import config, db, llm, ontology, progress


ICON_FAIL = progress.ICON["error"]


def _describe(index: int, row) -> dict:
    """Compact per-track payload. Key names are short because 23k tracks of them add up."""
    item = {
        "i": index,
        "a": row["artist"] or row["album_artist"],
        "t": row["title"],
        "al": row["album"],
        "y": row["year"],
    }
    raw = (row["genre_raw"] or "").strip()
    if raw:
        item["raw"] = raw
        hint = ontology.genre_hint(raw)
        if hint:
            # Only pass a hint we actually trust. Decade tags ("80s"), junk
            # ("127", "PROMO") and blanks map to None on purpose: a wrong hint is
            # worse than none, because the model tends to defer to it.
            item["hint"] = hint[1] or hint[0]
    return item


def _prompt(rows: Sequence) -> str:
    tracks = [_describe(i, row) for i, row in enumerate(rows)]
    return (ontology.tag_instructions() + "\n\n"
            + json.dumps(tracks, ensure_ascii=False))


def _unique(terms: list[str]) -> list[str]:
    """Order-preserving dedupe."""
    return list(dict.fromkeys(terms))


def _ingest(entry: dict, model: str, content_key: str) -> dict:
    """Translate one wire-format object into database columns."""
    row: dict = {}
    # Axes arrive as 0-100 integers (cheaper tokens, better behaved under the
    # schema) and are stored as 0.0-1.0 floats to match the Spotify convention.
    for short, name in ontology.AXIS_KEYS.items():
        row[name] = max(0.0, min(1.0, float(entry[short]) / 100.0))
    for short, name in ontology.GEMS_KEYS.items():
        row[name] = max(0.0, min(100.0, float(entry[short])))
    # Deduplicate here rather than in the schema: `uniqueItems` is valid JSON
    # Schema but vLLM's grammar backend 500s on it, so the constraint has to live
    # in code. A repeated term would otherwise double-count in query scoring.
    row["adjectives_json"] = json.dumps(_unique(entry["m"]), ensure_ascii=False)
    row["contexts_json"] = json.dumps(_unique(entry["c"]), ensure_ascii=False)
    # Genre is derived, never asked for: see ontology.STYLES_BY_GENRE for why.
    row["discogs_style"] = entry["s"]
    row["discogs_genre"] = ontology.genre_for_style(entry["s"])
    row["confidence"] = int(entry["k"])
    row["model"] = model
    row["ontology_version"] = ontology.ONTOLOGY_VERSION
    row["content_key"] = content_key
    row["tagged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    row["error"] = None
    return row


def _tag_batch(rows: Sequence, session) -> tuple[Sequence, dict[int, dict] | None, str]:
    """Run one batch. Returns (rows, {index: entry} or None, error message)."""
    try:
        result = llm.complete_json(
            _prompt(rows), ontology.tag_schema(), schema_name="moods",
            # Sized from the batch, not from LLM_MAX_TOKENS: an over-generous
            # reservation is what makes vLLM decode these almost serially.
            max_tokens=config.tag_max_tokens(len(rows)),
            session=session)
        entries = {int(e["i"]): e for e in result["r"]}
        missing = [i for i in range(len(rows)) if i not in entries]
        if missing:
            return rows, None, f"model omitted {len(missing)} of {len(rows)} tracks"
        return rows, entries, ""
    except Exception as exc:  # noqa: BLE001 - reported per batch, never fatal
        return rows, None, str(exc)


def _chunks(rows: Sequence, size: int) -> Iterator[Sequence]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def _label(row) -> str:
    """`Artist — Title` for the progress line, falling back to the filename."""
    artist = (row["artist"] or row["album_artist"] or "").strip()
    title = (row["title"] or "").strip()
    if artist and title:
        return f"{artist} — {title}"
    return title or artist or row["path"]


def run(limit: int | None = None, target: str | None = None,
        batch_size: int | None = None, concurrency: int | None = None,
        conn=None, verbose: bool = False) -> dict[str, int]:
    batch_size = batch_size or config.TAG_BATCH_SIZE
    concurrency = concurrency or config.TAG_CONCURRENCY
    owns_conn = conn is None
    conn = conn or db.connect()

    pending = db.stale_tracks(conn, limit=limit, target=target)
    if not pending:
        progress.note("done", "nothing to tag — everything is current")
        if owns_conn:
            conn.close()
        return {"tagged": 0, "failed": 0}

    batches = list(_chunks(pending, batch_size))
    progress.note("tag", f"{len(pending):,} tracks to characterise, "
                         f"{len(batches):,} batches of {batch_size}, "
                         f"{concurrency} concurrent")

    session = llm.new_session()
    model = llm.resolve_model(session)
    bar = progress.Progress(len(pending), "tag", unit="tracks", verbose=verbose)
    tagged = failed = 0
    retry_singles: list = []

    # as_completed, not map: map yields in submission order, so a single slow
    # request blocks every finished batch behind it from being committed. On the
    # first full run that showed up as a dead stall -- 16 workers busy, the
    # database frozen at 520 rows, and no way to tell progress from a hang.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_tag_batch, batch, session) for batch in batches]
        for future in as_completed(futures):
            rows, entries, error = future.result()
            if entries is None:
                # A failed batch is split rather than abandoned: one unparseable
                # track should not cost the other nineteen.
                if len(rows) > 1:
                    bar.note("warn", f"batch of {len(rows)} failed ({error[:80]}) "
                                     f"— retrying individually")
                    retry_singles.extend(rows)
                else:
                    db.write_error(conn, rows[0]["id"], error)
                    failed += 1
                    bar.advance(f"{ICON_FAIL} {_label(rows[0])}")
                continue
            for index, row in enumerate(rows):
                try:
                    db.write_mood(conn, row["id"],
                                  _ingest(entries[index], model, row["content_key"]))
                    tagged += 1
                    bar.advance(_label(row))
                except (KeyError, ValueError, TypeError) as exc:
                    db.write_error(conn, row["id"], f"bad field: {exc}")
                    failed += 1
                    bar.advance(f"{ICON_FAIL} {_label(row)}")
            conn.commit()

    if retry_singles:
        bar.note("tag", f"retrying {len(retry_singles)} tracks individually")
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_tag_batch, [row], session)
                       for row in retry_singles]
            for future in as_completed(futures):
                rows, entries, error = future.result()
                row = rows[0]
                if entries is None:
                    db.write_error(conn, row["id"], error)
                    failed += 1
                    bar.advance(f"{ICON_FAIL} {_label(row)}")
                    continue
                try:
                    db.write_mood(conn, row["id"],
                                  _ingest(entries[0], model, row["content_key"]))
                    tagged += 1
                    bar.advance(_label(row))
                except (KeyError, ValueError, TypeError) as exc:
                    db.write_error(conn, row["id"], f"bad field: {exc}")
                    failed += 1
                    bar.advance(f"{ICON_FAIL} {_label(row)}")
        conn.commit()

    bar.close(f"tagged {tagged:,}" + (f", failed {failed}" if failed else ""))
    if owns_conn:
        conn.close()
    return {"tagged": tagged, "failed": failed}
