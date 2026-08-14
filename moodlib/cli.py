"""Argument parsing and dispatch. Run via ../run.py.

Every flag mirrors a .env key: the file sets the default, a flag wins for a
single invocation. `None` therefore means "not given" everywhere below, so the
config value survives -- never bake a literal into a default here, or the .env
key silently stops working.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from moodlib import config, db, ontology, playlist, progress, query, scan, tag


def _add_scan(sub) -> None:
    p = sub.add_parser("scan", help="index the library into SQLite (incremental)")
    p.add_argument("--library", type=Path, help="override LIBRARY_PATH")
    p.add_argument("--force", action="store_true",
                   help="re-probe every file, ignoring size/mtime")
    p.add_argument("--workers", type=int, help="override SCAN_WORKERS")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print one line per file instead of a live status line")


def _add_tag(sub) -> None:
    p = sub.add_parser("tag", help="mood-tag stale tracks")
    p.add_argument("target", nargs="?",
                   help='limit to a path prefix, e.g. "Pendulum" or "Pendulum/Hold Your Colour"')
    p.add_argument("--all", action="store_true", help="tag everything stale (default)")
    p.add_argument("--limit", type=int, help="stop after N tracks — use to validate first")
    p.add_argument("--batch-size", type=int, help="override TAG_BATCH_SIZE")
    p.add_argument("--concurrency", type=int, help="override TAG_CONCURRENCY")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print one line per track instead of a live status line")


def _add_stats(sub) -> None:
    p = sub.add_parser("stats", help="coverage, staleness and score distributions")
    p.add_argument("--full", action="store_true",
                   help="include per-dimension distributions and correlations")


def _add_playlist(sub) -> None:
    p = sub.add_parser("playlist", help="generate a playlist from a mood description")
    p.add_argument("mood", help='e.g. "melancholy but driving, late night"')
    p.add_argument("-n", "--count", type=int, help="override PLAYLIST_SIZE")
    p.add_argument("-o", "--output", type=Path, help="destination .m3u8")
    p.add_argument("--max-per-artist", type=int, help="override MAX_PER_ARTIST")
    p.add_argument("--min-confidence", type=int, choices=(0, 1, 2),
                   help="override MIN_CONFIDENCE")
    p.add_argument("--genre", action="append", metavar="G",
                   help="restrict to a Discogs genre (repeatable; overrides "
                        "any genre inferred from the request)")
    p.add_argument("--style", action="append", metavar="S",
                   help="restrict to a Discogs style (repeatable)")
    p.add_argument("--year-from")
    p.add_argument("--year-to")
    p.add_argument("--title", help="override the generated playlist title")
    p.add_argument("--seed", type=int, help="make the selection reproducible")
    p.add_argument("--explain", action="store_true",
                   help="print the translated query, its rationale and per-pick scores")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Mood-based playlist generator for a local music library.")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_scan(sub)
    _add_tag(sub)
    _add_stats(sub)
    _add_playlist(sub)
    return parser


def cmd_scan(args) -> int:
    scan.build(root=args.library, force=args.force, workers=args.workers,
               verbose=args.verbose)
    return 0


def cmd_tag(args) -> int:
    result = tag.run(limit=args.limit, target=args.target,
                     batch_size=args.batch_size, concurrency=args.concurrency,
                     verbose=args.verbose)
    return 0 if result["failed"] == 0 else 1


def cmd_stats(args) -> int:
    conn = db.connect()
    counts = db.counts(conn)
    icon = progress.ICON
    print(f"{icon['library']} library    {config.LIBRARY_PATH}")
    print(f"{icon['db']} database   {config.DB_PATH}")
    print(f"{icon['stats']} ontology   v{ontology.ONTOLOGY_VERSION}")
    print()
    print(f"{icon['track']} tracks             {counts['tracks']:>7,}")
    print(f"   {icon['done']} tagged & fresh  {counts['tagged']:>7,}")
    for label, key, mark in (
        ("never tagged", "never_tagged", icon["tag"]),
        ("stale (content)", "stale_content", icon["changed"]),
        ("stale (ontology)", "stale_ontology", icon["changed"]),
        ("errors", "errors", icon["error"]),
        ("missing", "missing", icon["missing"]),
    ):
        # Only the interesting lines carry a marker; a zero is not news.
        flag = mark if counts[key] else "  "
        print(f"   {flag} {label:<16} {counts[key]:>7,}")

    rows = list(conn.execute(
        "SELECT m.* FROM moods m JOIN tracks t ON t.id = m.track_id "
        "WHERE m.error IS NULL AND m.content_key = t.content_key"))
    if not rows:
        print(f"\n{icon['warn']} nothing tagged yet — run `tag --limit 200` first")
        conn.close()
        return 0

    confidence = [r["confidence"] or 0 for r in rows]
    spread = {level: confidence.count(level) for level in (0, 1, 2)}
    print(f"\n{icon['llm']} confidence   "
          f"0 (guessed) {spread[0]:,}   1 (knows artist) {spread[1]:,}   "
          f"2 (knows track) {spread[2]:,}")
    if spread[2] == len(rows):
        print(f"   {icon['warn']} every track is confidence 2 — the model is not "
              "being honest about what it recognises.")

    def distribution(name: str, values: list[float], scale: float) -> str:
        values = [v for v in values if v is not None]
        if not values:
            return f"  {name:<18} (no data)"
        mean = statistics.fmean(values)
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        # A flat distribution means the model hedged instead of characterising,
        # which is the one failure this report exists to catch.
        flag = f"  {progress.ICON['warn']}clustered" if sd < 0.08 * scale else ""
        bar = "█" * round(20 * mean / scale)
        return (f"   {name:<18} {bar:<20} mean {mean / scale:4.2f}  "
                f"sd {sd / scale:4.2f}{flag}")

    print(f"\n{icon['stats']} audio axes (0-1, Spotify vocabulary)")
    for name in ontology.AXES:
        print(distribution(name, [r[name] for r in rows], 1.0))
    print(f"\n{icon['stats']} GEMS-9 emotion (0-100)")
    for name in ontology.GEMS:
        print(distribution(name, [r[name] for r in rows], 100.0))

    genres: dict[str, int] = {}
    for row in rows:
        genres[row["discogs_genre"]] = genres.get(row["discogs_genre"], 0) + 1
    print(f"\n{icon['track']} Discogs genres")
    for name, count in sorted(genres.items(), key=lambda kv: -kv[1])[:12]:
        print(f"   {count:>7,}  {name}")

    if args.full:
        print(f"\n{icon['stats']} GEMS correlations "
              "(near 1.0 means two dimensions have collapsed into one)")
        collapsed = False
        names = list(ontology.GEMS)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                xs = [r[a] for r in rows]
                ys = [r[b] for r in rows]
                try:
                    r = statistics.correlation(xs, ys)
                except (statistics.StatisticsError, ValueError):
                    continue
                if abs(r) > 0.85:
                    print(f"   {icon['warn']} {a} ~ {b}: {r:+.2f}")
                    collapsed = True
        if not collapsed:
            print(f"   {icon['done']} all nine dimensions are distinct")

    conn.close()
    return 0


def cmd_playlist(args) -> int:
    conn = db.connect()
    if db.counts(conn)["tagged"] == 0:
        conn.close()
        raise SystemExit("nothing is tagged yet — run `scan` then `tag --all` first")

    icon = progress.ICON
    progress.note("query", f"understanding “{args.mood}” …")
    structured = query.translate(args.mood)
    db.record_query(conn, args.mood, structured)
    title = (args.title or structured.get("title") or "").strip()
    progress.note("playlist", f"“{title}”" if title else "(untitled)")
    progress.note("llm", structured.get("rationale", "").strip())

    limits = query.constraints(
        structured,
        min_confidence=args.min_confidence,
        genres=args.genre or (), styles=args.style or (),
        year_from=args.year_from, year_to=args.year_to)
    limits["min_confidence"] = max(limits["min_confidence"], config.MIN_CONFIDENCE)

    described = []
    if limits["genres"] or limits["styles"]:
        described.append("/".join([*limits["styles"], *limits["genres"]]))
    if limits["year_from"] or limits["year_to"]:
        described.append(f"{limits['year_from'] or '…'}–{limits['year_to'] or '…'}")
    if limits["min_confidence"]:
        described.append(f"k≥{limits['min_confidence']}")
    if described:
        progress.note("info", "filtering to " + ", ".join(described))

    scored = query.score_all(conn, structured, **limits)
    if not scored:
        conn.close()
        raise SystemExit(f"{icon['warn']} no tracks matched the filters — try "
                         "relaxing --genre/--year/--min-confidence")

    count = args.count or config.PLAYLIST_SIZE
    chosen = query.select(
        scored, count,
        max_per_artist=(config.MAX_PER_ARTIST if args.max_per_artist is None
                        else args.max_per_artist),
        seed=args.seed)

    progress.note("playlist", f"scored {len(scored):,} candidates, "
                              f"selected {len(chosen)}")

    if args.explain:
        print()
        constrained = {
            name: spec for name, spec in structured.items()
            if isinstance(spec, dict) and float(spec.get("weight") or 0) > 0
        }
        for name, spec in sorted(constrained.items(),
                                 key=lambda kv: -float(kv[1]["weight"])):
            print(f"  {name:<18} target {spec['target']:>3}  weight {spec['weight']:.2f}")
        for key in ("adjectives", "avoid", "contexts"):
            if structured.get(key):
                print(f"   {key:<18} {', '.join(structured[key])}")
        print()

    # Confidence shown per pick: a playlist built from guesses should look
    # different from one built from tracks the model actually recognised.
    marks = {2: "🎯", 1: "👤", 0: "❓"}
    for index, item in enumerate(chosen, 1):
        mark = marks.get(int(item.parts["confidence"]), "❓")
        print(f"{index:3d}. {mark} {item.artist} — {item.title}")
        if args.explain:
            parts = item.parts
            print(f"        score {item.score:.3f}  "
                  f"axis {parts['axis']:.2f}  gems {parts['gems']:.2f}  "
                  f"vocab {parts['vocab']:+.2f}")

    if len(chosen) < count:
        # Almost always the per-artist cap against a narrow candidate pool.
        # Say so rather than quietly handing back a shorter playlist.
        print(f"\n{icon['warn']} wanted {count} tracks but only {len(chosen)} "
              "qualified — raise --max-per-artist, lower --min-confidence, or "
              "relax the filters")

    # The generated title drives the default filename, because Music.app names
    # an imported playlist after the file rather than the #PLAYLIST directive.
    destination = args.output or (
        config.DATA_DIR / f"{playlist.filename_for(title)}.m3u8")
    written = playlist.write(chosen, destination, title=title)
    total = sum(item.duration for item in chosen)
    print(f"\n{icon['done']} “{title}” — {len(chosen)} tracks, "
          f"{total / 60:.0f} min → {written}")
    print(f"   {icon['playlist']} import via Music.app → File → Library → "
          "Import Playlist…")
    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "scan": cmd_scan, "tag": cmd_tag,
        "stats": cmd_stats, "playlist": cmd_playlist,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        progress.note("warn", "interrupted — progress is committed, re-run to resume")
        return 130
