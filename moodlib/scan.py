"""Stage 1: build and maintain the track index from the files themselves.

`scan` is always incremental and safe to re-run; there is no separate "update"
command. The state machine each file falls through:

    new track          nothing matches                   -> insert, tag later
    tags changed       same file, content_key differs    -> update, re-tag
    file touched only  mtime/size differ, tags identical -> stat update, NO re-tag
    renamed            inode matches at a new path       -> update path, keep id
    rewritten/re-added inode gone, content_key matches   -> update path, keep id
    deleted            nothing matches                   -> soft-mark missing
    ontology moved on  handled in db.stale_where()       -> re-tag

Identity is resolved in that order -- path, then inode, then content_key -- and
the order matters. Music renames files from their tags, so a tag edit changes the
path AND the content_key at the same time; only the inode survives that, and
without it such a track looks exactly like a deletion plus an unrelated new file.
Conversely a re-added file keeps its tags but gets a fresh inode, which is what
content_key is there to catch. See db.py for the full identity model.

Two safety rails, both learned from how this library actually fails: deletions
are soft, and a scan that would mark more than SCAN_MISSING_ABORT_PCT of known
tracks missing refuses to do anything at all. The overwhelmingly likely cause of
the latter is an unmounted NAS, and a scan that "successfully" marks 23,000
tracks gone is indistinguishable from a real library wipe.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from moodlib import config, db, progress


@dataclass
class ScanReport:
    added: int = 0
    changed: int = 0
    touched: int = 0
    moved: int = 0
    missing: int = 0
    restored: int = 0
    unchanged: int = 0
    probe_failed: int = 0
    #: Of `moved`, how many were matched by inode rather than by tags. These are
    #: the ones a tags-only scheme would have lost.
    moved_by_inode: int = 0

    def summary(self) -> str:
        moved = f"moved {self.moved}"
        if self.moved_by_inode:
            moved += f" ({self.moved_by_inode} by inode)"
        return (
            f"added {self.added}, changed {self.changed}, {moved}, "
            f"touched {self.touched}, missing {self.missing}, "
            f"restored {self.restored}, unchanged {self.unchanged}"
            + (f", probe failures {self.probe_failed}" if self.probe_failed else "")
        )


def _walk(root: Path) -> dict[str, os.stat_result]:
    """Every audio file under root, keyed by path relative to the library.

    Paths are stored NFC-normalised. macOS hands them back in NFD, and comparing
    an NFD path from disk against an NFC path from the database silently misses
    for every track with an accent in its name.
    """
    extensions = tuple(e.lower() for e in config.AUDIO_EXTENSIONS)
    found: dict[str, os.stat_result] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith(".") or not name.lower().endswith(extensions):
                continue
            full = Path(dirpath) / name
            try:
                found[unicodedata.normalize("NFC", str(full.relative_to(root)))] = full.stat()
            except (OSError, ValueError):
                continue
    return found


#: Leading "07 " or "1-05 " track numbers. Capped at two digits on purpose:
#: three would eat the title of "127 Hours Theme". A two-digit title like
#: "99 Problems" is still stripped wrongly, but this fallback only runs on files
#: with no tags at all, and the artist/album recovered alongside it carry most of
#: the signal the model needs.
_TRACK_NUMBER = re.compile(r"^\s*(?:\d{1,2}[-\s.]+)?\d{1,2}[\s.\-]+")


def _from_path(relative: str) -> dict[str, str]:
    """Recover artist/album/title from the path when the tags are empty.

    The library's layout is strictly `Artist / Album / Track` (its own CLAUDE.md
    says so), and a small share of files carry no usable tags at all -- the
    Massacre rip in this collection has only an iTunSMPB atom. Those rows would
    otherwise reach the model as blanks, which is not a track description but an
    invitation to hallucinate. `_` is Music's substitute for an illegal
    character, almost always `:`.
    """
    parts = Path(relative).parts
    stem = Path(relative).stem
    title = _TRACK_NUMBER.sub("", stem).strip().replace("_", ":")
    return {
        "artist": parts[0] if len(parts) >= 3 else "",
        "album": parts[-2] if len(parts) >= 2 else "",
        "title": title,
    }


def _probe(args: tuple[Path, str]) -> dict | None:
    """Read tags and stream info for one file via ffprobe."""
    root, relative = args
    full = root / relative
    try:
        result = subprocess.run(
            [config.FFPROBE_BIN, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(full)],
            capture_output=True, timeout=120)
        data = json.loads(result.stdout)
    except Exception:
        return None

    fmt = data.get("format", {})
    tags = {k.lower(): v for k, v in fmt.get("tags", {}).items()}

    def tag(*keys: str) -> str:
        for key in keys:
            value = tags.get(key)
            if value not in (None, ""):
                return str(value).strip()
        return ""

    def number(value) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # Fill only what the tags do not supply — a real tag always wins.
    fallback = _from_path(relative)

    return {
        "path": relative,
        "artist": tag("artist") or fallback["artist"],
        "album": tag("album") or fallback["album"],
        "album_artist": tag("album_artist", "albumartist"),
        "title": tag("title") or fallback["title"],
        "genre_raw": tag("genre"),
        # ffprobe surfaces the year as `date` on MP4 and `year` on some ID3s.
        "year": tag("date", "year")[:4],
        "duration": number(fmt.get("duration")),
        # INITIALKEY is already Camelot notation ("5A") in this library, but only
        # ~17% of tracks carry it. Stored honestly: present where present.
        "camelot_key": tag("initialkey", "initial_key", "tkey"),
    }


def build(root: Path | None = None, force: bool = False, conn=None,
          workers: int | None = None, verbose: bool = False) -> ScanReport:
    root = config.require_library(root)
    config.require_ffprobe()
    workers = workers or config.SCAN_WORKERS
    owns_conn = conn is None
    conn = conn or db.connect()
    report = ScanReport()

    progress.note("library", f"scanning {root}")
    on_disk = _walk(root)
    progress.note("scan", f"{len(on_disk):,} audio files on disk")

    known = {
        row["path"]: row
        for row in conn.execute(
            "SELECT id, path, size, mtime, inode, duration, content_key, "
            "missing_since FROM tracks")
    }

    disappeared = [p for p in known if p not in on_disk]

    # --- safety rail, checked BEFORE any write ---------------------------
    # The rail measures net *loss of files from the volume*, not path churn.
    # Those are very different things, and conflating them makes it fire on the
    # wrong event: a library reorganisation renames thousands of paths while the
    # file count barely moves. Observed here -- 5,934 paths vanished in one pass
    # while 22,966 of 22,987 files were still present, and a churn-based rail
    # refused a scan that was merely going to follow some renames.
    #
    # Net shortfall keeps the property that matters. An unmounted share reports
    # zero files and still aborts; a half-mounted one reports far too few and
    # still aborts. A rename storm nets out to roughly zero and proceeds, where
    # inode matching resolves it correctly.
    present_known = [p for p in known if known[p]["missing_since"] is None]
    shortfall = len(present_known) - len(on_disk)
    if present_known and shortfall > 0:
        pct = 100.0 * shortfall / len(present_known)
        if pct > config.SCAN_MISSING_ABORT_PCT:
            raise SystemExit(
                f"refusing to scan: {root} holds {len(on_disk)} audio files but "
                f"{len(present_known)} are known — {shortfall} ({pct:.1f}%) have "
                f"vanished from the volume, above the "
                f"{config.SCAN_MISSING_ABORT_PCT}% abort threshold.\n"
                "Is the NAS fully mounted? Nothing has been changed.\n"
                "Raise SCAN_MISSING_ABORT_PCT in .env if this really is intended.")

    # --- decide which files actually need ffprobe ------------------------
    needs_probe: list[str] = []
    backfill: list[tuple[int, int, int]] = []
    for relative, stat in on_disk.items():
        row = known.get(relative)
        if row is None:
            needs_probe.append(relative)
        elif force or row["size"] != stat.st_size or row["mtime"] != stat.st_mtime:
            needs_probe.append(relative)
        else:
            report.unchanged += 1
            # Identity comes from stat(), not from ffprobe, so it can be filled in
            # for untouched files at no cost. Without this an existing database
            # would never acquire inodes at all: unchanged files are never
            # probed, so they would sit at NULL and the inode branch would be
            # dead for every track that mattered.
            if row["inode"] != stat.st_ino:
                backfill.append((stat.st_dev, stat.st_ino, row["id"]))

    if backfill:
        conn.executemany(
            "UPDATE tracks SET device = ?, inode = ? WHERE id = ?", backfill)
        conn.commit()
        progress.note("db", f"recorded file identity for {len(backfill):,} tracks")

    if needs_probe:
        progress.note("probe", f"reading tags from {len(needs_probe):,} changed files "
                               f"({report.unchanged:,} unchanged)")
    else:
        progress.note("probe", f"no files changed ({report.unchanged:,} unchanged)")

    # --- indexes over the tracks that vanished from their old path --------
    # These are what a newly-seen file is matched against before we conclude it
    # is genuinely new. Built once; entries are consumed as they are claimed.
    gone_by_inode: dict[int, list[str]] = {}
    gone_by_key: dict[str, list[str]] = {}
    for path in disappeared:
        row = known[path]
        if row["missing_since"] is not None:
            continue
        if row["inode"] is not None:
            gone_by_inode.setdefault(row["inode"], []).append(path)
        gone_by_key.setdefault(row["content_key"], []).append(path)
    claimed: set[str] = set()

    # --- probe and upsert -------------------------------------------------
    bar = progress.Progress(len(needs_probe), "probe", verbose=verbose)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = ((root, relative) for relative in needs_probe)
        for number, probed in enumerate(pool.map(_probe, jobs), 1):
            if probed is None:
                report.probe_failed += 1
                bar.advance("(unreadable)")
                continue
            bar.advance(probed["path"])
            relative = probed["path"]
            stat = on_disk[relative]
            key = db.content_key(probed["artist"], probed["album"], probed["title"],
                                 probed["genre_raw"], probed["year"])
            fields = {**probed, "size": stat.st_size, "mtime": stat.st_mtime,
                      "device": stat.st_dev, "inode": stat.st_ino,
                      "content_key": key}
            previous = known.get(relative)

            if previous is None:
                previous_path = _claim(stat, key, probed["duration"],
                                       gone_by_inode, gone_by_key,
                                       claimed, known, report)
                if previous_path is not None:
                    previous = known[previous_path]

            if previous is None:
                report.added += 1
                conn.execute(
                    """INSERT INTO tracks
                       (path, device, inode, artist, album, album_artist, title,
                        genre_raw, year, duration, size, mtime, camelot_key,
                        content_key, missing_since)
                       VALUES (:path, :device, :inode, :artist, :album,
                               :album_artist, :title, :genre_raw, :year,
                               :duration, :size, :mtime, :camelot_key,
                               :content_key, NULL)""", fields)
            else:
                if previous["missing_since"] is not None:
                    report.restored += 1
                elif previous["content_key"] != key:
                    report.changed += 1
                elif previous["path"] == relative:
                    # Bytes moved but the tags did not -- artwork embedded, a
                    # container rewrite. Update stats, leave the mood row alone.
                    report.touched += 1
                # Update in place by id: the row keeps its identity, so the mood
                # table is never rewritten and nothing referencing it breaks.
                conn.execute(
                    """UPDATE tracks SET
                         path=:path, device=:device, inode=:inode,
                         artist=:artist, album=:album, album_artist=:album_artist,
                         title=:title, genre_raw=:genre_raw, year=:year,
                         duration=:duration, size=:size, mtime=:mtime,
                         camelot_key=:camelot_key, content_key=:content_key,
                         missing_since=NULL
                       WHERE id=:id""", {**fields, "id": previous["id"]})

            if number % 2000 == 0:
                conn.commit()
    bar.clear()
    conn.commit()

    # --- deletions --------------------------------------------------------
    report.missing = _mark_missing(conn, disappeared, known, claimed)
    conn.commit()
    if owns_conn:
        conn.close()

    progress.note("done", report.summary())
    return report


def _claim(stat, key: str, duration: float | None,
           gone_by_inode: dict[int, list[str]], gone_by_key: dict[str, list[str]],
           claimed: set[str], known: dict, report: ScanReport) -> str | None:
    """Decide whether a newly-seen file is really a track we already know.

    Inode first. It survives both a plain rename and the rename Music performs
    when a tag edit changes the filename -- the case where content_key has also
    changed and therefore cannot help.

    Duration corroborates it, not size. Inodes are recycled after a delete, so a
    bare inode match could in principle attach an old track's mood row to an
    unrelated new file. Size looks like the obvious guard and is the wrong one:
    editing a tag changes the file's length, which is exactly the case this
    branch exists to catch, so a size check rejects every legitimate match.
    Duration is stable across a tag edit and differs between recordings.

    content_key second, for a file that was rewritten or re-added and so has a
    new inode. That match must be *unambiguous* -- exactly one candidate. Tags
    are not unique across 23k tracks (this library holds 139 genuine duplicate
    tag-sets), and a wrong attribution silently describes the wrong song forever,
    whereas re-tagging costs seconds.
    """
    for path in gone_by_inode.get(stat.st_ino, []):
        if path in claimed:
            continue
        previous = known[path]["duration"]
        if previous is not None and duration is not None \
                and abs(previous - duration) > 1.0:
            continue
        claimed.add(path)
        report.moved += 1
        report.moved_by_inode += 1
        return path

    candidates = [p for p in gone_by_key.get(key, []) if p not in claimed]
    if len(candidates) == 1:
        claimed.add(candidates[0])
        report.moved += 1
        return candidates[0]
    return None


def _mark_missing(conn, disappeared: list[str], known: dict,
                  claimed: set[str]) -> int:
    """Soft-mark whatever was not claimed by a move.

    Soft, never a delete: a track that reappears keeps its tagging, and a row
    that turns out to have moved somewhere we could not match is recoverable
    rather than silently gone.
    """
    missing = 0
    for path in disappeared:
        if path in claimed or known[path]["missing_since"] is not None:
            continue
        conn.execute(
            "UPDATE tracks SET missing_since = datetime('now') WHERE id = ?",
            (known[path]["id"],))
        missing += 1
    return missing
