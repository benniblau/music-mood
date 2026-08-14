"""The scan state machine, exercised against real files in a temp directory.

Every case here corresponds to a row of the table in scan.py's docstring. The
move case in particular is not hypothetical: the first real re-scan of this
library found 456 files renamed underneath it by another process, and an earlier
version of the code turned all of them into phantom rows.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from moodlib import config, db, scan

# ffmpeg is only needed to *synthesise* the fixtures; the scanner itself reads
# them with mutagen and has no external binary dependency.
pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed (fixtures only)")


def _make_track(root: Path, relative: str, *, title: str, artist: str = "Tester",
                album: str = "Album") -> Path:
    """A real, tiny, tagged m4a the scanner can read."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
         "-t", "1", "-c:a", "aac",
         "-metadata", f"title={title}",
         "-metadata", f"artist={artist}",
         "-metadata", f"album={album}",
         str(path)],
        capture_output=True, check=True)
    return path


@pytest.fixture()
def library(tmp_path, monkeypatch):
    root = tmp_path / "Music"
    root.mkdir()
    monkeypatch.setattr(config, "LIBRARY_PATH", root)
    # These libraries hold a handful of tracks, so removing even one blows past
    # a percentage threshold tuned for 23,000. Disable the rail by default and
    # let the one test that is actually about it set its own value.
    monkeypatch.setattr(config, "SCAN_MISSING_ABORT_PCT", 100.0)
    return root


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.sqlite3")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    connection = db.connect(tmp_path / "test.sqlite3")
    yield connection
    connection.close()


def _tag(conn, path: str) -> None:
    """Pretend the tagger ran, so we can prove a mood row survives a move."""
    row = conn.execute("SELECT id, content_key FROM tracks WHERE path = ?",
                       (path,)).fetchone()
    db.write_mood(conn, row["id"], {
        "energy": 0.5, "confidence": 2, "model": "test",
        "ontology_version": 1, "content_key": row["content_key"],
        "tagged_at": "now", "error": None,
    })
    conn.commit()


def test_new_track_is_added(library, conn):
    _make_track(library, "Tester/Album/01 One.m4a", title="One")
    report = scan.build(root=library, conn=conn)
    assert report.added == 1
    assert conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 1


def test_unchanged_rescan_probes_nothing(library, conn):
    _make_track(library, "Tester/Album/01 One.m4a", title="One")
    scan.build(root=library, conn=conn)
    report = scan.build(root=library, conn=conn)
    assert (report.added, report.changed, report.touched) == (0, 0, 0)
    assert report.unchanged == 1


def test_touching_a_file_does_not_restage_it_for_tagging(library, conn):
    path = _make_track(library, "Tester/Album/01 One.m4a", title="One")
    scan.build(root=library, conn=conn)
    _tag(conn, "Tester/Album/01 One.m4a")

    # Same tags, different bytes and mtime -- an artwork embed or a container
    # rewrite. Re-tagging here would be pure waste.
    os.utime(path, (path.stat().st_atime + 10, path.stat().st_mtime + 10))
    report = scan.build(root=library, conn=conn)

    assert report.touched == 1
    assert report.changed == 0
    assert db.stale_tracks(conn) == []


def test_editing_a_tag_restages_the_track(library, conn):
    _make_track(library, "Tester/Album/01 One.m4a", title="One")
    scan.build(root=library, conn=conn)
    _tag(conn, "Tester/Album/01 One.m4a")

    _make_track(library, "Tester/Album/01 One.m4a", title="One (Remastered)")
    report = scan.build(root=library, conn=conn)

    assert report.changed == 1
    assert [r["path"] for r in db.stale_tracks(conn)] == ["Tester/Album/01 One.m4a"]


def test_rename_carries_the_mood_row_to_the_new_path(library, conn):
    old = _make_track(library, "Tester/Album/01 One.m4a", title="One")
    scan.build(root=library, conn=conn)
    _tag(conn, "Tester/Album/01 One.m4a")

    old.rename(old.with_name("1-01 One.m4a"))
    report = scan.build(root=library, conn=conn)

    assert report.moved == 1
    assert report.missing == 0
    # The mood row never moved -- it is keyed by the track's stable id, and the
    # track kept that id across the rename.
    linked = conn.execute(
        "SELECT t.path FROM moods m JOIN tracks t ON t.id = m.track_id").fetchall()
    assert [r["path"] for r in linked] == ["Tester/Album/1-01 One.m4a"]
    # ...and the moved track must not look stale, or the move bought nothing.
    assert db.stale_tracks(conn) == []


def test_rename_of_an_untagged_track_leaves_no_phantom(library, conn):
    # The regression that cost 837 rows on the first real re-scan: move
    # detection used to require an existing mood row, so untagged renames became
    # a permanently-missing duplicate of a track that was right there.
    old = _make_track(library, "Tester/Album/01 One.m4a", title="One")
    scan.build(root=library, conn=conn)

    old.rename(old.with_name("1-01 One.m4a"))
    report = scan.build(root=library, conn=conn)

    assert report.moved == 1
    rows = list(conn.execute("SELECT path, missing_since FROM tracks"))
    assert len(rows) == 1
    assert rows[0]["missing_since"] is None


def test_inode_disambiguates_identically_tagged_renames(library, conn):
    # Two files with the same tags both get renamed. content_key cannot tell them
    # apart, but their inodes can -- so both are tracked rather than re-added.
    a = _make_track(library, "Tester/Album/01 One.m4a", title="Same")
    b = _make_track(library, "Tester/Album/02 Two.m4a", title="Same")
    scan.build(root=library, conn=conn)

    a.rename(a.with_name("11 One.m4a"))
    b.rename(b.with_name("12 Two.m4a"))
    report = scan.build(root=library, conn=conn)

    assert report.moved_by_inode == 2
    assert report.added == 0
    assert conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 2


def test_ambiguous_match_is_not_guessed_when_the_inode_is_gone(library, conn):
    # The genuinely undecidable case: two tracks share a content_key AND both
    # were rewritten, so neither inode survives. Guessing which is which risks
    # describing the wrong song forever; re-tagging costs seconds. So: treat as
    # new rather than pick one.
    _make_track(library, "Tester/Album/01 One.m4a", title="Same")
    _make_track(library, "Tester/Album/02 Two.m4a", title="Same")
    scan.build(root=library, conn=conn)

    for old, new in (("01 One.m4a", "11 One.m4a"), ("02 Two.m4a", "12 Two.m4a")):
        source = library / "Tester" / "Album" / old
        # Copy-then-delete gives the replacement a fresh inode, which is what
        # Music's add-back does and what a restore from backup does.
        shutil.copy2(source, source.with_name(new))
        source.unlink()

    report = scan.build(root=library, conn=conn)

    assert report.moved_by_inode == 0
    assert report.moved == 0
    assert {r["path"] for r in db.stale_tracks(conn)} == {
        "Tester/Album/11 One.m4a", "Tester/Album/12 Two.m4a"}


def test_tag_edit_that_also_renames_the_file_keeps_the_track(library, conn):
    """The case a path- or tags-keyed schema cannot survive.

    Music renames files from their tags, so editing a title changes the path AND
    the content_key in one step. With neither of those stable, the track is
    indistinguishable from a deletion plus an unrelated new file: it loses its
    row, its id, and anything referencing it. The inode is the only thing that
    survives, which is why identity resolution consults it first.
    """
    path = _make_track(library, "Tester/Album/01 One.m4a", title="One")
    scan.build(root=library, conn=conn)
    original_id = conn.execute("SELECT id FROM tracks").fetchone()["id"]
    _tag(conn, "Tester/Album/01 One.m4a")

    # Edit the tags in place and then rename to match, which is what Music does.
    # "In place" is load-bearing: mutagen (and Music, for a file track) rewrites
    # the atoms inside the existing file, so the inode is preserved. Writing the
    # new bytes into the same open file reproduces that; os.replace() would not,
    # because it swaps in a different inode entirely.
    retagged = path.with_name("tmp.m4a")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
         "-t", "1", "-c:a", "aac", "-metadata", "title=One (Remastered)",
         "-metadata", "artist=Tester", "-metadata", "album=Album", str(retagged)],
        capture_output=True, check=True)
    payload = retagged.read_bytes()
    retagged.unlink()
    with open(path, "wb") as handle:      # truncate + rewrite, same inode
        handle.write(payload)
    path.rename(path.with_name("01 One (Remastered).m4a"))

    report = scan.build(root=library, conn=conn)

    rows = list(conn.execute("SELECT id, path, missing_since FROM tracks"))
    assert len(rows) == 1, "the track was duplicated instead of followed"
    assert rows[0]["id"] == original_id, "the track lost its identity"
    assert rows[0]["path"] == "Tester/Album/01 One (Remastered).m4a"
    assert rows[0]["missing_since"] is None
    assert report.missing == 0
    # Its tags really did change, so it must be re-tagged -- but as the same
    # track, not as a stranger.
    assert [r["id"] for r in db.stale_tracks(conn)] == [original_id]


def test_deletion_is_soft(library, conn):
    path = _make_track(library, "Tester/Album/01 One.m4a", title="One")
    _make_track(library, "Tester/Album/02 Two.m4a", title="Two")
    scan.build(root=library, conn=conn)

    path.unlink()
    report = scan.build(root=library, conn=conn)

    assert report.missing == 1
    row = conn.execute(
        "SELECT missing_since FROM tracks WHERE path = ?",
        ("Tester/Album/01 One.m4a",)).fetchone()
    assert row is not None and row["missing_since"] is not None


def test_restored_file_comes_back(library, conn):
    path = _make_track(library, "Tester/Album/01 One.m4a", title="One")
    _make_track(library, "Tester/Album/02 Two.m4a", title="Two")
    scan.build(root=library, conn=conn)
    backup = path.read_bytes()
    path.unlink()
    scan.build(root=library, conn=conn)

    path.write_bytes(backup)
    report = scan.build(root=library, conn=conn)

    assert report.restored == 1
    row = conn.execute("SELECT missing_since FROM tracks WHERE path = ?",
                       ("Tester/Album/01 One.m4a",)).fetchone()
    assert row["missing_since"] is None


def test_mass_disappearance_aborts_without_touching_the_database(library, conn,
                                                                 monkeypatch):
    # An unmounted NAS looks exactly like a library wipe from here.
    monkeypatch.setattr(config, "SCAN_MISSING_ABORT_PCT", 10.0)
    for n in range(5):
        _make_track(library, f"Tester/Album/0{n} T{n}.m4a", title=f"T{n}")
    scan.build(root=library, conn=conn)
    for track in library.rglob("*.m4a"):
        track.unlink()

    with pytest.raises(SystemExit, match="refusing to scan"):
        scan.build(root=library, conn=conn)

    still_present = conn.execute(
        "SELECT COUNT(*) FROM tracks WHERE missing_since IS NULL").fetchone()[0]
    assert still_present == 5


def test_mass_rename_is_not_mistaken_for_mass_deletion(library, conn):
    """The rail must measure files lost from the volume, not paths churned.

    A library reorganisation renames thousands of paths while the file count
    barely moves. A churn-based rail refuses that scan, which is the wrong call:
    the files are all there, and inode matching would have followed every one.
    Seen live -- 5,934 paths vanished in a single pass while 22,966 of 22,987
    files were still present.
    """
    monkeypatch_pct = 10.0
    import moodlib.config as cfg
    original = cfg.SCAN_MISSING_ABORT_PCT
    cfg.SCAN_MISSING_ABORT_PCT = monkeypatch_pct
    try:
        for n in range(10):
            _make_track(library, f"Tester/Album/{n:02d} T{n}.m4a", title=f"T{n}")
        scan.build(root=library, conn=conn)

        for track in sorted(library.rglob("*.m4a")):
            track.rename(track.with_name(f"renamed-{track.name}"))

        report = scan.build(root=library, conn=conn)   # must not raise

        assert report.moved_by_inode == 10
        assert report.missing == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM tracks WHERE missing_since IS NULL"
        ).fetchone()[0] == 10
    finally:
        cfg.SCAN_MISSING_ABORT_PCT = original


def test_untagged_file_falls_back_to_its_path(library, conn):
    # ~1% of this library carries no usable tags at all. Blanks reaching the
    # model are an invitation to hallucinate, and the layout is strictly
    # Artist/Album/Track, so the path is real information.
    path = library / "Pendulum" / "Hold Your Colour" / "01 Prelude.m4a"
    path.parent.mkdir(parents=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
         "-t", "1", "-c:a", "aac", str(path)],
        capture_output=True, check=True)

    scan.build(root=library, conn=conn)
    row = conn.execute("SELECT artist, album, title FROM tracks").fetchone()
    assert (row["artist"], row["album"], row["title"]) == (
        "Pendulum", "Hold Your Colour", "Prelude")
