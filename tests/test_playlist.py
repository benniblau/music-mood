"""M3U rendering.

The interesting part is the paths. The same files are reached at different mount
points by different machines -- /mnt/media/Music over NFS on the server,
/Volumes/…/Music over SMB on a Mac -- so an absolute path that is correct where
the playlist was generated is broken where it is imported. That is not a corner
case; it is the normal situation once the app runs on a server.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from moodlib import playlist
from moodlib.query import Scored

LIBRARY = Path("/mnt/media/Music")


def _track(path: str, artist: str = "Pendulum", title: str = "Slam",
           duration: float = 254.4) -> Scored:
    return Scored(path=path, artist=artist, title=title, album="Hold Your Colour",
                  year="2005", duration=duration, score=1.0, parts={})


def _paths(text: str) -> list[str]:
    return [line for line in text.splitlines()
            if line and not line.startswith("#")]


def test_absolute_paths_use_the_library_root():
    text = playlist.render([_track("Pendulum/Hold Your Colour/03 Slam.m4a")],
                           library=LIBRARY, path_prefix="", relative=False)
    assert _paths(text) == ["/mnt/media/Music/Pendulum/Hold Your Colour/03 Slam.m4a"]


def test_relative_paths_carry_no_root_at_all():
    """The whole point: nothing machine-specific survives into the file."""
    text = playlist.render([_track("Pendulum/Hold Your Colour/03 Slam.m4a")],
                           library=LIBRARY, relative=True)
    assert _paths(text) == ["Pendulum/Hold Your Colour/03 Slam.m4a"]
    assert "/mnt" not in text and "Volumes" not in text


def test_relative_beats_a_configured_prefix():
    # Both exist to solve the same mismatch, so they must not compose into
    # "/Volumes/…/Artist/…" with a relative flag set and quietly ignored.
    text = playlist.render([_track("Air/Moon Safari/01 La Femme d'Argent.m4a")],
                           library=LIBRARY, path_prefix="/Volumes/NAS/Music",
                           relative=True)
    assert _paths(text) == ["Air/Moon Safari/01 La Femme d'Argent.m4a"]


def test_prefix_rewrites_the_root_when_not_relative():
    text = playlist.render([_track("Air/Moon Safari/01 La Femme d'Argent.m4a")],
                           library=LIBRARY, path_prefix="/Volumes/NAS/Music/",
                           relative=False)
    assert _paths(text) == ["/Volumes/NAS/Music/Air/Moon Safari/01 La Femme d'Argent.m4a"]


def test_the_header_survives_the_track_loop():
    # `title` was once reused as the loop variable for each track's title.
    text = playlist.render([_track("a/b/c.m4a"), _track("d/e/f.m4a")],
                           library=LIBRARY, title="Neon Rain Drive", relative=True)
    assert text.splitlines()[1] == "#PLAYLIST:Neon Rain Drive"


def test_extinf_lines_pair_with_paths():
    tracks = [_track("a/b/c.m4a", "Bonobo", "Kiara", 302.7),
              _track("d/e/f.m4a", "Air", "Run", 211.2)]
    lines = playlist.render(tracks, library=LIBRARY, relative=True).splitlines()
    assert lines[1] == "#EXTINF:303,Bonobo - Kiara"
    assert lines[2] == "a/b/c.m4a"
    assert lines[3] == "#EXTINF:211,Air - Run"


def test_a_relative_playlist_resolves_from_the_library_root(tmp_path):
    """Saved into the library, every entry must point at a real file.

    This is the rule the format imposes and the one a user will trip over: the
    entries resolve against the *playlist file's* directory, not the library's.
    """
    library = tmp_path / "Music"
    song = library / "Pendulum" / "Hold Your Colour" / "03 Slam.m4a"
    song.parent.mkdir(parents=True)
    song.write_bytes(b"not really audio")

    written = playlist.write([_track("Pendulum/Hold Your Colour/03 Slam.m4a")],
                             library / "Late Night.m3u8", library=library,
                             relative=True)
    for entry in _paths(written.read_text()):
        assert (written.parent / entry).exists(), entry

    # And the failure mode, stated so it cannot be mistaken for a bug later:
    # the same file one directory away resolves to nothing.
    elsewhere = tmp_path / "Downloads"
    elsewhere.mkdir()
    for entry in _paths(written.read_text()):
        assert not (elsewhere / entry).exists()


@pytest.mark.parametrize("title,expected", [
    ("Sunny Electric Dreams", "Sunny Electric Dreams"),   # spaces are legal
    ("AC/DC Forever", "ACDC Forever"),                    # separator is not
    ("  ..trailing.. ", "trailing"),
    ("", "playlist"),
])
def test_filename_keeps_the_title_readable(title, expected):
    # Music.app names the imported playlist after the file, ignoring #PLAYLIST,
    # so slugging this URL-style put "Sunny-Electric-Dreams" in the sidebar.
    assert playlist.filename_for(title) == expected
