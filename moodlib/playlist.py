"""Stage 4: a selection -> an .m3u8 file you import manually.

Written as `.m3u8` rather than `.m3u`: the extension is the conventional signal
that the file is UTF-8, and this library is full of non-ASCII artist names
(Röyksopp, Sigur Rós, Verschiedene Interpreten). A `.m3u` with UTF-8 bytes inside
is a coin flip on how any given importer decodes it.

Paths are NFC-normalised. macOS hands filenames back in NFD, and an NFD path
written into a playlist may not match the same file the importer looks up by an
NFC name.

Whether they are absolute or relative is a choice -- see `render()`. It matters
more than it sounds: the server reaches the library over NFS and a Mac reaches
the same files over SMB, at different paths, so an absolute path that is correct
on one is broken on the other.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from moodlib import config
from moodlib.query import Scored

#: Characters a filename genuinely cannot carry: the POSIX path separator, the
#: classic macOS one, the Windows reserved set (for playlists copied elsewhere),
#: and control characters. Everything else -- crucially including spaces -- is
#: left alone.
_UNSAFE = re.compile(r'[/\\:*?"<>|\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")


def filename_for(title: str, fallback: str = "playlist") -> str:
    """Turn a title into a filename that still reads like the title.

    This matters more than it looks: Music.app names an imported playlist after
    the *file*, ignoring the `#PLAYLIST` directive inside it. The filename is
    therefore what ends up in the sidebar, so it has to stay human.

    An earlier version slugged this the way a URL would -- spaces to hyphens --
    and "Sunny Electric Dreams" duly appeared in the music library as
    "Sunny-Electric-Dreams". Filenames are not URLs: a space is perfectly legal
    on every filesystem this runs on, and replacing it corrupts the one thing
    the title was generated for.
    """
    cleaned = _UNSAFE.sub("", unicodedata.normalize("NFC", title or ""))
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    # A leading dot hides the file; trailing dots and spaces are stripped by
    # some filesystems anyway, so drop them rather than let the name drift.
    cleaned = cleaned.strip(". ")
    return cleaned[:80].strip() or fallback


def render(tracks: list[Scored], library: Path | None = None,
           path_prefix: str | None = None, title: str = "",
           relative: bool | None = None) -> str:
    """Render the selection as .m3u8 text.

    Three ways to write the paths, because "where is the music" has three
    different answers depending on who reads the file:

        relative        Artist/Album/01 Track.m4a
        path_prefix     /Volumes/Media/Music/Artist/…
        (default)       <LIBRARY_PATH>/Artist/…

    **Relative wins when the file will be read on a different machine.** The
    server sees the library over NFS at /mnt/media/Music and a Mac sees the same
    files over SMB at /Volumes/…; an absolute path written by one is meaningless
    to the other. A relative entry is resolved by the importer against the
    playlist file's own directory, so the same file works on both -- at the cost
    of one rule: **the .m3u8 has to sit in the library root**. Left in
    ~/Downloads it resolves against ~/Downloads and finds nothing.

    `path_prefix` (M3U_PATH_PREFIX) is the other answer to the same problem:
    rewrite the root to whatever the *consuming* machine calls it. That keeps
    the file freely movable, but bakes one particular client's mount point into
    the server's config, so it only suits a single-client setup.
    """
    root = library or config.LIBRARY_PATH
    prefix = config.M3U_PATH_PREFIX if path_prefix is None else path_prefix
    relative = config.M3U_RELATIVE_PATHS if relative is None else relative

    lines = ["#EXTM3U"]
    if title:
        # Honoured by VLC, foobar2000 and friends. Music.app ignores it and uses
        # the filename instead, which is why `filename_for()` exists -- writing
        # both covers either importer.
        lines.append(f"#PLAYLIST:{title}")
    if relative:
        base = ""
    elif prefix:
        base = prefix.rstrip("/") + "/"
    else:
        base = str(root).rstrip("/") + "/"
    for item in tracks:
        seconds = int(round(item.duration or 0))
        artist = item.artist or "Unknown Artist"
        name = item.title or Path(item.path).stem
        lines.append(f"#EXTINF:{seconds},{artist} - {name}")
        lines.append(unicodedata.normalize("NFC", f"{base}{item.path}"))
    return "\n".join(lines) + "\n"


def write(tracks: list[Scored], destination: Path, library: Path | None = None,
          path_prefix: str | None = None, title: str = "",
          relative: bool | None = None) -> Path:
    destination = Path(destination)
    if destination.suffix.lower() not in (".m3u8", ".m3u"):
        destination = destination.with_suffix(".m3u8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # No BOM: Music.app and most importers treat a leading BOM as part of the
    # first line and then fail to match `#EXTM3U`.
    destination.write_text(render(tracks, library, path_prefix, title, relative),
                           encoding="utf-8")
    return destination
