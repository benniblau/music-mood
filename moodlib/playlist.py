"""Stage 4: a selection -> an .m3u8 file you import manually.

Written as `.m3u8` rather than `.m3u`: the extension is the conventional signal
that the file is UTF-8, and this library is full of non-ASCII artist names
(Röyksopp, Sigur Rós, Verschiedene Interpreten). A `.m3u` with UTF-8 bytes inside
is a coin flip on how any given importer decodes it.

Paths are absolute and NFC-normalised. macOS hands filenames back in NFD, and an
NFD path written into a playlist may not match the same file the importer looks
up by an NFC name.
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
           path_prefix: str | None = None, title: str = "") -> str:
    root = library or config.LIBRARY_PATH
    prefix = config.M3U_PATH_PREFIX if path_prefix is None else path_prefix

    lines = ["#EXTM3U"]
    if title:
        # Honoured by VLC, foobar2000 and friends. Music.app ignores it and uses
        # the filename instead, which is why `slug()` exists -- writing both
        # covers either importer.
        lines.append(f"#PLAYLIST:{title}")
    for item in tracks:
        seconds = int(round(item.duration or 0))
        artist = item.artist or "Unknown Artist"
        title = item.title or Path(item.path).stem
        lines.append(f"#EXTINF:{seconds},{artist} - {title}")
        # M3U_PATH_PREFIX rewrites the library root for the eventual server
        # deployment, where the same files sit under a different mount point.
        base = prefix.rstrip("/") if prefix else str(root).rstrip("/")
        lines.append(unicodedata.normalize("NFC", f"{base}/{item.path}"))
    return "\n".join(lines) + "\n"


def write(tracks: list[Scored], destination: Path, library: Path | None = None,
          path_prefix: str | None = None, title: str = "") -> Path:
    destination = Path(destination)
    if destination.suffix.lower() not in (".m3u8", ".m3u"):
        destination = destination.with_suffix(".m3u8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # No BOM: Music.app and most importers treat a leading BOM as part of the
    # first line and then fail to match `#EXTM3U`.
    destination.write_text(render(tracks, library, path_prefix, title),
                           encoding="utf-8")
    return destination
