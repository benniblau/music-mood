"""Live progress reporting for the long-running commands.

Two audiences, and they want opposite things:

* A terminal wants to see *motion* -- which file is being touched right now --
  without 23,000 lines scrolling past. It gets one line, rewritten in place.
* A redirected log (`nohup run.py tag --all > tag.log`) wants a readable record.
  Carriage returns turn that into an unreadable smear, so it gets periodic
  milestone lines instead.

Which one you get is decided by whether stderr is a TTY, not by a flag, because
getting it wrong is invisible until you go looking at the log. `--verbose` opts
into one line per item in either mode, for when you genuinely want the firehose.
"""
from __future__ import annotations

import shutil
import sys
import time

#: Emoji are used as line-level markers, one per kind of event, so a log can be
#: skimmed for the interesting lines. They are not decoration on every word.
ICON = {
    "scan": "📂", "probe": "🔍", "tag": "🎨", "query": "💭", "playlist": "🎧",
    "stats": "📊", "done": "✅", "warn": "⚠️ ", "error": "❌", "info": "•",
    "added": "➕", "changed": "✏️ ", "moved": "🔀", "missing": "👻",
    "restored": "♻️ ", "touched": "👆", "unchanged": "😴", "track": "🎵",
    "db": "🗄️ ", "llm": "🤖", "library": "💿", "time": "⏱️ ",
}


def _fit(text: str, width: int) -> str:
    """Trim to width, keeping the tail -- the filename is the informative end."""
    if width <= 1 or len(text) <= width:
        return text
    return "…" + text[-(width - 1):]


def _duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def note(kind: str, message: str) -> None:
    """A one-off line that is never overwritten."""
    sys.stderr.write(f"{ICON.get(kind, ICON['info'])} {message}\n")
    sys.stderr.flush()


class Progress:
    """Counter with a live current-item display.

    `interval` and `every` control the redirected-log cadence: a milestone is
    printed when either threshold is crossed, so a fast stage stays terse and a
    slow one still shows signs of life.
    """

    def __init__(self, total: int, kind: str, unit: str = "files", *,
                 verbose: bool = False, interval: float = 30.0,
                 every: int = 500, stream=None):
        self.total = max(total, 0)
        self.kind = kind
        self.unit = unit
        self.verbose = verbose
        self.interval = interval
        self.every = every
        self.stream = stream or sys.stderr
        self.done = 0
        self.started = time.time()
        self._last_emit = 0.0
        self._last_count = 0
        self._live = self.stream.isatty() and not verbose
        self._dirty = False

    # -- reporting ---------------------------------------------------------
    def _rate(self) -> float:
        elapsed = time.time() - self.started
        return self.done / elapsed if elapsed > 0 else 0.0

    def _suffix(self) -> str:
        rate = self._rate()
        parts = [f"{self.done}/{self.total}" if self.total else str(self.done)]
        if rate > 0:
            parts.append(f"{rate:.1f}/s")
            if self.total:
                parts.append(f"{ICON['time']}{_duration((self.total - self.done) / rate)} left")
        return "  ".join(parts)

    def advance(self, item: str = "", count: int = 1) -> None:
        self.done += count
        now = time.time()

        if self.verbose:
            self.stream.write(f"{ICON.get(self.kind, '•')} {self._suffix()}  {item}\n")
            self.stream.flush()
            return

        if self._live:
            # Rewrite one line. Throttled to ~12 Hz: any faster is invisible and
            # just burns syscalls on a 23,000-item run.
            if now - self._last_emit < 0.08 and self.done < self.total:
                return
            width = shutil.get_terminal_size((100, 24)).columns
            head = f"{ICON.get(self.kind, '•')} {self._suffix()}  "
            line = head + _fit(item, max(0, width - len(head) - 1))
            self.stream.write("\r\033[2K" + line)
            self.stream.flush()
            self._last_emit = now
            self._dirty = True
            return

        # Redirected: milestones only.
        if (self.done - self._last_count >= self.every
                or now - self._last_emit >= self.interval
                or self.done == self.total):
            self.stream.write(f"{ICON.get(self.kind, '•')} {self._suffix()}  {item}\n")
            self.stream.flush()
            self._last_emit = now
            self._last_count = self.done

    def note(self, kind: str, message: str) -> None:
        """Emit a standalone line without losing the live line."""
        self.clear()
        self.stream.write(f"{ICON.get(kind, ICON['info'])} {message}\n")
        self.stream.flush()

    def clear(self) -> None:
        if self._dirty:
            self.stream.write("\r\033[2K")
            self.stream.flush()
            self._dirty = False

    def close(self, summary: str = "") -> None:
        self.clear()
        if summary:
            elapsed = _duration(time.time() - self.started)
            self.stream.write(
                f"{ICON['done']} {summary}  ({elapsed}, {self._rate():.2f}/s)\n")
            self.stream.flush()

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc) -> None:
        self.clear()
