"""Progress reporting.

The failure this guards against is silent and annoying rather than dangerous: a
carriage-return status line written into a redirected log turns a six-hour run
into one unreadable smear, and nobody notices until they go looking for what
happened. So the TTY / not-a-TTY split is asserted rather than assumed.
"""
from __future__ import annotations

import io

from moodlib import progress


class FakeStream(io.StringIO):
    """A StringIO that can claim to be a terminal."""

    def __init__(self, tty: bool):
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_redirected_output_never_uses_carriage_returns():
    stream = FakeStream(tty=False)
    bar = progress.Progress(100, "probe", every=25, stream=stream)
    for n in range(100):
        bar.advance(f"file-{n}.m4a")
    bar.close("done")
    written = stream.getvalue()
    assert "\r" not in written
    assert "\033" not in written, "no ANSI escapes in a log file"


def test_redirected_output_emits_periodic_milestones():
    stream = FakeStream(tty=False)
    bar = progress.Progress(100, "probe", every=25, interval=1e9, stream=stream)
    for n in range(100):
        bar.advance(f"file-{n}.m4a")
    lines = [l for l in stream.getvalue().splitlines() if l.strip()]
    # The first item reports immediately -- a long job should show it has
    # started rather than sitting silent until the first milestone. After that,
    # every 25th: 1, 26, 51, 76, and a final line on completion.
    assert len(lines) == 5
    assert "1/100" in lines[0]
    assert "100/100" in lines[-1]


def test_live_output_rewrites_one_line():
    stream = FakeStream(tty=True)
    bar = progress.Progress(10, "tag", stream=stream)
    for n in range(10):
        bar.advance(f"Artist {n} — Title {n}")
    written = stream.getvalue()
    assert "\r" in written
    assert written.count("\n") == 0, "the live line must not scroll"


def test_verbose_prints_every_item_even_on_a_tty():
    stream = FakeStream(tty=True)
    bar = progress.Progress(5, "tag", verbose=True, stream=stream)
    for n in range(5):
        bar.advance(f"track-{n}")
    lines = [l for l in stream.getvalue().splitlines() if l.strip()]
    assert len(lines) == 5
    assert "\r" not in stream.getvalue()
    assert all(f"track-{n}" in lines[n] for n in range(5))


def test_note_does_not_get_swallowed_by_the_live_line():
    stream = FakeStream(tty=True)
    bar = progress.Progress(10, "tag", stream=stream)
    bar.advance("something")
    bar.note("warn", "a batch failed")
    assert "a batch failed\n" in stream.getvalue()


def test_long_paths_keep_their_tail():
    # The filename is the informative end of a path, so truncation drops the
    # front. Losing the filename would defeat the point of showing it at all.
    fitted = progress._fit("Artist/A Very Long Album Title/07 The Track.m4a", 20)
    assert fitted.endswith("The Track.m4a")
    assert len(fitted) == 20


def test_short_text_is_untouched():
    assert progress._fit("short.m4a", 40) == "short.m4a"


def test_durations_read_naturally():
    assert progress._duration(45) == "45s"
    assert progress._duration(600) == "10m"
    assert progress._duration(9000) == "2.5h"
