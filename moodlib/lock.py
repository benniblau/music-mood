"""One writer at a time, across processes.

`webapp/jobs.py` already refuses to run two jobs at once, but that lock lives in
one process's memory. On a server there are now several processes that can write
the same database -- the gunicorn service, a `run.py tag` someone starts over
SSH, a systemd timer doing the nightly scan -- and none of them can see the
others' state.

Two of them writing at once is not a hypothetical annoyance. SQLite in WAL mode
takes concurrent *readers* happily but still serialises writers, so the loser
gets `database is locked` somewhere in the middle of a six-hour tagging run,
after it has already spent the time. Worse, both would be marking the same rows
stale and re-tagging them, paying twice for one result.

An advisory file lock is the right size of tool here: it is held by the kernel
against the open file descriptor, so it is released when the process exits for
any reason, including a kill -9 or a machine losing power. Nothing has to clean
up a stale lock file afterwards, which is the failure mode that makes
lock-file-by-convention schemes worse than no lock at all.
"""
from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

from moodlib import config


class Busy(RuntimeError):
    """Another process holds the write lock."""


@contextmanager
def writer(name: str = "library", path: Path | None = None):
    """Hold the exclusive write lock for the duration of the block.

    Raises `Busy` rather than waiting. A scan takes minutes and a full tag run
    takes hours, so a caller that blocked would look hung for the rest of the
    day; the honest answer is to say who has it and let the user decide.
    """
    config.ensure_dirs()
    path = path or config.DATA_DIR / f"{name}.lock"
    handle = open(path, "a+")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Busy(f"another process is already running {name} "
                       f"({holder(path) or 'pid unknown'})") from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield
    finally:
        handle.close()          # releases the lock


def holder(path: Path) -> str:
    """Describe whoever holds the lock, for the error message."""
    try:
        pid = path.read_text().strip()
    except OSError:
        return ""
    return f"pid {pid}" if pid else ""
