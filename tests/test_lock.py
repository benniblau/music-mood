"""The cross-process write lock.

Once the app runs as a service, several processes on the box can write the same
database -- the gunicorn service, the nightly timer, and whatever someone types
over SSH. SQLite in WAL mode serialises writers, so the loser of that race finds
out with `database is locked` somewhere in the middle of a six-hour tagging run,
having already spent the six hours.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from moodlib import lock

ROOT = Path(__file__).resolve().parent.parent


def test_the_lock_excludes_a_second_holder(tmp_path):
    path = tmp_path / "library.lock"
    with lock.writer(path=path):
        # Same process, so this has to be a second file descriptor to be a real
        # test -- flock is per-open-file, and re-locking the same handle would
        # succeed and prove nothing.
        with pytest.raises(lock.Busy):
            _acquire_in_a_subprocess(path)


def test_the_lock_is_free_again_afterwards(tmp_path):
    path = tmp_path / "library.lock"
    with lock.writer(path=path):
        pass
    with lock.writer(path=path):       # would raise if the first leaked
        pass


def test_a_killed_holder_does_not_wedge_the_lock(tmp_path):
    """The reason this is a flock and not a lock file by convention.

    A lock file that a process has to delete on the way out is worse than no
    lock: kill -9 the tagger, or lose power mid-run, and every later run refuses
    to start until someone works out which file to remove. The kernel releases a
    flock when the fd closes, whatever closed it.
    """
    path = tmp_path / "library.lock"
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {str(ROOT)!r})
            from moodlib import lock
            with lock.writer(path=__import__("pathlib").Path({str(path)!r})):
                print("held", flush=True)
                time.sleep(60)
        """)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(lock.Busy):
            with lock.writer(path=path):
                pass
        holder.kill()
        holder.wait(timeout=10)
    finally:
        if holder.poll() is None:                     # pragma: no cover
            holder.kill()

    with lock.writer(path=path):                      # released by the kernel
        pass


def test_the_error_names_who_holds_it(tmp_path):
    path = tmp_path / "library.lock"
    with lock.writer(path=path):
        with pytest.raises(lock.Busy, match=r"pid \d+"):
            _acquire_in_a_subprocess(path)


def _acquire_in_a_subprocess(path: Path) -> None:
    """Try to take the lock from another process; raise lock.Busy if it is held."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(ROOT)!r})
            import pathlib
            from moodlib import lock
            try:
                with lock.writer(path=pathlib.Path({str(path)!r})):
                    pass
            except lock.Busy as exc:
                print(exc)
                sys.exit(9)
        """)], capture_output=True, text=True, timeout=30)
    if result.returncode == 9:
        raise lock.Busy(result.stdout.strip())
    assert result.returncode == 0, result.stderr
