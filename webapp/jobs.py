"""Background runner for the two long commands.

`scan` takes ~2 minutes and `tag` can take hours, so neither can run inside a
request. They run in a worker thread instead, and the page polls for status.

One job at a time, deliberately. Both commands write to the same SQLite database
and the tagger already saturates the LLM endpoint at its configured concurrency;
letting a second run start would not go faster, it would just make the progress
numbers lie.

Progress is read back out of the database rather than plumbed through from the
worker. `db.counts()` already knows how many tracks are tagged and how many are
stale, which is exactly what a progress bar wants, and deriving it means the
worker needs no callback and the two cannot disagree.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

from moodlib import db, scan, tag


@dataclass
class Job:
    name: str
    started: float
    finished: float | None = None
    error: str | None = None
    result: str = ""
    #: tracks tagged when the job began, so a tagging run can report its own
    #: progress rather than the library's lifetime total.
    baseline: int = 0
    target: int = 0

    @property
    def running(self) -> bool:
        return self.finished is None

    def as_dict(self, conn=None) -> dict:
        payload = {
            "name": self.name,
            "running": self.running,
            "elapsed": (self.finished or time.time()) - self.started,
            "error": self.error,
            "result": self.result,
            "done": 0,
            "target": self.target,
        }
        if self.name == "tag" and self.target and conn is not None:
            payload["done"] = max(0, db.counts(conn)["tagged"] - self.baseline)
        return payload


_lock = threading.Lock()
_current: Job | None = None


def current() -> Job | None:
    return _current


def _run(job: Job, work: Callable[[], str]) -> None:
    global _current
    try:
        job.result = work()
    except Exception as exc:                      # noqa: BLE001 - surfaced in the UI
        job.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        job.finished = time.time()


def start(name: str) -> tuple[bool, str]:
    """Begin a scan or tag run. Returns (started, message)."""
    global _current
    with _lock:
        if _current is not None and _current.running:
            return False, f"{_current.name} is already running"

        conn = db.connect()
        counts = db.counts(conn)
        if name == "scan":
            job = Job("scan", time.time())
            work = lambda: scan.build().summary()
        elif name == "tag":
            pending = (counts["never_tagged"] + counts["stale_content"]
                       + counts["stale_ontology"] + counts["errors"])
            if not pending:
                conn.close()
                return False, "nothing to tag — everything is current"
            job = Job("tag", time.time(), baseline=counts["tagged"], target=pending)
            work = lambda: _describe_tag(tag.run())
        else:
            conn.close()
            return False, f"unknown job {name!r}"
        conn.close()

        _current = job
        threading.Thread(target=_run, args=(job, work), daemon=True).start()
        return True, f"{name} started"


def _describe_tag(result: dict) -> str:
    text = f"tagged {result['tagged']:,}"
    if result["failed"]:
        text += f", failed {result['failed']}"
    return text
