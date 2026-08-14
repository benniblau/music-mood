"""Background runner for the two long commands.

`scan` takes ~2 minutes and `tag` can take hours, so neither can run inside a
request. They run in a worker thread instead, and the page polls for status.

One job at a time, deliberately. Both commands write to the same SQLite database
and the tagger already saturates the LLM endpoint at its configured concurrency;
letting a second run start would not go faster, it would just make the progress
numbers lie.

*How much is done* is read back out of the database, not plumbed through from the
worker. `db.counts()` already knows how many tracks are tagged and how many are
stale, which is exactly what a progress bar wants, and deriving it means the web
UI and `run.py stats` cannot disagree.

*What it is doing right now* cannot come from the database -- there is no row for
"reading tags from 4,586 changed files" -- so the job subscribes to
`moodlib.progress`, the same source the terminal's live line reads. The split is
deliberate: the authoritative numbers stay derived, and only the transient
detail is pushed. A scan spends four phases of very different lengths behind a
single "scanning" label, which on a network share is indistinguishable from a
hang.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

from moodlib import db, progress, scan, tag


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

    # --- what the worker is doing right now --------------------------------
    # Fed by moodlib.progress, which is the same source the terminal's live line
    # reads. Without these the page can only say "scanning, 4m elapsed" for the
    # whole of a scan, which is indistinguishable from a hang -- and a scan has
    # four distinct phases with very different durations.
    #: Last milestone: "reading tags from 4,586 changed files", and so on.
    phase: str = ""
    phase_icon: str = ""
    #: The file or track being handled at this instant.
    item: str = ""
    #: Progress *within the current phase*, which is not the same as progress
    #: through the job -- a scan probes only what changed.
    step_done: int = 0
    step_total: int = 0
    step_unit: str = ""
    rate: float = 0.0
    eta: float | None = None

    @property
    def running(self) -> bool:
        return self.finished is None

    def observe(self, event: dict) -> None:
        """Receive one progress event. Called from the worker thread."""
        if event["type"] == "note":
            self.phase = event["message"]
            self.phase_icon = progress.ICON.get(event["kind"], "")
            # A milestone ends the phase the counters belonged to; leaving them
            # up would show a finished stage's numbers against the new label.
            self.item = ""
            self.step_done = self.step_total = 0
            return
        self.item = event.get("item") or ""
        self.step_done = event.get("done") or 0
        self.step_total = event.get("total") or 0
        self.step_unit = event.get("unit") or ""
        self.rate = event.get("rate") or 0.0
        self.eta = event.get("eta")

    def as_dict(self, conn=None) -> dict:
        payload = {
            "name": self.name,
            "running": self.running,
            "elapsed": (self.finished or time.time()) - self.started,
            "error": self.error,
            "result": self.result,
            "done": 0,
            "target": self.target,
            "phase": self.phase,
            "phase_icon": self.phase_icon,
            "item": self.item,
            "step_done": self.step_done,
            "step_total": self.step_total,
            "step_unit": self.step_unit,
            "rate": round(self.rate, 2),
            "eta": self.eta,
        }
        if self.name == "tag" and self.target and conn is not None:
            payload["done"] = max(0, db.counts(conn)["tagged"] - self.baseline)
        return payload


_lock = threading.Lock()
_current: Job | None = None


def current() -> Job | None:
    """The most recent job, running or not.

    Deliberately not cleared when a run ends: the finished job is how the page
    reports the outcome of a scan or tag that has already completed. Callers must
    therefore check `.running` rather than treating a non-None job as "busy" --
    reading "a job exists" as "a job just finished" is what once made the home
    page reload itself in a loop.
    """
    return _current


def _run(job: Job, work: Callable[[], str]) -> None:
    # The worker reports through moodlib.progress, exactly as it does for the
    # terminal; the job just listens. Cleared in `finally` so a finished job
    # cannot keep mutating itself if a stray worker thread outlives it.
    progress.set_watcher(job.observe)
    try:
        job.result = work()
    except Exception as exc:                      # noqa: BLE001 - surfaced in the UI
        job.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        progress.set_watcher(None)
        job.item = ""
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
