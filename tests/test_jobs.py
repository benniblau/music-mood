"""What the web UI can say about a running job.

The counts stay derived from the database, so the page and `run.py stats` cannot
disagree. But no row records *"reading tags from 4,586 changed files"*, and
without that a scan is a single "scanning" label for four phases of wildly
different length — which on a network share is indistinguishable from a hang.
That detail is pushed from the worker through `moodlib.progress`.
"""
from __future__ import annotations

import io
import time

import pytest

from moodlib import progress
from webapp.jobs import Job


@pytest.fixture(autouse=True)
def _no_leftover_watcher():
    yield
    progress.set_watcher(None)


@pytest.fixture
def job():
    job = Job("scan", started=0.0)
    progress.set_watcher(job.observe)
    return job


def _bar(total, kind="probe", **kw):
    return progress.Progress(total, kind, stream=io.StringIO(), **kw)


def test_a_milestone_becomes_the_phase(job):
    progress.note("probe", "reading tags from 4,586 changed files")
    assert job.phase == "reading tags from 4,586 changed files"
    assert job.phase_icon == progress.ICON["probe"]


def test_the_current_item_and_its_counters_come_through(job):
    _bar(4586).advance("Pendulum/Hold Your Colour/03 Slam.m4a")
    assert job.item == "Pendulum/Hold Your Colour/03 Slam.m4a"
    assert (job.step_done, job.step_total) == (1, 4586)
    assert job.rate > 0
    assert job.eta is not None


def test_a_new_phase_clears_the_previous_one_s_counters(job):
    _bar(4586).advance("Pendulum/Hold Your Colour/03 Slam.m4a")
    progress.note("db", "recorded file identity for 18,389 tracks")

    # Otherwise the page shows a finished stage's numbers under a new label,
    # which reads as progress that jumped backwards.
    assert job.item == ""
    assert (job.step_done, job.step_total) == (0, 0)


def test_a_countless_stage_still_reports_motion(job):
    """The library walk cannot know its total until it has finished."""
    bar = _bar(0, kind="scan")
    bar.advance("Artist/Album/01 Track.m4a")
    assert job.step_total == 0       # no total to count towards
    assert job.step_done == 1        # a number that moves is the whole point
    assert job.eta is None


def test_events_are_throttled_rather_than_one_per_item(job):
    """23,000 items must not mean 23,000 updates.

    The poller asks every couple of seconds; anything faster than a few hertz is
    work nobody sees. This is why step_done can lag the true count by a fraction
    of a second, which is fine for a display and would not be for a total.
    """
    bar = _bar(0, kind="scan")
    for n in range(200):
        bar.advance(f"Artist/Album/{n:02d} Track.m4a")
    assert bar.done == 200
    assert job.step_done == 1        # all 200 landed inside one throttle window

    time.sleep(0.25)                 # just past the 0.2s window
    bar.advance("Artist/Album/99 Track.m4a")
    assert job.step_done == 201      # and it catches up to the true count


def test_the_watcher_never_breaks_the_run():
    """A failing display must not take down a six-hour tagging run."""
    def explode(event):
        raise RuntimeError("the UI is on fire")

    progress.set_watcher(explode)
    progress.note("tag", "still fine")          # must not raise
    _bar(10).advance("also fine")               # must not raise


def test_the_cli_pays_nothing_when_nobody_is_watching():
    progress.set_watcher(None)
    bar = _bar(10)
    bar.advance("x")                            # no watcher, no event, no error
    assert bar.done == 1


def test_status_payload_carries_the_detail(job):
    progress.note("probe", "reading tags from 4,586 changed files")
    _bar(4586).advance("Air/Moon Safari/01 La Femme d'Argent.m4a")
    payload = job.as_dict()
    for key in ("phase", "phase_icon", "item", "step_done", "step_total",
                "step_unit", "rate", "eta"):
        assert key in payload, key
    assert payload["item"].endswith("La Femme d'Argent.m4a")
