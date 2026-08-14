"""Web UI invariants.

These guard one specific bug, because it was invisible in code review and very
visible in the browser: the home page reloading itself forever.

The runner keeps the last job around after it ends, so `/jobs/status` reports
`{idle: false, running: false}` for the rest of the server's life. The page used
to read that as "a job just finished, pick up the new numbers" and reload -- on
every single page load, from then on. The two tests below pin the two halves of
that: the server keeps reporting a finished job (by design), and the client only
reloads for a transition it actually watched happen.
"""
from __future__ import annotations

import re
from pathlib import Path

from webapp.jobs import Job

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "webapp" / "templates" / "index.html"


def test_a_finished_job_is_still_reported():
    job = Job("scan", started=0.0)
    assert job.running

    job.finished, job.result = 12.0, "added 3"
    payload = job.as_dict()
    assert payload["running"] is False
    assert payload["result"] == "added 3"      # the outcome is what the page shows
    assert payload["elapsed"] == 12.0


def test_the_page_only_reloads_on_a_transition_it_saw():
    script = INDEX.read_text()
    assert "location.reload()" in script, "the page still reloads after a job"

    # Every reload must sit under the flag that records having seen the job
    # running. An unguarded one is the flicker, exactly as it shipped.
    for match in re.finditer(r"location\.reload\(\)", script):
        guard = script.rfind("if (sawRunning)", 0, match.start())
        assert guard != -1 and "\n\n" not in script[guard:match.start()], (
            "reload is not guarded by sawRunning — this is the flicker bug")
