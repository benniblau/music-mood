"""Gunicorn settings.

Every value comes from `.env` through `moodlib.config`, so the invariant holds
here too: one module reads the environment and everything else asks it. Nothing
in this file is a literal that `.env` cannot change, except the worker count --
which is not a tuning knob at all, for the reason below.

**One worker, always.** The scan/tag runner keeps its state in module memory
(`webapp/jobs.py`): which job is running, when it started, what it was told to
do. Two workers would each have their own copy, so the POST that starts a scan
would land on one process and the status polls on the other, which would answer
"nothing is running" for the entire run -- and both could start a scan at once,
because the mutex guarding that is a `threading.Lock`, not a system-wide one.
Concurrency comes from threads inside the one process instead, which is the
right shape anyway: this app's requests are a template render plus a SQLite
read, and the slow one (cover art off a network mount) is I/O-bound.

That is a real ceiling, not a temporary shortcut. Making the runner
multi-process safe means moving job state into the database, and the cost of
not doing it is a household-scale app that serves one household.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Imported under another name on purpose: gunicorn reads this module's globals
# as settings and `config` is one of its own (the -c flag), so binding that name
# here makes it reject the whole file with "Not a string".
from moodlib import config as settings  # noqa: E402

chdir = str(ROOT)
bind = f"{settings.WEB_HOST}:{settings.WEB_PORT}"
proc_name = "music-mood"

# See the module docstring. Do not raise this.
workers = 1
worker_class = "gthread"
threads = settings.WEB_THREADS

#: Derived from LLM_TIMEOUT, not configured -- gunicorn's 30s default is well
#: under the 240s a mood translation is allowed to spend waiting on the model,
#: so the default would kill the worker mid-call and return 502 for a request
#: that was about to succeed.
timeout = settings.web_request_timeout()

#: Scanning and tagging run in daemon threads, so a restart drops whatever was
#: in flight rather than waiting hours for it. That is safe by construction:
#: tagging commits every batch and resumes where it stopped, and a scan re-runs
#: from scratch in minutes. Waiting would just make `systemctl restart` hang.
graceful_timeout = 30

keepalive = 5

#: Journald captures stdout/stderr, so both logs go there rather than to files
#: nothing rotates. Access logging is off unless asked for -- see WEB_ACCESS_LOG.
accesslog = "-" if settings.WEB_ACCESS_LOG else None
errorlog = "-"
loglevel = "info"


def on_starting(server) -> None:
    """Say what this is bound to, and object to the two footguns worth naming."""
    server.log.info("music-mood → http://%s (%d thread%s)",
                    bind, threads, "" if threads == 1 else "s")
    server.log.info("library %s · database %s", settings.LIBRARY_PATH, settings.DB_PATH)

    if settings.WEB_HOST not in ("127.0.0.1", "localhost", "::1"):
        server.log.warning(
            "bound beyond loopback — this app has no accounts and no "
            "authentication, so anything that can reach %s can retag the "
            "library", bind)
    if settings.WEB_SECRET_KEY == "music-mood-local":
        server.log.warning(
            "WEB_SECRET_KEY is still the shipped default; set a private value "
            "in .env before exposing this to a network")
    if not settings.LIBRARY_PATH.exists():
        server.log.warning(
            "library path %s does not exist — is the share mounted? The UI "
            "will start, but a scan will refuse to run", settings.LIBRARY_PATH)
