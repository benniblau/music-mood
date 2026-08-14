"""WSGI entry point for a real server.

    .venv/bin/gunicorn -c gunicorn.conf.py wsgi:app

Deliberately separate from `webapp/app.py`, which keeps the development runner
(`app.run()`, one process, the reloader) that this must never be confused with.
Importing this module never starts a server; the WSGI container owns that.
"""
from __future__ import annotations

import sys
from pathlib import Path

# gunicorn puts its `chdir` on sys.path, but this file has to work when it is
# imported some other way too -- a systemd unit with the wrong WorkingDirectory
# should fail on something clearer than a bare ImportError.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from webapp.app import app  # noqa: E402

__all__ = ["app"]
