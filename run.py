#!/usr/bin/env python3
"""Entry point. No install step, no packaging — just `python3 run.py <command>`.

If launched with an interpreter that lacks the dependencies, this re-execs
itself under ./.venv, so `./run.py` and a bare `python3 run.py` both work
regardless of which Python happens to be first on PATH.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _reexec_under_venv() -> None:
    venv = ROOT / ".venv"
    venv_python = venv / "bin" / "python3"
    # Compare prefixes, not interpreter paths: .venv/bin/python3 is a symlink to
    # the base interpreter, so resolve() makes the two look identical and we
    # would decide we had already re-execed when we had not.
    already = Path(sys.prefix).resolve() == venv.resolve()
    if already or not venv_python.exists():
        raise SystemExit(
            "missing dependencies. Set up the environment with:\n"
            "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt")
    os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])


try:
    import requests  # noqa: F401
except ImportError:
    _reexec_under_venv()

from moodlib.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
