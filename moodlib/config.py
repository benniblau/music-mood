"""Configuration — the only module in the project that reads the environment.

Everything tunable lives in `.env` (see `.env.example`); nothing operational is a
literal anywhere else. `grep -rn "os.environ" moodlib/` should match this file
and nothing else -- that is the invariant worth protecting, because it means
there is exactly one place to look when behaviour needs explaining.

The ontology is the deliberate exception. Axes, GEMS dimensions, the
adjective->GEMS mapping and the Discogs taxonomy live in `ontology.py` as code,
not here: they are a versioned data structure with internal relationships, and
changing one has to bump ONTOLOGY_VERSION so stale rows get re-tagged. A .env
edit cannot do that.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path | None = None) -> None:
    """Read KEY=value lines from .env without clobbering real environment vars.

    setdefault, not assignment: an exported variable always wins over the file,
    so a one-off `LLM_MODEL=x python3 run.py ...` works as expected.
    """
    path = path or PROJECT_ROOT / ".env"
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


load_env()


def _str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _csv(key: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(key, "").strip() or default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


# --- library ---------------------------------------------------------------
LIBRARY_PATH = Path(_str("LIBRARY_PATH", "~/Music")).expanduser()
AUDIO_EXTENSIONS = _csv("AUDIO_EXTENSIONS", ".m4a,.mp3,.mp4,.flac,.aiff,.wav")
SCAN_WORKERS = _int("SCAN_WORKERS", 24)

#: Abort a scan that would mark more than this share of known tracks missing.
#: The overwhelmingly likely cause is an unmounted NAS, and a scan that
#: "successfully" marks 23,000 tracks gone is indistinguishable from a wipe.
SCAN_MISSING_ABORT_PCT = _float("SCAN_MISSING_ABORT_PCT", 10.0)

#: Top-level folders that hold many artists rather than one. Inherited from the
#: library's own CLAUDE.md -- these are compilation buckets, not artists.
COMPILATION_DIRS = frozenset(_csv(
    "COMPILATION_DIRS",
    "Compilations,Various Artists,Various Artist,Verschiedene Interpreten,"
    "VA,Soundtrack,Unknown Artist,Unknown Error",
))

# --- llm endpoint ----------------------------------------------------------
LLM_URL = _str("LLM_URL", "http://localhost")
LLM_PORT = _str("LLM_PORT", "8888")
LLM_API_PATH = _str("LLM_API_PATH", "/v1")
LLM_MODEL = _str("LLM_MODEL", "")          # empty -> discovered from /v1/models
LLM_API_KEY = _str("LLM_API_KEY", "")      # vLLM usually needs none

# --- llm behaviour ---------------------------------------------------------
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.3)
#: Ceiling for one-off calls (query translation). Tagging does NOT use this --
#: it sizes its own budget from the batch, see tag_max_tokens() below.
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 2000)
#: A 20-track batch measured ~80s against this endpoint. 240s leaves ample
#: headroom while still failing a genuinely hung request fast enough that a
#: worker is not tied up for the length of the retry chain.
LLM_TIMEOUT = _int("LLM_TIMEOUT", 240)
LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 3)

# --- tagging ---------------------------------------------------------------
# Measured on this endpoint, tracks/s for the whole run (not per request):
#
#   batch 20, max_tokens 8000, 4 concurrent   0.19   <- the obvious config
#   batch 20, max_tokens 8000, 1 concurrent   0.25
#   batch 10, max_tokens 2600, 8 concurrent   0.77
#   batch 10, max_tokens 2600, 16 concurrent  1.08
#   batch 10, max_tokens 2600, 28 concurrent  1.17   <- diminishing
#
# Bigger batches are worse, which is the opposite of the intuition. vLLM
# schedules against the *reserved* output budget, so an 8000-token max_tokens
# lets far fewer sequences run at once than the ~19 this server can hold; the
# batch then decodes almost serially. Keeping the reservation tight is what buys
# the concurrency, so the two settings are not independent -- hence
# tag_max_tokens() below rather than a free-floating LLM_MAX_TOKENS.
TAG_BATCH_SIZE = _int("TAG_BATCH_SIZE", 10)
TAG_CONCURRENCY = _int("TAG_CONCURRENCY", 20)
#: Output tokens one track's mood object costs. Measured ~200; 220 leaves slack.
TAG_TOKENS_PER_TRACK = _int("TAG_TOKENS_PER_TRACK", 220)

# --- scoring ---------------------------------------------------------------
SCORE_AXIS_WEIGHT = _float("SCORE_AXIS_WEIGHT", 1.0)    # alpha
SCORE_GEMS_WEIGHT = _float("SCORE_GEMS_WEIGHT", 1.0)    # beta
SCORE_VOCAB_WEIGHT = _float("SCORE_VOCAB_WEIGHT", 0.15)
#: Multiplier applied to a confidence-0 track's score. Confidence 2 scores 1.0,
#: 1 interpolates. Guessed tracks still compete, just from behind.
SCORE_CONFIDENCE_FLOOR = _float("SCORE_CONFIDENCE_FLOOR", 0.75)

# --- playlist --------------------------------------------------------------
PLAYLIST_SIZE = _int("PLAYLIST_SIZE", 40)
MAX_PER_ARTIST = _int("MAX_PER_ARTIST", 2)
MIN_CONFIDENCE = _int("MIN_CONFIDENCE", 0)
#: Size of the scoring band sampled from, as a multiple of the playlist size.
#: >1 means repeated runs of the same mood return different (still good) picks.
PLAYLIST_POOL_FACTOR = _float("PLAYLIST_POOL_FACTOR", 3.0)
M3U_PATH_PREFIX = _str("M3U_PATH_PREFIX", "")

# --- paths -----------------------------------------------------------------
DATA_DIR = Path(_str("DATA_DIR", str(PROJECT_ROOT / "data")))
DB_PATH = Path(_str("DB_PATH", str(DATA_DIR / "mood.sqlite3")))


def llm_endpoint() -> str:
    """Compose LLM_URL + LLM_PORT + LLM_API_PATH into a base URL.

    Kept as a function rather than a constant so a test can monkeypatch the
    parts. LLM_PORT is load-bearing: a self-hosted vLLM rarely listens on 80, and
    omitting the port is a silent connection timeout rather than a clear error. A
    URL that already carries an explicit port wins, to avoid `host:80:8888`.
    """
    url = LLM_URL.rstrip("/")
    has_port = ":" in url.split("//", 1)[-1]
    if LLM_PORT and not has_port:
        url = f"{url}:{LLM_PORT}"
    return url + "/" + LLM_API_PATH.strip("/")


def tag_max_tokens(batch_size: int) -> int:
    """Output budget for a tagging batch, derived from its size.

    Derived rather than configured because the two are coupled: a generous fixed
    budget silently collapses throughput by starving vLLM's scheduler (see the
    table above). Sizing it from the batch means changing TAG_BATCH_SIZE stays
    safe, which a standalone LLM_MAX_TOKENS did not.
    """
    return batch_size * TAG_TOKENS_PER_TRACK + 400


def require_library(root: Path | None = None) -> Path:
    """Fail loudly if the library is not reachable.

    The library lives on a network mount, so the common failure is the share
    simply not being mounted. Without this check a scan "succeeds" over zero
    files, which reads as "nothing to do" rather than "the disk is gone".
    """
    root = root or LIBRARY_PATH
    if not root.exists():
        raise SystemExit(
            f"library not found: {root}\n"
            "Is the NAS mounted? Set LIBRARY_PATH in .env or pass --library.")
    if not root.is_dir():
        raise SystemExit(f"library path is not a directory: {root}")
    return root



def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
