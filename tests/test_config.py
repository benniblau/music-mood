"""Configuration hygiene.

Two failure modes these prevent, both of which look like nothing is wrong: a
setting that quietly stops being configurable because someone inlined a literal,
and a key documented in .env.example that no code reads, so changing it does
nothing and the user has no way to tell.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from moodlib import config

ROOT = Path(__file__).resolve().parent.parent
ENV_KEY = re.compile(r"^([A-Z][A-Z0-9_]*)=", re.M)
# Keep this alternation in step with the readers in config.py — a missing
# one makes a genuinely-read key look undocumented, which is how WEB_DEBUG
# first failed this test.
CONFIG_READ = re.compile(r'_(?:str|int|float|bool|csv)\("([A-Z0-9_]+)"')


def _config_source() -> str:
    return (ROOT / "moodlib" / "config.py").read_text()


def test_only_config_reads_the_environment():
    offenders = [
        path.relative_to(ROOT)
        for path in (ROOT / "moodlib").glob("*.py")
        if path.name != "config.py" and "os.environ" in path.read_text()
    ]
    assert offenders == [], "settings must be read through config.py, not directly"


@pytest.mark.parametrize("filename", [".env.example", ".env"])
def test_documented_keys_are_actually_read(filename):
    path = ROOT / filename
    if not path.exists():
        pytest.skip(f"{filename} not present")
    documented = set(ENV_KEY.findall(path.read_text()))
    read = set(CONFIG_READ.findall(_config_source()))
    unread = sorted(documented - read)
    assert unread == [], f"{filename} documents keys nothing reads: {unread}"


def test_every_setting_is_documented_in_the_example():
    documented = set(ENV_KEY.findall((ROOT / ".env.example").read_text()))
    read = set(CONFIG_READ.findall(_config_source()))
    undocumented = sorted(read - documented)
    assert undocumented == [], f"settings missing from .env.example: {undocumented}"


def test_llm_endpoint_composes_url_and_port(monkeypatch):
    # A self-hosted vLLM rarely listens on 80, so dropping LLM_PORT is a silent
    # connection timeout rather than an obvious error.
    monkeypatch.setattr(config, "LLM_URL", "http://llm.example")
    monkeypatch.setattr(config, "LLM_PORT", "8888")
    monkeypatch.setattr(config, "LLM_API_PATH", "/v1")
    assert config.llm_endpoint() == "http://llm.example:8888/v1"


def test_llm_endpoint_does_not_double_up_an_explicit_port(monkeypatch):
    monkeypatch.setattr(config, "LLM_URL", "http://host:9000")
    monkeypatch.setattr(config, "LLM_PORT", "8888")
    monkeypatch.setattr(config, "LLM_API_PATH", "/v1")
    assert config.llm_endpoint() == "http://host:9000/v1"


def test_tag_budget_scales_with_batch_size():
    # The coupling that cost a stalled run: a fixed, over-generous max_tokens
    # starves vLLM's scheduler and collapses throughput. Deriving it from the
    # batch is what makes changing TAG_BATCH_SIZE safe.
    small = config.tag_max_tokens(10)
    large = config.tag_max_tokens(20)
    assert large > small
    assert large - small == 10 * config.TAG_TOKENS_PER_TRACK


def test_every_reader_helper_is_covered_by_the_key_scan():
    """The scan above must know about every `_x("KEY", ...)` helper.

    Otherwise a key read through an unlisted helper looks undocumented, and the
    two tests that police the .env surface start reporting phantom problems.
    """
    helpers = set(re.findall(r"^def _(\w+)\(key: str", _config_source(), re.M))
    known = set(CONFIG_READ.pattern.split("(?:")[1].split(")")[0].split("|"))
    assert helpers <= known, f"CONFIG_READ does not know about: {helpers - known}"


def test_web_settings_are_configurable():
    # The UI must not hardcode what the CLI takes from .env.
    for name in ("WEB_HOST", "WEB_PORT", "WEB_DEBUG", "WEB_SECRET_KEY",
                 "WEB_COVER_CACHE_SECONDS", "WEB_RECENT_QUERIES",
                 "WEB_PLAYLIST_SIZES", "WEB_POLL_SECONDS"):
        assert hasattr(config, name), name


def test_bool_setting_rejects_ambiguous_values(monkeypatch):
    # "WEB_DEBUG=no" silently enabling debug on a LAN-bound app would be a
    # nasty surprise, so anything unrecognised falls back to the default.
    from moodlib.config import _bool
    for raw, expected in [("true", True), ("YES", True), ("on", True), ("1", True),
                          ("false", False), ("no", False), ("off", False), ("0", False),
                          ("", False), ("banana", False)]:
        monkeypatch.setenv("MM_TEST_BOOL", raw)
        assert _bool("MM_TEST_BOOL", False) is expected, raw
