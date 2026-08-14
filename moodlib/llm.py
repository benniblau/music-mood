"""OpenAI-compatible client for the local vLLM endpoint.

Everything the model returns is constrained by a strict JSON schema, so callers
get a validated object rather than prose to parse. Three details were established
by testing against the actual endpoint and are load-bearing:

* `chat_template_kwargs: {enable_thinking: false}` -- without it Qwen3.6 spends a
  large share of the completion budget on reasoning we throw away.
* `response_format: {type: "json_schema", strict: true}` -- vLLM's guided
  decoding then makes an out-of-vocabulary value structurally impossible, which
  is what lets the ontology's closed enums actually hold.
* the model name is discovered from `/v1/models` when unset, so a server-side
  model swap does not require an .env edit.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

import requests

from moodlib import config, progress


class LLMError(RuntimeError):
    pass


_model_lock = threading.Lock()
_resolved_model: str | None = None
_announced = False


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
    return headers


def resolve_model(session: requests.Session | None = None) -> str:
    """The model id to send. Discovered once from /v1/models unless pinned."""
    global _resolved_model, _announced
    with _model_lock:
        if _resolved_model:
            return _resolved_model
        if config.LLM_MODEL:
            _resolved_model = config.LLM_MODEL
        else:
            endpoint = config.llm_endpoint()
            try:
                get = (session or requests).get
                response = get(f"{endpoint}/models", headers=_headers(), timeout=30)
                response.raise_for_status()
                models = response.json().get("data") or []
                if not models:
                    raise LLMError("endpoint returned no models")
                _resolved_model = models[0]["id"]
            except requests.RequestException as exc:
                raise LLMError(
                    f"cannot reach the LLM at {endpoint} ({exc}).\n"
                    "Check LLM_URL and LLM_PORT in .env — port 80 is closed on "
                    "the usual host, so LLM_PORT is required.") from exc
        if not _announced:
            # Announce the resolved endpoint once, so a wrong .env value shows up
            # as a visible mismatch rather than a confusing timeout later.
            progress.note("llm", f"{config.llm_endpoint()}  model={_resolved_model}")
            _announced = True
        return _resolved_model


def complete_json(prompt: str, schema: dict, *, schema_name: str = "out",
                  max_tokens: int | None = None, temperature: float | None = None,
                  session: requests.Session | None = None) -> Any:
    """One schema-constrained completion, with retry on transport and parse errors.

    Retries cover the failure modes actually seen against vLLM: a transport blip,
    a 5xx while the server is loading, and -- rarely -- a completion truncated at
    max_tokens, which yields valid-looking JSON that does not parse. A 4xx is not
    retried: a malformed schema will be malformed the second time too.
    """
    endpoint = config.llm_endpoint()
    model = resolve_model(session)
    http = session or requests
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
    }

    last: Exception | None = None
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            response = http.post(f"{endpoint}/chat/completions", headers=_headers(),
                                 json=payload, timeout=config.LLM_TIMEOUT)
            if 400 <= response.status_code < 500:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            content = choice["message"]["content"]
            if choice.get("finish_reason") == "length":
                raise LLMError("completion hit max_tokens; response truncated")
            return json.loads(content)
        except LLMError as exc:
            if "HTTP 4" in str(exc):
                raise
            last = exc
        except (requests.RequestException, json.JSONDecodeError, KeyError,
                IndexError, TypeError) as exc:
            last = exc
        if attempt < config.LLM_MAX_RETRIES - 1:
            time.sleep(2 ** attempt)

    raise LLMError(f"failed after {config.LLM_MAX_RETRIES} attempts: {last}")


def new_session() -> requests.Session:
    """A session with a connection pool sized for the tagging concurrency."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=config.TAG_CONCURRENCY,
        pool_maxsize=config.TAG_CONCURRENCY,
        max_retries=0,  # retry policy lives in complete_json, which knows the schema
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
