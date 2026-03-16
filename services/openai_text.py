"""Minimal OpenAI Responses API client for labeling and summaries."""

from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_OPENAI_TEXT_MODEL = "gpt-5-nano"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL_SENTINELS = {"", "auto", "default"}
_RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRY_ATTEMPTS = 3
_LOGGER = logging.getLogger(__name__)


class OpenAITextError(RuntimeError):
    """Raised when an OpenAI text request fails."""


def default_openai_text_model() -> str:
    """Return the configured default OpenAI text model."""

    return _load_openai_config()["default_model"]


def resolve_openai_text_model(model_name: str | None) -> str:
    """Resolve one requested model name against backend defaults."""

    candidate = str(model_name or "").strip()
    if candidate.lower() in _DEFAULT_MODEL_SENTINELS:
        return default_openai_text_model()
    return candidate or default_openai_text_model()


def request_openai_text(
    prompt: str,
    *,
    model_name: str | None = None,
    instructions: str | None = None,
    max_output_tokens: int = 160,
    timeout_sec: int = 45,
) -> str:
    """Generate plain text with OpenAI's Responses API."""

    config = _load_openai_config()
    api_key = config["api_key"]
    if not api_key:
        raise OpenAITextError("OpenAI API key is not configured on backend.")

    payload: dict[str, Any] = {
        "model": resolve_openai_text_model(model_name),
        "input": str(prompt or "").strip(),
        "max_output_tokens": max(16, int(max_output_tokens)),
        "reasoning": {"effort": "minimal"},
        "store": False,
    }
    if instructions:
        payload["instructions"] = instructions

    request = Request(
        f"{config['base_url'].rstrip('/')}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error = "OpenAI request failed."
    for attempt in range(1, _MAX_RETRY_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=timeout_sec) as response:
                raw = response.read().decode("utf-8")
                response_headers = dict(response.headers.items())
        except HTTPError as exc:
            last_error = _extract_error_message(exc)
            _emit_debug_event(
                event="http_error",
                model_name=str(payload["model"]),
                detail=f"attempt={attempt} status={getattr(exc, 'code', 'unknown')} error={last_error}",
            )
            if attempt < _MAX_RETRY_ATTEMPTS and _is_retryable_http_error(exc):
                _sleep_before_retry(attempt)
                continue
            raise OpenAITextError(last_error) from exc
        except URLError as exc:
            last_error = "OpenAI API is unavailable."
            _emit_debug_event(
                event="network_error",
                model_name=str(payload["model"]),
                detail=f"attempt={attempt} error={last_error}",
            )
            if attempt < _MAX_RETRY_ATTEMPTS:
                _sleep_before_retry(attempt)
                continue
            raise OpenAITextError(last_error) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OpenAITextError("OpenAI response was not valid JSON.") from exc

        text = _extract_output_text(parsed)
        if text:
            _log_usage_if_enabled(
                model_name=str(parsed.get("model") or payload["model"]),
                payload=parsed,
                response_headers=response_headers,
            )
            return text

        if attempt < _MAX_RETRY_ATTEMPTS and _response_looks_retryable(parsed):
            last_error = "OpenAI response did not contain any text output."
            _emit_debug_event(
                event="empty_output_retry",
                model_name=str(parsed.get("model") or payload["model"]),
                detail=f"attempt={attempt} status={parsed.get('status') or 'unknown'}",
            )
            _sleep_before_retry(attempt)
            continue
        _emit_debug_event(
            event="empty_output_error",
            model_name=str(parsed.get("model") or payload["model"]),
            detail=f"attempt={attempt} status={parsed.get('status') or 'unknown'}",
        )
        raise OpenAITextError("OpenAI response did not contain any text output.")

    raise OpenAITextError(last_error)


@lru_cache(maxsize=1)
def _load_openai_config() -> dict[str, str]:
    """Load OpenAI configuration from process env or backend .env."""

    backend_env = _read_env_file(_backend_env_path())
    return {
        "api_key": _first_nonempty(
            os.getenv("OPENAI_API_KEY"),
            backend_env.get("OPENAI_API_KEY"),
        ),
        "base_url": _first_nonempty(
            os.getenv("OPENAI_BASE_URL"),
            backend_env.get("OPENAI_BASE_URL"),
            DEFAULT_OPENAI_BASE_URL,
        ),
        "default_model": _first_nonempty(
            os.getenv("OPENAI_TEXT_MODEL"),
            backend_env.get("OPENAI_TEXT_MODEL"),
            DEFAULT_OPENAI_TEXT_MODEL,
        ),
        "log_usage": _first_nonempty(
            os.getenv("OPENAI_LOG_USAGE"),
            backend_env.get("OPENAI_LOG_USAGE"),
            "0",
        ),
    }


def _extract_output_text(payload: dict[str, Any]) -> str:
    """Extract the primary text output from a Responses API payload."""

    top_level = payload.get("output_text")
    if isinstance(top_level, str) and top_level.strip():
        return top_level.strip()

    output = payload.get("output")
    if not isinstance(output, list):
        return ""

    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "").strip().lower()
            if block_type not in {"output_text", "text"}:
                continue
            value = block.get("text")
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())
                continue
            if isinstance(value, dict):
                nested = str(value.get("value") or "").strip()
                if nested:
                    chunks.append(nested)

    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _extract_error_message(exc: HTTPError) -> str:
    """Parse one readable API error from an HTTPError."""

    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except Exception:
        return f"OpenAI request failed with status {exc.code}."

    error = payload.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        if message:
            return message
    return f"OpenAI request failed with status {exc.code}."


def _log_usage_if_enabled(
    *,
    model_name: str,
    payload: dict[str, Any],
    response_headers: dict[str, Any],
) -> None:
    """Emit one temporary usage log for successful OpenAI calls."""

    if not _is_truthy_env(_load_openai_config().get("log_usage")):
        return

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    request_id = str(response_headers.get("x-request-id") or "").strip() or "unknown"
    project_id = str(response_headers.get("openai-project") or "").strip() or "unknown"
    _LOGGER.warning(
        "OpenAI usage model=%s input_tokens=%s output_tokens=%s total_tokens=%s project=%s request_id=%s",
        model_name,
        input_tokens,
        output_tokens,
        total_tokens,
        project_id,
        request_id,
    )
    print(
        f"[OPENAI_USAGE] model={model_name} input_tokens={input_tokens} "
        f"output_tokens={output_tokens} total_tokens={total_tokens} "
        f"project={project_id} request_id={request_id}"
    )


def _emit_debug_event(*, event: str, model_name: str, detail: str) -> None:
    """Emit temporary stdout diagnostics for OpenAI integration debugging."""

    if not _is_truthy_env(_load_openai_config().get("log_usage")):
        return
    print(f"[OPENAI_DEBUG] event={event} model={model_name} {detail}")


def _is_retryable_http_error(exc: HTTPError) -> bool:
    """Return whether one HTTP error should be retried automatically."""

    if int(getattr(exc, "code", 0) or 0) in _RETRYABLE_HTTP_STATUSES:
        return True
    message = _extract_error_message(exc).lower()
    return "server had an error" in message or "rate limit" in message


def _response_looks_retryable(payload: dict[str, Any]) -> bool:
    """Identify response payloads that may recover on one quick retry."""

    status = str(payload.get("status") or "").strip().lower()
    if status in {"incomplete", "queued", "failed"}:
        return True

    error = payload.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip().lower()
        if "server had an error" in message or "rate limit" in message:
            return True
    return False


def _sleep_before_retry(attempt: int) -> None:
    """Back off briefly before retrying transient API failures."""

    time.sleep(min(3.0, 0.5 * max(1, attempt)))


def _is_truthy_env(value: str | None) -> bool:
    """Parse simple env toggles such as 1/true/on/yes."""

    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _backend_env_path() -> Path:
    """Locate backend .env file for local development and deployment."""

    return Path(__file__).resolve().parents[1] / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    """Read a simple KEY=VALUE env file without extra dependencies."""

    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def _first_nonempty(*values: Any) -> str:
    """Return the first non-empty string candidate."""

    for value in values:
        candidate = str(value or "").strip()
        if candidate:
            return candidate
    return ""
