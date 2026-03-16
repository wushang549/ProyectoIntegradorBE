"""Regression tests for OpenAI retry behavior."""

from __future__ import annotations

from io import BytesIO
from urllib.error import HTTPError

from services.openai_text import OpenAITextError, request_openai_text


class _FakeResponse:
    """Minimal urlopen response stub."""

    def __init__(self, payload: bytes, headers: dict[str, str] | None = None) -> None:
        self._payload = payload
        self.headers = headers or {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _fake_http_error(status_code: int, message: str) -> HTTPError:
    """Build one HTTPError with a JSON API payload."""

    payload = f'{{"error": {{"message": "{message}"}}}}'.encode("utf-8")
    return HTTPError(
        url="https://api.openai.com/v1/responses",
        code=status_code,
        msg="error",
        hdrs=None,
        fp=BytesIO(payload),
    )


def test_request_openai_text_retries_transient_server_errors(monkeypatch) -> None:
    """Transient OpenAI 500s should be retried before failing."""

    attempts = {"count": 0}

    monkeypatch.setattr(
        "services.openai_text._load_openai_config",
        lambda: {
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-5-nano",
        },
    )
    monkeypatch.setattr("services.openai_text.time.sleep", lambda _seconds: None)

    def fake_urlopen(_request, timeout=0):  # noqa: ARG001
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise _fake_http_error(
                500,
                "The server had an error processing your request. Please retry.",
            )
        return _FakeResponse(b'{"output_text":"Small Portions"}')

    monkeypatch.setattr("services.openai_text.urlopen", fake_urlopen)

    label = request_openai_text("label this cluster", max_output_tokens=64)

    assert label == "Small Portions"
    assert attempts["count"] == 3


def test_request_openai_text_does_not_retry_non_retryable_http_errors(monkeypatch) -> None:
    """Client-side 400s should surface immediately."""

    monkeypatch.setattr(
        "services.openai_text._load_openai_config",
        lambda: {
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-5-nano",
        },
    )
    monkeypatch.setattr("services.openai_text.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "services.openai_text.urlopen",
        lambda _request, timeout=0: (_ for _ in ()).throw(
            _fake_http_error(400, "Bad request payload.")
        ),
    )

    try:
        request_openai_text("label this cluster", max_output_tokens=64)
    except OpenAITextError as exc:
        assert "Bad request payload." in str(exc)
    else:
        raise AssertionError("Expected OpenAITextError for non-retryable 400 response.")
