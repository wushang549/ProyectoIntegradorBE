"""Regression tests for Ollama model discovery."""

from __future__ import annotations

import json
import subprocess
import urllib.error

from services.labeling import _resolve_ollama_cli_command, list_ollama_models


class _FakeHTTPResponse:
    """Minimal context manager for mocking urllib responses."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload.encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_list_ollama_models_prefers_http_api(monkeypatch) -> None:
    """The backend should use Ollama's HTTP API when it is reachable."""

    payload = json.dumps(
        {
            "models": [
                {
                    "name": "qwen3:4b",
                    "digest": "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7",
                    "size": 2497293931,
                }
            ]
        }
    )

    monkeypatch.setattr(
        "services.labeling.urllib.request.urlopen",
        lambda *_args, **_kwargs: _FakeHTTPResponse(payload),
    )
    monkeypatch.setattr(
        "services.labeling._list_ollama_models_via_cli",
        lambda: [{"name": "cli-fallback"}],
    )

    assert list_ollama_models() == [
        {
            "name": "qwen3:4b",
            "id": "359d7dd4bcda",
            "size": "2.5 GB",
        }
    ]


def test_list_ollama_models_falls_back_to_cli_when_api_is_unavailable(monkeypatch) -> None:
    """The backend should still find models when only the CLI path works."""

    def fake_urlopen(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    def fake_run(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        assert command == [r"C:\Ollama\ollama.exe", "list"]
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=(
                "NAME              ID              SIZE      MODIFIED     \n"
                "deepseek-r1:8b    6995872bfe4c    5.2 GB    46 hours ago\n"
            ),
            stderr="",
        )

    monkeypatch.setattr("services.labeling.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("services.labeling._resolve_ollama_cli_command", lambda: r"C:\Ollama\ollama.exe")
    monkeypatch.setattr("services.labeling.subprocess.run", fake_run)

    assert list_ollama_models() == [
        {
            "name": "deepseek-r1:8b",
            "id": "6995872bfe4c",
            "size": "5.2 GB",
            "modified": "46 hours ago",
        }
    ]


def test_resolve_ollama_cli_command_prefers_env_override(monkeypatch) -> None:
    """A configured Ollama executable should take precedence over PATH lookup."""

    monkeypatch.setenv("OLLAMA_BIN", r"C:\custom\ollama.exe")
    monkeypatch.setattr("services.labeling.shutil.which", lambda *_args: None)

    assert _resolve_ollama_cli_command() == r"C:\custom\ollama.exe"
