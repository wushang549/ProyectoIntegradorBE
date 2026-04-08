"""Shared backend configuration helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_LOCAL_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
)


def backend_env_path() -> Path:
    """Locate the backend .env file."""

    return Path(__file__).resolve().parents[1] / ".env"


@lru_cache(maxsize=1)
def load_backend_env() -> dict[str, str]:
    """Load backend env values from ProyectoIntegradorBE/.env when present."""

    return read_env_file(backend_env_path())


def read_env_file(path: Path) -> dict[str, str]:
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


def first_nonempty(*values: Any) -> str:
    """Return the first non-empty string candidate."""

    for value in values:
        candidate = str(value or "").strip()
        if candidate:
            return candidate
    return ""


def get_backend_setting(*names: str, default: str = "") -> str:
    """Resolve one backend setting from process env first, then local .env."""

    backend_env = load_backend_env()
    candidates: list[str] = []
    for name in names:
        candidates.append(os.getenv(name, ""))
        candidates.append(backend_env.get(name, ""))
    resolved = first_nonempty(*candidates)
    return resolved or default


def load_cors_allowed_origins() -> list[str]:
    """Resolve allowed CORS origins from env, defaulting to local development."""

    raw = get_backend_setting("CORS_ALLOWED_ORIGINS")
    if not raw:
        return list(DEFAULT_LOCAL_CORS_ORIGINS)

    parts = raw.replace("\n", ",").replace(";", ",").split(",")
    origins: list[str] = []
    seen: set[str] = set()
    for part in parts:
        candidate = part.strip().rstrip("/")
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        origins.append(candidate)

    return origins or list(DEFAULT_LOCAL_CORS_ORIGINS)
