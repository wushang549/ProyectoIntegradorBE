"""Authentication helpers backed by Supabase Auth."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import Header, HTTPException, status


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    """Authenticated Supabase user extracted from an access token."""

    user_id: str
    email: str | None = None


def require_authenticated_user(authorization: str | None = Header(default=None)) -> AuthenticatedUser:
    """Resolve the current Supabase user from a bearer token."""

    token = _extract_bearer_token(authorization)
    supabase_url, api_key = _load_supabase_auth_config()
    request = Request(
        f"{supabase_url.rstrip('/')}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": api_key,
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN}:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired access token.") from exc
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Supabase auth lookup failed.") from exc
    except URLError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Supabase auth lookup failed.") from exc

    user_id = str(payload.get("id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired access token.")

    email = str(payload.get("email") or "").strip() or None
    return AuthenticatedUser(user_id=user_id, email=email)


def _extract_bearer_token(authorization: str | None) -> str:
    """Extract the token part from an Authorization header."""

    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header.")

    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header.")
    return parts[1].strip()


@lru_cache(maxsize=1)
def _load_supabase_auth_config() -> tuple[str, str]:
    """Load Supabase URL and an API key from env or frontend local config."""

    backend_env = _read_env_file(_backend_env_path())
    frontend_env = _read_env_file(_frontend_env_path())

    supabase_url = _first_nonempty(
        os.getenv("SUPABASE_URL"),
        backend_env.get("SUPABASE_URL"),
        os.getenv("VITE_SUPABASE_URL"),
        frontend_env.get("VITE_SUPABASE_URL"),
    )
    api_key = _first_nonempty(
        os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        backend_env.get("SUPABASE_SERVICE_ROLE_KEY"),
        os.getenv("SUPABASE_SECRET_KEY"),
        backend_env.get("SUPABASE_SECRET_KEY"),
        os.getenv("SUPABASE_PUBLISHABLE_KEY"),
        backend_env.get("SUPABASE_PUBLISHABLE_KEY"),
        os.getenv("SUPABASE_ANON_KEY"),
        backend_env.get("SUPABASE_ANON_KEY"),
        os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY"),
        os.getenv("VITE_SUPABASE_ANON_KEY"),
        frontend_env.get("VITE_SUPABASE_PUBLISHABLE_KEY"),
        frontend_env.get("VITE_SUPABASE_ANON_KEY"),
    )

    if not supabase_url or not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase auth is not configured on backend.",
        )

    return supabase_url, api_key


def _backend_env_path() -> Path:
    """Locate backend .env file for local development and deployment."""

    return Path(__file__).resolve().parents[1] / ".env"


def _frontend_env_path() -> Path:
    """Locate the frontend .env.local file for local development fallback."""

    return Path(__file__).resolve().parents[2] / "ProyectoIntegradorUI" / "my-react-app" / ".env.local"


def _read_env_file(path: Path) -> dict[str, str]:
    """Read a simple KEY=VALUE env file without external dependencies."""

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
