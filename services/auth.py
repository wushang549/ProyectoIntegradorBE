"""Authentication helpers backed by Supabase Auth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import Header, HTTPException, status

from services.config import first_nonempty, get_backend_setting


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
    """Load Supabase URL and a backend API key."""

    supabase_url = get_backend_setting("SUPABASE_URL")
    api_key = first_nonempty(
        get_backend_setting("SUPABASE_SERVICE_ROLE_KEY"),
        get_backend_setting("SUPABASE_SECRET_KEY"),
    )

    if not supabase_url or not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase auth is not configured on backend.",
        )

    return supabase_url, api_key
