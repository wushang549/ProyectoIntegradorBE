"""Best-effort Supabase persistence for analysis runs and artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from services.storage import (
    clusters_file,
    embeddings_file,
    ensure_analysis_dir,
    hierarchy_file,
    insights_file,
    items_file,
    overview_file,
    umap_file,
    upsert_index_entry,
    write_json_file,
)


@dataclass(frozen=True, slots=True)
class SupabasePersistenceConfig:
    """Configuration required to persist analysis artifacts into Supabase."""

    url: str
    service_role_key: str
    storage_bucket: str


def upsert_analysis_run(record: dict[str, Any]) -> None:
    """Upsert one analysis run row into Supabase."""

    config = _load_config()
    if config is None:
        return

    payload = [record]
    _request(
        method="POST",
        url=f"{config.url.rstrip('/')}/rest/v1/analysis_runs?on_conflict=id",
        headers=_postgrest_headers(config.service_role_key, "resolution=merge-duplicates,return=minimal"),
        body=json.dumps(payload).encode("utf-8"),
    )


def upsert_analysis_results(record: dict[str, Any]) -> None:
    """Upsert one analysis_results row into Supabase."""

    config = _load_config()
    if config is None:
        return

    payload = [record]
    _request(
        method="POST",
        url=f"{config.url.rstrip('/')}/rest/v1/analysis_results?on_conflict=analysis_id",
        headers=_postgrest_headers(config.service_role_key, "resolution=merge-duplicates,return=minimal"),
        body=json.dumps(payload).encode("utf-8"),
    )


def patch_analysis_run(*, owner_id: str, analysis_id: str, fields: dict[str, Any]) -> None:
    """Patch selected columns on one analysis_runs row."""

    config = _load_config()
    if config is None:
        return

    payload = dict(fields)
    payload["updated_at"] = payload.get("updated_at") or _utc_now_iso()
    query = urlencode(
        [
            ("owner_id", f"eq.{owner_id}"),
            ("id", f"eq.{analysis_id}"),
        ]
    )
    _request(
        method="PATCH",
        url=f"{config.url.rstrip('/')}/rest/v1/analysis_runs?{query}",
        headers=_postgrest_headers(config.service_role_key, "return=minimal"),
        body=json.dumps(payload).encode("utf-8"),
    )


def patch_analysis_results(*, owner_id: str, analysis_id: str, fields: dict[str, Any]) -> None:
    """Patch selected columns on one analysis_results row."""

    config = _load_config()
    if config is None:
        return

    payload = dict(fields)
    payload["updated_at"] = payload.get("updated_at") or _utc_now_iso()
    query = urlencode(
        [
            ("owner_id", f"eq.{owner_id}"),
            ("analysis_id", f"eq.{analysis_id}"),
        ]
    )
    _request(
        method="PATCH",
        url=f"{config.url.rstrip('/')}/rest/v1/analysis_results?{query}",
        headers=_postgrest_headers(config.service_role_key, "return=minimal"),
        body=json.dumps(payload).encode("utf-8"),
    )


def upload_artifact(
    *,
    owner_id: str,
    analysis_id: str,
    filename: str,
    content_type: str,
    file_body: bytes,
) -> str | None:
    """Upload one artifact file into Supabase Storage and return its path."""

    config = _load_config()
    if config is None:
        return None

    object_path = f"{owner_id}/{analysis_id}/{filename}"
    encoded_path = quote(f"{config.storage_bucket}/{object_path}", safe="/")
    _request(
        method="POST",
        url=f"{config.url.rstrip('/')}/storage/v1/object/{encoded_path}",
        headers={
            "Authorization": f"Bearer {config.service_role_key}",
            "apikey": config.service_role_key,
            "content-type": content_type,
            "cache-control": "max-age=3600",
            "x-upsert": "true",
        },
        body=file_body,
    )
    return object_path


def list_analysis_runs(*, owner_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Load recent analysis run rows for one owner from Supabase."""

    return _select_rows(
        table="analysis_runs",
        filters={
            "owner_id": f"eq.{owner_id}",
        },
        order="created_at.desc",
        limit=limit,
    )


def get_analysis_run(*, owner_id: str, analysis_id: str) -> dict[str, Any] | None:
    """Load one analysis run row for one owner from Supabase."""

    rows = _select_rows(
        table="analysis_runs",
        filters={
            "owner_id": f"eq.{owner_id}",
            "id": f"eq.{analysis_id}",
        },
        limit=1,
    )
    return rows[0] if rows else None


def get_analysis_results(*, owner_id: str, analysis_id: str) -> dict[str, Any] | None:
    """Load one analysis_results row for one owner from Supabase."""

    rows = _select_rows(
        table="analysis_results",
        filters={
            "owner_id": f"eq.{owner_id}",
            "analysis_id": f"eq.{analysis_id}",
        },
        limit=1,
    )
    return rows[0] if rows else None


def download_artifact(*, storage_path: str) -> bytes:
    """Download one private Storage object from Supabase."""

    config = _load_config()
    if config is None:
        raise RuntimeError("Supabase persistence is not configured.")

    encoded_path = quote(f"{config.storage_bucket}/{storage_path}", safe="/")
    return _request(
        method="GET",
        url=f"{config.url.rstrip('/')}/storage/v1/object/{encoded_path}",
        headers={
            "Authorization": f"Bearer {config.service_role_key}",
            "apikey": config.service_role_key,
        },
        body=None,
    )


def delete_analysis(*, owner_id: str, analysis_id: str) -> None:
    """Delete one persisted analysis and its storage artifacts from Supabase."""

    config = _load_config()
    if config is None:
        return

    run_row = get_analysis_run(owner_id=owner_id, analysis_id=analysis_id)
    storage_path = ""
    if isinstance(run_row, dict):
        storage_path = str(run_row.get("embeddings_storage_path") or "").strip()
    if not storage_path:
        storage_path = f"{owner_id}/{analysis_id}/embeddings.npy"

    if storage_path:
        _delete_artifact_if_exists(config=config, storage_path=storage_path)

    query = urlencode(
        [
            ("owner_id", f"eq.{owner_id}"),
            ("id", f"eq.{analysis_id}"),
        ]
    )
    _request(
        method="DELETE",
        url=f"{config.url.rstrip('/')}/rest/v1/analysis_runs?{query}",
        headers=_postgrest_headers(config.service_role_key, "return=minimal"),
        body=None,
    )

    remaining = get_analysis_run(owner_id=owner_id, analysis_id=analysis_id)
    if remaining is not None:
        raise RuntimeError("Supabase delete did not remove the analysis row.")


def hydrate_analysis_locally(*, owner_id: str, analysis_id: str) -> bool:
    """Recreate local analysis artifacts from Supabase for one owner."""

    run_row = get_analysis_run(owner_id=owner_id, analysis_id=analysis_id)
    results_row = get_analysis_results(owner_id=owner_id, analysis_id=analysis_id)
    if run_row is None and results_row is None:
        return False

    materialized = False

    if isinstance(run_row, dict):
        upsert_index_entry(_analysis_run_to_index_entry(run_row))
        materialized = True

    if isinstance(results_row, dict):
        ensure_analysis_dir(analysis_id)
        _write_optional_json(items_file(analysis_id), results_row.get("items_json"))
        _write_optional_json(overview_file(analysis_id), results_row.get("overview_json"))
        _write_optional_json(insights_file(analysis_id), results_row.get("insights_json"))
        _write_optional_json(clusters_file(analysis_id), results_row.get("clusters_json"))
        _write_optional_json(umap_file(analysis_id), results_row.get("umap_json"))
        _write_optional_json(hierarchy_file(analysis_id), results_row.get("hierarchy_json"))
        materialized = True

    storage_path = ""
    if isinstance(run_row, dict):
        storage_path = str(run_row.get("embeddings_storage_path") or "").strip()
    if not storage_path:
        storage_path = f"{owner_id}/{analysis_id}/embeddings.npy"

    if storage_path:
        try:
            ensure_analysis_dir(analysis_id)
            embeddings_file(analysis_id).write_bytes(download_artifact(storage_path=storage_path))
            materialized = True
        except Exception:
            pass

    return materialized


def is_configured() -> bool:
    """Return whether Supabase persistence credentials are available."""

    return _load_config() is not None


def _load_config() -> SupabasePersistenceConfig | None:
    """Load persistence config from backend env or env file."""

    backend_env = _read_env_file(_backend_env_path())
    frontend_env = _read_env_file(_frontend_env_path())

    url = _first_nonempty(
        os.getenv("SUPABASE_URL"),
        backend_env.get("SUPABASE_URL"),
        os.getenv("VITE_SUPABASE_URL"),
        frontend_env.get("VITE_SUPABASE_URL"),
    )
    service_role_key = _first_nonempty(
        os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        backend_env.get("SUPABASE_SERVICE_ROLE_KEY"),
        os.getenv("SUPABASE_SECRET_KEY"),
        backend_env.get("SUPABASE_SECRET_KEY"),
    )
    storage_bucket = _first_nonempty(
        os.getenv("SUPABASE_STORAGE_BUCKET"),
        backend_env.get("SUPABASE_STORAGE_BUCKET"),
        "analysis-artifacts",
    )

    if not url or not service_role_key:
        return None

    return SupabasePersistenceConfig(
        url=url,
        service_role_key=service_role_key,
        storage_bucket=storage_bucket,
    )


def _select_rows(
    *,
    table: str,
    filters: dict[str, str],
    order: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Query one PostgREST table and return rows."""

    config = _load_config()
    if config is None:
        return []

    query_items: list[tuple[str, str]] = [("select", "*")]
    for key, value in filters.items():
        query_items.append((key, value))
    if order:
        query_items.append(("order", order))
    if limit is not None:
        query_items.append(("limit", str(max(1, int(limit)))))

    raw = _request(
        method="GET",
        url=f"{config.url.rstrip('/')}/rest/v1/{table}?{urlencode(query_items)}",
        headers={
            "Authorization": f"Bearer {config.service_role_key}",
            "apikey": config.service_role_key,
        },
        body=None,
    )
    loaded = json.loads(raw.decode("utf-8") or "[]")
    if not isinstance(loaded, list):
        return []
    return [row for row in loaded if isinstance(row, dict)]


def _postgrest_headers(api_key: str, prefer: str) -> dict[str, str]:
    """Build standard PostgREST headers for Supabase REST calls."""

    return {
        "Authorization": f"Bearer {api_key}",
        "apikey": api_key,
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


def _request(*, method: str, url: str, headers: dict[str, str], body: bytes | None) -> bytes:
    """Perform one HTTP request and raise a compact runtime error on failure."""

    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            return response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Supabase request failed ({exc.code}): {detail or exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"Supabase request failed: {exc.reason}") from exc


def _delete_artifact_if_exists(*, config: SupabasePersistenceConfig, storage_path: str) -> None:
    """Delete one Storage object and ignore missing-object responses."""

    encoded_path = quote(f"{config.storage_bucket}/{storage_path}", safe="/")
    try:
        _request(
            method="DELETE",
            url=f"{config.url.rstrip('/')}/storage/v1/object/{encoded_path}",
            headers={
                "Authorization": f"Bearer {config.service_role_key}",
                "apikey": config.service_role_key,
            },
            body=None,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "(404)" in message:
            return
        raise


def _write_optional_json(path: Path, payload: Any) -> None:
    """Persist one JSON artifact when a payload exists."""

    if payload is None:
        return
    write_json_file(path, payload)


def _analysis_run_to_index_entry(run_row: dict[str, Any]) -> dict[str, Any]:
    """Translate a Supabase analysis_runs row into the local index shape."""

    return {
        "analysis_id": str(run_row.get("id") or "").strip(),
        "owner_id": str(run_row.get("owner_id") or "").strip(),
        "created_at": str(run_row.get("created_at") or "").strip(),
        "updated_at": str(run_row.get("updated_at") or "").strip(),
        "status": str(run_row.get("status") or "queued").strip() or "queued",
        "stage": str(run_row.get("raw_stage") or "queued").strip() or "queued",
        "pct": int(run_row.get("progress_pct") or 0),
        "total_records": int(run_row.get("total_records") or 0),
        "total_items": int(run_row.get("total_items") or 0),
        "item_count": int(run_row.get("total_items") or 0),
        "error": str(run_row.get("error_message") or "").strip() or None,
        "input_type": str(run_row.get("input_type") or "").strip() or None,
        "source_name": str(run_row.get("source_name") or "").strip() or None,
        "llm_model": str(run_row.get("llm_model") or "").strip() or None,
    }


def _utc_now_iso() -> str:
    """Current UTC timestamp in ISO format."""

    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _backend_env_path() -> Path:
    """Path to backend local env file."""

    return Path(__file__).resolve().parents[1] / ".env"


def _frontend_env_path() -> Path:
    """Path to frontend local env file for URL fallback."""

    return Path(__file__).resolve().parents[2] / "ProyectoIntegradorUI" / "my-react-app" / ".env.local"


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
