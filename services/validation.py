from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

MAX_CSV_SIZE_BYTES = 50 * 1024 * 1024
TEXT_COLUMN_ALIAS_PRIORITY = ("text", "transcript", "message", "content", "review", "comment", "body", "feedback")
MAX_METADATA_KEYS = 20
MAX_METADATA_KEY_LEN = 60
MAX_METADATA_VALUE_LEN = 240


def sanitize_preview(text: str, limit: int = 140) -> str:
    clean = " ".join((text or "").split())
    return clean[:limit]


def validate_text_input(text: str | None) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="text is required when input_type=text")
    return cleaned


def _decode_csv_bytes(payload: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return payload.decode(enc)
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400, detail="CSV could not be decoded")


def detect_text_column(columns: list[str]) -> str | None:
    by_lower: dict[str, list[str]] = {}
    for col in columns:
        key = (col or "").strip().lower()
        if not key:
            continue
        by_lower.setdefault(key, []).append(col)

    for alias in TEXT_COLUMN_ALIAS_PRIORITY:
        candidates = by_lower.get(alias, [])
        if not candidates:
            continue
        preferred = (alias, alias.capitalize(), alias.upper())
        for pref in preferred:
            if pref in candidates:
                return pref
        return candidates[0]
    return None


def normalize_text_column_rows(rows: list[dict[str, str]], source_column: str) -> tuple[list[dict[str, str]], str]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        normalized.append({"text": (row.get(source_column) or "").strip()})
    return normalized, source_column


def _sanitize_metadata_key(key: str, limit: int = MAX_METADATA_KEY_LEN) -> str:
    clean = " ".join(str(key or "").split()).strip()
    if not clean:
        return ""
    return clean[:limit]


def _sanitize_metadata_value(value: Any, limit: int = MAX_METADATA_VALUE_LEN) -> str:
    clean = " ".join(str(value or "").split()).strip()
    if not clean:
        return ""
    return clean[:limit]


def extract_rows_from_csv_payload(
    payload: bytes,
    max_rows: int | None = None,
    max_metadata_keys: int = MAX_METADATA_KEYS,
    max_metadata_value_len: int = MAX_METADATA_VALUE_LEN,
) -> tuple[list[dict[str, Any]], str]:
    decoded = _decode_csv_bytes(payload)
    reader = csv.DictReader(io.StringIO(decoded))
    headers = [h for h in (reader.fieldnames or []) if h is not None]

    source_column = detect_text_column(headers)
    if source_column is None:
        found_headers = ", ".join(headers) if headers else "(none)"
        raise HTTPException(
            status_code=400,
            detail=(
                "CSV must include a text column "
                f"(accepted: {', '.join(TEXT_COLUMN_ALIAS_PRIORITY)}). "
                f"Found headers: {found_headers}"
            ),
        )

    row_limit: int | None = None
    if max_rows is not None:
        try:
            row_limit = max(1, int(max_rows))
        except (TypeError, ValueError):
            row_limit = None

    metadata_key_limit = max(0, int(max_metadata_keys))
    metadata_headers = [h for h in headers if h != source_column]
    metadata_headers = metadata_headers[:metadata_key_limit]

    rows: list[dict[str, Any]] = []
    for row in reader:
        text_value = (row.get(source_column) or "").strip()
        if not text_value:
            continue

        metadata: dict[str, str] = {}
        for header in metadata_headers:
            safe_key = _sanitize_metadata_key(header)
            if not safe_key:
                continue
            raw_value = row.get(header)
            safe_value = _sanitize_metadata_value(raw_value, limit=max_metadata_value_len)
            if not safe_value:
                continue
            metadata[safe_key] = safe_value
            if len(metadata) >= metadata_key_limit:
                break

        rows.append({"text": text_value, "metadata": metadata})
        if row_limit is not None and len(rows) >= row_limit:
            break

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="CSV 'text' column must include at least 1 non-empty row",
        )
    return rows, source_column


def extract_texts_from_csv_payload(payload: bytes, max_rows: int | None = None) -> tuple[list[str], str]:
    rows, source_column = extract_rows_from_csv_payload(payload=payload, max_rows=max_rows)
    texts = [str(row.get("text", "")).strip() for row in rows if str(row.get("text", "")).strip()]
    return texts, source_column


def validate_csv_upload(
    file: UploadFile | None,
    max_rows: int | None = None,
) -> tuple[list[str], bytes, list[dict[str, Any]]]:
    if file is None:
        raise HTTPException(status_code=400, detail="file is required when input_type=csv")
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="file must have .csv extension")

    payload = file.file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    if len(payload) > MAX_CSV_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="CSV file exceeds 50MB limit")

    rows, _source_column = extract_rows_from_csv_payload(payload, max_rows=max_rows)
    texts = [str(row.get("text", "")).strip() for row in rows if str(row.get("text", "")).strip()]

    return texts, payload, rows


def parse_options_json(raw_options: str | None) -> dict[str, Any]:
    if not raw_options:
        return {}
    import json

    try:
        data = json.loads(raw_options)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"options must be valid JSON: {exc.msg}")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="options must be a JSON object")
    return data


def save_uploaded_payload(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
