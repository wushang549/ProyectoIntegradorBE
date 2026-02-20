from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

MAX_CSV_SIZE_BYTES = 50 * 1024 * 1024
TEXT_COLUMN_ALIAS_PRIORITY = ("text", "transcript", "message", "content", "review", "comment", "body", "feedback")


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


def extract_texts_from_csv_payload(payload: bytes, max_rows: int | None = None) -> tuple[list[str], str]:
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

    normalized_rows: list[dict[str, str]] = []
    for row in reader:
        normalized = {"text": (row.get(source_column) or "").strip()}
        if not normalized["text"]:
            continue
        normalized_rows.append(normalized)
        if row_limit is not None and len(normalized_rows) >= row_limit:
            break

    texts = [row["text"] for row in normalized_rows]
    if not texts:
        raise HTTPException(
            status_code=400,
            detail="CSV 'text' column must include at least 1 non-empty row",
        )
    return texts, source_column


def validate_csv_upload(file: UploadFile | None, max_rows: int | None = None) -> tuple[list[str], bytes]:
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

    texts, _source_column = extract_texts_from_csv_payload(payload, max_rows=max_rows)

    return texts, payload


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
