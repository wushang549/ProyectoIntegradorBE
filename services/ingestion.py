"""Ingestion services for text and CSV inputs."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any

from models.schemas import IngestedRecord
from utils.text_utils import normalize_rows, normalize_text


ACCEPTED_TEXT_COLUMNS = [
    "text",
    "reviews",
    "review",
    "comments",
    "comment",
    "content",
    "messages",
    "message",
    "information",
    "info",
]
_COLUMN_PRIORITY = {name: index for index, name in enumerate(ACCEPTED_TEXT_COLUMNS)}


class IngestionError(ValueError):
    """Raised when source input cannot be ingested."""


@dataclass(slots=True)
class IngestionPayload:
    """Raw request payload used by ingestion stage."""

    input_type: str
    text: str | None = None
    csv_bytes: bytes | None = None
    filename: str | None = None


def ingest(payload: IngestionPayload) -> list[IngestedRecord]:
    """Ingest records from text or CSV source."""

    if payload.input_type == "text":
        if payload.text is None:
            raise IngestionError("The 'text' field is required when input_type='text'.")
        return ingest_text(payload.text)

    if payload.input_type == "csv":
        if not payload.csv_bytes:
            raise IngestionError("The 'file' field is required when input_type='csv'.")
        return ingest_csv(payload.csv_bytes)

    raise IngestionError("Invalid input_type. Allowed values are 'text' and 'csv'.")


def ingest_text(text: str) -> list[IngestedRecord]:
    """Ingest plain text where each non-empty line is a record."""

    rows = normalize_rows(text.splitlines() if "\n" in text else [text])
    if not rows:
        raise IngestionError("No non-empty text rows were provided.")

    records: list[IngestedRecord] = []
    for idx, row in enumerate(rows):
        records.append(IngestedRecord(id=f"row_{idx}", text=row, metadata={}))
    return records


def ingest_csv(csv_bytes: bytes) -> list[IngestedRecord]:
    """Ingest CSV bytes and auto-detect the text column."""

    decoded = _decode_csv_bytes(csv_bytes)
    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise IngestionError("CSV must include a header row.")

    text_column = _detect_text_column(reader.fieldnames)
    if not text_column:
        accepted = ", ".join(ACCEPTED_TEXT_COLUMNS)
        raise IngestionError(
            "CSV text column not found. Accepted column names (case-insensitive): "
            f"{accepted}."
        )

    records: list[IngestedRecord] = []
    for row_idx, row in enumerate(reader):
        raw_value = row.get(text_column, "")
        row_text = normalize_text(str(raw_value))
        if not row_text:
            continue

        metadata: dict[str, Any] = {}
        for key, value in row.items():
            if key is None or key == text_column:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value

        records.append(
            IngestedRecord(
                id=f"row_{row_idx}",
                text=row_text,
                metadata=metadata,
            )
        )

    if not records:
        raise IngestionError("CSV does not contain non-empty text rows.")

    return records


def _decode_csv_bytes(csv_bytes: bytes) -> str:
    """Decode CSV bytes trying UTF-8 first, then latin-1."""

    try:
        return csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return csv_bytes.decode("latin-1")


def _normalize_column_name(column: str) -> str:
    """Normalize CSV headers for matching."""

    return normalize_text(column).lower().replace("-", "_").replace(" ", "_")


def _alias_candidates(column: str) -> set[str]:
    """Build singular/plural candidates for a column name."""

    base = _normalize_column_name(column)
    aliases = {base}

    if base.endswith("s"):
        aliases.add(base[:-1])
    else:
        aliases.add(f"{base}s")

    if base == "infos":
        aliases.add("info")
    if base == "messages":
        aliases.add("message")
    if base == "reviews":
        aliases.add("review")
    if base == "comments":
        aliases.add("comment")

    return aliases


def _detect_text_column(fieldnames: list[str]) -> str | None:
    """Detect the best text column based on configured priority."""

    best_priority = 10_000
    best_column: str | None = None

    for original in fieldnames:
        aliases = _alias_candidates(original)
        for alias in aliases:
            if alias not in _COLUMN_PRIORITY:
                continue
            priority = _COLUMN_PRIORITY[alias]
            if priority < best_priority:
                best_priority = priority
                best_column = original

    return best_column
