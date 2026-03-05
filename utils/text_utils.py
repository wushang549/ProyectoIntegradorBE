"""Text utility helpers for normalization and light segmentation."""

from __future__ import annotations

import re
from typing import Iterable

_CONJUNCTION_PATTERN = re.compile(
    r"\b(?:but|however|though|pero|aunque|sin\s+embargo)\b",
    flags=re.IGNORECASE,
)
_SEPARATOR_PATTERN = re.compile(r"[.!?;]+")
_SPACE_PATTERN = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Apply minimal normalization: trim + collapse spaces."""

    collapsed = _SPACE_PATTERN.sub(" ", value or "")
    return collapsed.strip()


def has_granulation_separator(text: str) -> bool:
    """Return True when the input contains split separators."""

    return bool(_SEPARATOR_PATTERN.search(text) or _CONJUNCTION_PATTERN.search(text))


def has_contrastive_conjunction(text: str) -> bool:
    """Return True when a contrastive conjunction is present (e.g., 'but')."""

    return bool(_CONJUNCTION_PATTERN.search(text))


def split_contrastive_pair(text: str, min_words_each: int = 4) -> list[str]:
    """Split into two parts around the first contrastive conjunction when meaningful."""

    if not has_contrastive_conjunction(text):
        return []

    marked = _CONJUNCTION_PATTERN.sub(" || ", text, count=1)
    raw_parts = re.split(r"\s*\|\|\s*", marked, maxsplit=1)
    parts = [normalize_text(part) for part in raw_parts if normalize_text(part)]
    if len(parts) != 2:
        return []

    if any(len(part.split()) < min_words_each for part in parts):
        return []
    return parts


def split_clauses(text: str) -> list[str]:
    """Split text into light clauses using punctuation and conjunctions."""

    marked = _CONJUNCTION_PATTERN.sub(" || ", text)
    raw_parts = re.split(r"\s*(?:[.!?;]+|\|\|)\s*", marked)
    return [normalize_text(part) for part in raw_parts if normalize_text(part)]


def normalize_rows(rows: Iterable[str]) -> list[str]:
    """Normalize row values and remove empty rows."""

    normalized = [normalize_text(row) for row in rows]
    return [row for row in normalized if row]


def build_preview(text: str, max_len: int = 120) -> str:
    """Normalize text for preview fields without truncating content."""

    _ = max_len  # kept for backward-compatible function signature
    return normalize_text(text)
