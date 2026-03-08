"""Granulation stage for light comment segmentation."""

from __future__ import annotations

import re

from models.schemas import GranulatedItem, IngestedRecord
from services.thematics import attach_theme_metadata
from utils.text_utils import (
    has_granulation_separator,
    normalize_text,
    split_contrastive_pair,
    split_sentences,
)

_LEADING_MARKER_PATTERNS = (
    re.compile(r"^(?:on the positive side|on the plus side|to be fair)\s*,\s*", flags=re.IGNORECASE),
)
_ORDER_PATTERNS = (
    (
        re.compile(r"^(?:we tried|i ordered)\s+the\s+(.+?)\s+(?:and|but)\s+it\s+was\s+(.+)$", flags=re.IGNORECASE),
        r"The \1 was \2",
    ),
    (
        re.compile(r"^(?:we tried|i ordered)\s+the\s+(.+?)\s+it\s+was\s+(.+)$", flags=re.IGNORECASE),
        r"The \1 was \2",
    ),
)


def granulate_records(records: list[IngestedRecord], granulate: bool = True) -> list[GranulatedItem]:
    """Convert records into cluster-ready items."""

    items: list[GranulatedItem] = []

    for record in records:
        parts = _segment_record(record.text, granulate=granulate)
        for chunk_index, chunk in enumerate(parts):
            item_id = record.id if len(parts) == 1 else f"{record.id}_g{chunk_index}"
            item_metadata = attach_theme_metadata(metadata=record.metadata, text=chunk)
            items.append(
                GranulatedItem(
                    id=item_id,
                    text=chunk,
                    source_id=record.id,
                    source_text=record.text,
                    chunk_index=chunk_index,
                    metadata=item_metadata,
                )
            )

    return items


def _segment_record(text: str, granulate: bool) -> list[str]:
    """Apply sentence-first segmentation tuned for multi-aspect reviews."""

    normalized = normalize_text(text)
    if not normalized:
        return []

    if not granulate:
        return [normalized]

    word_count = len(normalized.split())
    if word_count <= 18 and not has_granulation_separator(normalized):
        return [_normalize_segment_text(normalized)]

    sentences = split_sentences(normalized)
    if not sentences:
        return [_normalize_segment_text(normalized)]

    segments: list[str] = []
    for sentence in sentences:
        if _matches_order_pattern(sentence):
            segments.append(sentence)
            continue

        contrastive_parts = split_contrastive_pair(sentence, min_words_each=4)
        if contrastive_parts and _should_split_contrastive(sentence, contrastive_parts):
            segments.extend(contrastive_parts)
            continue
        segments.append(sentence)

    cleaned = [_normalize_segment_text(part) for part in segments]
    cleaned = [part for part in cleaned if part]
    merged = _merge_short_clauses(cleaned, min_words=4)
    merged = _cap_segment_count(merged, limit=6)
    if len(merged) <= 1:
        return [merged[0]] if merged else [normalized]
    return merged


def _merge_short_clauses(parts: list[str], min_words: int = 4) -> list[str]:
    """Merge very short clauses with neighbors to avoid tiny outlier fragments."""

    if not parts:
        return []

    merged: list[str] = []
    for part in parts:
        token_count = len(part.split())
        if token_count < min_words and merged:
            merged[-1] = f"{merged[-1]} {part}".strip()
            continue
        merged.append(part)

    if len(merged) >= 2 and len(merged[-1].split()) < min_words:
        merged[-2] = f"{merged[-2]} {merged[-1]}".strip()
        merged.pop()

    return merged


def _cap_segment_count(parts: list[str], limit: int = 6) -> list[str]:
    """Merge adjacent short parts until the segment count stays manageable."""

    merged = [part for part in parts if part]
    if limit <= 0 or len(merged) <= limit:
        return merged

    while len(merged) > limit:
        merge_at = 0
        best_size = None
        for idx in range(len(merged) - 1):
            combined = len(merged[idx].split()) + len(merged[idx + 1].split())
            if best_size is None or combined < best_size:
                best_size = combined
                merge_at = idx
        merged[merge_at] = f"{merged[merge_at]} {merged[merge_at + 1]}".strip()
        merged.pop(merge_at + 1)

    return merged


def _normalize_segment_text(text: str) -> str:
    """Clean discourse scaffolding while preserving the original meaning."""

    cleaned = normalize_text(text).strip(" \"'")
    if not cleaned:
        return ""

    for pattern in _LEADING_MARKER_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    for pattern, replacement in _ORDER_PATTERNS:
        updated = pattern.sub(replacement, cleaned)
        if updated != cleaned:
            cleaned = updated
            break

    lowered = cleaned.lower()
    if lowered.startswith("parts of it were also "):
        cleaned = f"Some parts were {cleaned[22:]}"
    elif lowered.startswith("parts of it were "):
        cleaned = f"Some parts were {cleaned[17:]}"
    elif lowered.startswith("felt "):
        cleaned = f"It {cleaned}"
    elif lowered.startswith("looked "):
        cleaned = f"It {cleaned}"
    elif lowered.startswith("was "):
        cleaned = f"It {cleaned}"
    elif lowered.startswith("were "):
        cleaned = f"They {cleaned}"

    cleaned = re.sub(r"\s*,\s*$", "", cleaned)
    cleaned = normalize_text(cleaned).strip(" ,;:-")
    return cleaned


def _matches_order_pattern(text: str) -> bool:
    """Return True when the sentence should stay intact for later normalization."""

    candidate = normalize_text(text).strip(" \"'")
    return any(pattern.match(candidate) for pattern, _ in _ORDER_PATTERNS)


def _should_split_contrastive(source_text: str, parts: list[str]) -> bool:
    """Only split contrastive statements when both sides carry enough content."""

    if len(parts) != 2:
        return False

    total_words = max(1, len(source_text.split()))
    part_sizes = [len(part.split()) for part in parts]
    if any(size < 4 for size in part_sizes):
        return False

    # Avoid splitting when one side is only a tiny tail.
    return all((size / total_words) >= 0.25 for size in part_sizes)
