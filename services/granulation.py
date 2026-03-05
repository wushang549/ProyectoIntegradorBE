"""Granulation stage for light comment segmentation."""

from __future__ import annotations

from models.schemas import GranulatedItem, IngestedRecord
from services.thematics import attach_theme_metadata
from utils.text_utils import (
    has_granulation_separator,
    split_clauses,
    split_contrastive_pair,
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
    """Apply minimal segmentation according to project rules."""

    if not granulate:
        return [text]

    contrastive_parts = split_contrastive_pair(text, min_words_each=4)
    if contrastive_parts and _should_split_contrastive(text, contrastive_parts):
        return contrastive_parts

    word_count = len(text.split())
    if word_count <= 24 and not has_granulation_separator(text):
        return [text]

    contains_separator = has_granulation_separator(text)
    separator_count = sum(text.count(token) for token in ".!?;")
    if len(text) <= 220 and (not contains_separator or separator_count <= 1):
        return [text]

    clauses = split_clauses(text)
    merged = _merge_short_clauses(clauses, min_words=4)
    if len(merged) <= 1:
        return [text]
    if len(merged) > 3 and word_count < 45:
        return [text]
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
