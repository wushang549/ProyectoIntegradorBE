"""Internal lightweight types used across service modules."""

from __future__ import annotations

from typing import Any, TypedDict


class ItemDict(TypedDict):
    """Cluster-ready item representation."""

    id: str
    text: str
    source_id: str
    source_text: str
    chunk_index: int
    metadata: dict[str, Any]


class ClusterSummaryDict(TypedDict, total=False):
    """Cluster summary structure used in artifacts and API responses."""

    cluster_id: int
    size: int
    top_terms: list[str]
    representatives: list[str]
    representative_ids: list[str]
    label: str
