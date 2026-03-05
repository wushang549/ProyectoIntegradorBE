"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AnalysisOptions(BaseModel):
    """Runtime options for the analysis pipeline."""

    model_config = ConfigDict(extra="ignore")

    k_clusters: int | None = Field(default=None, ge=2, le=100)
    granulate: bool = True
    granulate_return_items: bool = True


class TabLinks(BaseModel):
    """Static links returned after analysis creation."""

    overview: str
    map: str
    clusters: str
    granulate: str
    hierarchy: str
    insights: str
    status: str


class CreateAnalysisResponse(BaseModel):
    """Response payload returned by POST /analysis."""

    analysis_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    created_at: datetime
    tabs: TabLinks


class RecentAnalysisItem(BaseModel):
    """Entry used for /analysis/recent responses."""

    analysis_id: str
    created_at: datetime
    updated_at: datetime
    status: Literal["queued", "processing", "completed", "failed"]
    stage: str
    pct: int
    total_records: int = 0
    total_items: int = 0


class AnalysisStatusResponse(BaseModel):
    """Status information for async job polling."""

    analysis_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    stage: str
    pct: int
    stage_label: str
    message: str
    elapsed_sec: float
    created_at: datetime
    updated_at: datetime
    error: str | None = None


class IngestedRecord(BaseModel):
    """Input record after ingestion stage."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class GranulatedItem(BaseModel):
    """Single text item used for embedding and clustering."""

    id: str
    text: str
    source_id: str
    source_text: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class HierarchyNodeResponse(BaseModel):
    """Hierarchy node payload consumed by frontend tree components."""

    node_id: str
    parent_id: str | None
    children_ids: list[str] = Field(default_factory=list)
    size: int
    height: float
    label: str | None = None
    cohesion: float | None = None
    similarity: float | None = None
    descendant_leaf_count: int | None = None
    dominant_cluster_id: int | None = None
    dominant_cluster_share: float | None = None
    summary: str | None = None


class HierarchyLeafResponse(BaseModel):
    """Leaf reference used by frontend tree interactions."""

    id: str
    node_id: str
    cluster_id: int


class HierarchyResponse(BaseModel):
    """Hierarchy endpoint response contract."""

    root_id: str
    nodes: dict[str, HierarchyNodeResponse]
    leaves: list[HierarchyLeafResponse]
