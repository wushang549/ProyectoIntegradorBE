from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AnalysisOptions(BaseModel):
    k_clusters: int = Field(default=8, ge=2, le=100)
    umap_n_neighbors: int = Field(default=15, ge=2, le=200)
    umap_min_dist: float = Field(default=0.1, ge=0.0, le=0.99)
    granulate: bool = True
    granulate_max_rows: int = Field(default=200, ge=1, le=5000)
    granulate_return_items: bool = False
    label_internal_nodes: bool = True


class CreateAnalysisResponse(BaseModel):
    analysis_id: str
    status: Literal["queued", "processing"]
    created_at: datetime
    tabs: dict[str, str]


class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress: dict[str, Any]
    error: str | None = None


class OverviewResponse(BaseModel):
    counts: dict[str, int]
    top_clusters: list[dict[str, Any]]
    top_aspects: list[dict[str, Any]]
    timing: dict[str, float]


class MapResponse(BaseModel):
    points: list[dict[str, Any]]
    clusters: list[dict[str, Any]]


class ClustersResponse(BaseModel):
    clusters: list[dict[str, Any]]


class GranulateResponse(BaseModel):
    mode: Literal["text", "csv"]
    aggregate_aspect_summary: list[dict[str, Any]]
    items_included: int
    items_total: int
    items: list[dict[str, Any]] = Field(default_factory=list)


class HierarchyResponse(BaseModel):
    root_id: str
    nodes: dict[str, dict[str, Any]]
    leaves: list[dict[str, Any]]
