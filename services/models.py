from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AnalysisOptions(BaseModel):
    k_clusters: int = Field(default=8, ge=2, le=100)
    umap_n_neighbors: int = Field(default=15, ge=2, le=200)
    umap_min_dist: float = Field(default=0.1, ge=0.0, le=0.99)
    umap_scale_points: bool = True
    umap_scale_clamp: float = Field(default=3.0, ge=1.5, le=6.0)
    granulate: bool = True
    granulate_max_rows: int = Field(default=200, ge=1, le=5000)
    granulate_per_cluster: bool = False
    granulate_per_cluster_max_rows: int = Field(default=80, ge=1, le=500)
    granulate_return_items: bool = False
    label_internal_nodes: bool = True
    llm_label_budget: int = Field(default=120, ge=0, le=10000)
    preview_char_limit: int = Field(default=320, ge=120, le=1200)
    insights_theme_count: int = Field(default=5, ge=3, le=8)


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
    advanced: dict[str, Any] | None = None


class ClustersResponse(BaseModel):
    clusters: list[dict[str, Any]]


class GranulateResponse(BaseModel):
    mode: Literal["text", "csv"]
    aggregate_aspect_summary: list[dict[str, Any]]
    per_cluster_aggregate: list[dict[str, Any]] = Field(default_factory=list)
    items_included: int
    items_total: int
    items: list[dict[str, Any]] = Field(default_factory=list)


class HierarchyResponse(BaseModel):
    root_id: str
    nodes: dict[str, dict[str, Any]]
    leaves: list[dict[str, Any]]


class InsightsResponse(BaseModel):
    key_findings: list[str]
    theme_summary: list[dict[str, Any]]
    quality_warnings: list[str]


class RecentAnalysisEntry(BaseModel):
    analysis_id: str
    status: str
    created_at: str | None = None
    item_count: int = 0
