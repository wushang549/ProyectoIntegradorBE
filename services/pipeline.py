"""End-to-end analysis pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from models.schemas import AnalysisOptions
from services.clustering import (
    auto_cluster_with_buckets,
    choose_auto_k,
    cut_clusters,
    evaluate_cluster_partition,
    normalize_k_clusters,
)
from services.embeddings import compute_embeddings
from services.granulation import granulate_records
from services.hierarchy import build_hierarchy, enrich_hierarchy_nodes
from services.ingestion import IngestionPayload, ingest
from services.insights import build_insights
from services.labeling import apply_labels
from services.storage import (
    clusters_file,
    embeddings_file,
    ensure_analysis_dir,
    hierarchy_file,
    insights_file,
    items_file,
    label_cache_file,
    now_utc_iso,
    overview_file,
    umap_file,
    write_embeddings,
    write_json_file,
)
from services.summaries import build_cluster_summaries
from utils.text_utils import build_preview

ProgressCallback = Callable[[str, int, str, str, int | None, int | None], None]


@dataclass(slots=True)
class PipelinePayload:
    """Raw payload accepted by the background pipeline."""

    input_type: str
    text: str | None = None
    csv_bytes: bytes | None = None
    filename: str | None = None


def run_analysis_pipeline(
    analysis_id: str,
    payload: PipelinePayload,
    options: AnalysisOptions,
    progress: ProgressCallback,
) -> dict[str, Any]:
    """Run all analysis stages and persist artifacts to disk."""

    ensure_analysis_dir(analysis_id)
    options.granulate_return_items = True

    progress("ingestion", 10, "Ingestion", "Reading input rows.", None, None)
    records = ingest(
        IngestionPayload(
            input_type=payload.input_type,
            text=payload.text,
            csv_bytes=payload.csv_bytes,
            filename=payload.filename,
        )
    )

    progress(
        "granulation",
        20,
        "Granulation",
        "Generating granular items.",
        len(records),
        None,
    )
    granulated_items = granulate_records(records, granulate=options.granulate)
    items_as_dicts = [item.model_dump() for item in granulated_items]
    write_json_file(
        items_file(analysis_id),
        {
            "analysis_id": analysis_id,
            "total_records": len(records),
            "total_items": len(items_as_dicts),
            "items": items_as_dicts,
        },
    )

    progress(
        "embeddings",
        35,
        "Embeddings",
        "Computing embedding vectors.",
        len(records),
        len(items_as_dicts),
    )
    texts = [item["text"] for item in items_as_dicts]
    embedding_result = compute_embeddings(texts)
    vectors = embedding_result.vectors
    write_embeddings(embeddings_file(analysis_id), vectors)

    progress(
        "hierarchy",
        50,
        "Hierarchy",
        "Building hierarchical clustering tree.",
        len(records),
        len(items_as_dicts),
    )
    hierarchy_data = build_hierarchy(vectors)
    write_json_file(hierarchy_file(analysis_id), hierarchy_data)

    progress(
        "clusters",
        60,
        "Clusters",
        "Cutting hierarchical tree into flat clusters.",
        len(records),
        len(items_as_dicts),
    )
    n_items = len(items_as_dicts)
    linkage_matrix = np.asarray(hierarchy_data.get("linkage_matrix", []), dtype=float)
    if options.k_clusters is None:
        auto_partition = auto_cluster_with_buckets(vectors=vectors, items=items_as_dicts)
        cluster_labels = np.asarray(auto_partition.get("cluster_labels", []), dtype=int)
        if cluster_labels.size != n_items:
            auto_k = choose_auto_k(linkage_matrix=linkage_matrix, vectors=vectors, n_items=n_items)
            default_k = int(auto_k.get("k_clusters", normalize_k_clusters(None, n_items)))
            cluster_labels = cut_clusters(linkage_matrix, n_items=n_items, k_clusters=default_k)
            k_selection = {
                "mode": "auto",
                "strategy": "global_fallback",
                "selected_k": default_k,
                "target_k": int(auto_k.get("target_k", default_k)),
                "selected_quality": auto_k.get("selected_quality", {}),
                "candidates": auto_k.get("candidates", []),
            }
        else:
            default_k = int(auto_partition.get("k_clusters", len(np.unique(cluster_labels))))
            if default_k <= 0:
                default_k = int(len(np.unique(cluster_labels)))
            k_selection = {
                "mode": "auto",
                "strategy": "bucketed_theme_partition",
                "selected_k": int(default_k),
                "bucket_diagnostics": auto_partition.get("bucket_diagnostics", []),
            }
    else:
        requested_k = int(options.k_clusters)
        default_k = normalize_k_clusters(requested_k, n_items)
        cluster_labels = cut_clusters(linkage_matrix, n_items=n_items, k_clusters=default_k)
        k_selection = {
            "mode": "manual",
            "requested_k": requested_k,
            "selected_k": default_k,
        }
    if cluster_labels.size:
        default_k = int(len(np.unique(cluster_labels)))
        k_selection["selected_k"] = int(default_k)

    cluster_quality = evaluate_cluster_partition(vectors=vectors, cluster_labels=cluster_labels)
    cluster_quality["selected_k"] = int(default_k)
    cluster_summaries = build_cluster_summaries(items_as_dicts, vectors, cluster_labels)

    progress(
        "umap",
        70,
        "UMAP",
        "Projecting embeddings to 2D coordinates.",
        len(records),
        len(items_as_dicts),
    )
    umap_points = _compute_umap_points(items_as_dicts, vectors)
    write_json_file(
        umap_file(analysis_id),
        {
            "analysis_id": analysis_id,
            "points": umap_points,
        },
    )

    progress(
        "labeling",
        80,
        "Labeling",
        "Generating cluster labels.",
        len(records),
        len(items_as_dicts),
    )
    labeled_clusters = apply_labels(
        cluster_summaries=cluster_summaries,
        k_clusters=default_k,
        cache_path=label_cache_file(analysis_id),
    )
    hierarchy_data = enrich_hierarchy_nodes(
        hierarchy_data=hierarchy_data,
        items=items_as_dicts,
        cluster_labels=cluster_labels,
        clusters=labeled_clusters,
    )
    write_json_file(hierarchy_file(analysis_id), hierarchy_data)

    progress(
        "summaries",
        90,
        "Summaries",
        "Computing overview and cluster summaries.",
        len(records),
        len(items_as_dicts),
    )
    cluster_distribution = _build_cluster_distribution(labeled_clusters, total_items=n_items)
    write_json_file(
        clusters_file(analysis_id),
        {
            "analysis_id": analysis_id,
            "k_clusters": default_k,
            "cluster_labels": cluster_labels.tolist(),
            "clusters": labeled_clusters,
        },
    )
    write_json_file(
        overview_file(analysis_id),
        {
            "analysis_id": analysis_id,
            "created_at": now_utc_iso(),
            "total_records": len(records),
            "total_items": n_items,
            "k_clusters": default_k,
            "embedding_method": embedding_result.method,
            "k_selection": k_selection,
            "cluster_quality": cluster_quality,
            "cluster_distribution": cluster_distribution,
        },
    )

    progress(
        "insights",
        97,
        "Insights",
        "Generating insight heuristics.",
        len(records),
        len(items_as_dicts),
    )
    insights_data = build_insights(
        total_items=n_items,
        clusters=labeled_clusters,
        embedding_method=embedding_result.method,
        cluster_quality=cluster_quality,
    )
    write_json_file(insights_file(analysis_id), insights_data)

    progress(
        "completed",
        100,
        "Completed",
        "Analysis completed.",
        len(records),
        len(items_as_dicts),
    )

    return {
        "analysis_id": analysis_id,
        "total_records": len(records),
        "total_items": n_items,
        "k_clusters": default_k,
    }


def _build_cluster_distribution(
    clusters: list[dict[str, Any]],
    total_items: int,
) -> list[dict[str, Any]]:
    """Build percentage distribution per cluster."""

    if total_items <= 0:
        return []

    result: list[dict[str, Any]] = []
    for cluster in sorted(clusters, key=lambda item: item["size"], reverse=True):
        size = int(cluster["size"])
        result.append(
            {
                "cluster_id": int(cluster["cluster_id"]),
                "label": str(cluster.get("label") or f"Cluster {cluster['cluster_id']}"),
                "size": size,
                "pct": round((size / total_items) * 100.0, 3),
            }
        )
    return result


def _compute_umap_points(items: list[dict[str, Any]], vectors: np.ndarray) -> list[dict[str, Any]]:
    """Compute UMAP 2D coordinates and scaled coordinates."""

    n_items = len(items)
    if n_items == 0:
        return []

    if n_items == 1:
        return [
            {
                "id": items[0]["id"],
                "x_raw": 0.0,
                "y_raw": 0.0,
                "x": 0.5,
                "y": 0.5,
            }
        ]

    projection = _project_umap(vectors)
    scaled_x = _minmax_scale(projection[:, 0])
    scaled_y = _minmax_scale(projection[:, 1])

    points: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        points.append(
            {
                "id": item["id"],
                "x_raw": float(projection[idx, 0]),
                "y_raw": float(projection[idx, 1]),
                "x": float(scaled_x[idx]),
                "y": float(scaled_y[idx]),
                "preview": build_preview(str(item.get("text", ""))),
            }
        )
    return points


def _project_umap(vectors: np.ndarray) -> np.ndarray:
    """Project vectors to 2D with deterministic UMAP fallback."""

    try:
        from umap import UMAP  # Local import so app can boot without optional dependency installed.

        n_neighbors = min(15, max(2, vectors.shape[0] - 1))
        reducer = UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
        )
        projected = reducer.fit_transform(vectors)
        return projected.astype(np.float32)
    except Exception:
        centered = vectors - vectors.mean(axis=0, keepdims=True)
        u_matrix, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        components = u_matrix[:, :2] * singular_values[:2]
        if components.shape[1] < 2:
            components = np.pad(components, ((0, 0), (0, 2 - components.shape[1])), mode="constant")
        return components.astype(np.float32)


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    """Scale one coordinate axis to [0, 1]."""

    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-12:
        return np.full_like(values, 0.5, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)
