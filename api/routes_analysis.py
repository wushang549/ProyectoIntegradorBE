"""Analysis API routes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile

from jobs import worker
from models.schemas import (
    AnalysisOptions,
    CreateAnalysisResponse,
    HierarchyLeafResponse,
    HierarchyNodeResponse,
    HierarchyResponse,
    TabLinks,
)
from services.clustering import cut_clusters, normalize_k_clusters
from services.hierarchy import enrich_hierarchy_nodes, refine_hierarchy_labels_for_nodes
from services.insights import build_overall_summary, build_overall_summary_fallback
from services.pipeline import PipelinePayload, run_analysis_pipeline
from services.storage import (
    analysis_exists,
    artifacts_ready,
    clusters_file,
    embeddings_file,
    hierarchy_file,
    hierarchy_label_cache_file,
    insights_file,
    items_file,
    label_cache_file,
    list_recent,
    overview_file,
    read_embeddings,
    read_json_file,
    umap_file,
    write_json_file,
)
from services.summaries import build_cluster_summaries
from services.labeling import (
    LabelingError,
    apply_labels,
    default_ollama_model,
    list_ollama_models,
    normalize_requested_ollama_model,
    validate_requested_ollama_model,
)
from utils.text_utils import build_preview

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("", response_model=CreateAnalysisResponse)
async def create_analysis(
    input_type: str = Form(...),
    text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    options: str | None = Form(default=None),
) -> CreateAnalysisResponse:
    """Create an asynchronous NLP analysis job."""

    source_type = input_type.strip().lower()
    if source_type not in {"text", "csv"}:
        raise HTTPException(status_code=422, detail="input_type must be 'text' or 'csv'.")

    parsed_options = _parse_options(options)
    parsed_options.granulate_return_items = True

    if source_type == "text" and not (text or "").strip():
        raise HTTPException(status_code=422, detail="The 'text' field is required for input_type='text'.")

    csv_bytes: bytes | None = None
    filename: str | None = None
    if source_type == "csv":
        if file is None:
            raise HTTPException(status_code=422, detail="The 'file' field is required for input_type='csv'.")
        csv_bytes = await file.read()
        if not csv_bytes:
            raise HTTPException(status_code=422, detail="Uploaded CSV file is empty.")
        filename = file.filename

    analysis_id = uuid4().hex
    created_at = datetime.now(timezone.utc)

    payload = PipelinePayload(
        input_type=source_type,
        text=text,
        csv_bytes=csv_bytes,
        filename=filename,
    )

    worker.create_job(analysis_id)

    def _run() -> None:
        worker.set_processing(analysis_id)
        try:
            run_analysis_pipeline(
                analysis_id=analysis_id,
                payload=payload,
                options=parsed_options,
                progress=lambda stage, pct, stage_label, message, total_records, total_items: worker.update_progress(
                    analysis_id,
                    stage=stage,
                    pct=pct,
                    stage_label=stage_label,
                    message=message,
                    total_records=total_records,
                    total_items=total_items,
                ),
            )
            worker.set_completed(analysis_id)
        except Exception as exc:  # pragma: no cover - defensive safety for worker thread
            worker.set_failed(analysis_id, str(exc))

    worker.submit_job(analysis_id, _run)

    base = f"/v1/analysis/{analysis_id}"
    tabs = TabLinks(
        overview=f"{base}/overview",
        map=f"{base}/map",
        clusters=f"{base}/clusters",
        granulate=f"{base}/granulate",
        hierarchy=f"{base}/hierarchy",
        insights=f"{base}/insights",
        status=f"{base}/status",
    )

    return CreateAnalysisResponse(
        analysis_id=analysis_id,
        status="queued",
        created_at=created_at,
        tabs=tabs,
    )


@router.get("/models")
def analysis_models() -> dict[str, Any]:
    """Return installed Ollama models for frontend selectors."""

    default_model = default_ollama_model()
    models = []
    for entry in list_ollama_models():
        if not isinstance(entry, dict):
            continue
        model_name = str(entry.get("name") or "").strip()
        if not model_name:
            continue
        models.append(
            {
                "name": model_name,
                "id": str(entry.get("id") or "").strip(),
                "size": str(entry.get("size") or "").strip(),
                "modified": str(entry.get("modified") or "").strip(),
                "is_default": model_name == default_model,
            }
        )

    return {
        "default_model": default_model,
        "models": models,
    }


@router.get("/recent")
def recent_analyses(limit: int = Query(default=10, ge=1, le=100)) -> dict[str, Any]:
    """Return recently submitted analyses."""

    items = []
    for entry in list_recent(limit=limit):
        if not isinstance(entry, dict):
            continue
        raw_stage = str(entry.get("stage") or "").strip().lower() or "queued"
        normalized = dict(entry)
        normalized["raw_stage"] = raw_stage
        normalized["stage"] = _normalize_run_stage(raw_stage)
        items.append(normalized)

    return {"items": items}


@router.get("/{analysis_id}/status")
def analysis_status(analysis_id: str) -> dict[str, Any]:
    """Return asynchronous status/progress for one analysis."""

    payload = worker.get_job_payload(analysis_id)
    if payload is not None:
        updated_at = payload["updated_at"].isoformat() if hasattr(payload["updated_at"], "isoformat") else str(payload["updated_at"])
        pct = int(payload.get("pct", 0))
        raw_stage = str(payload.get("stage") or "").strip().lower() or "queued"
        stage = _normalize_run_stage(raw_stage)
        return {
            "analysis_id": analysis_id,
            "status": payload.get("status", "processing"),
            "progress": {
                "stage": stage,
                "raw_stage": raw_stage,
                "pct": pct,
                "stage_label": payload.get("stage_label", "Processing"),
                "message": payload.get("message", "Processing analysis"),
                "stage_pct": pct,
                "elapsed_sec": float(payload.get("elapsed_sec", 0.0)),
            },
            "error": payload.get("error"),
            "debug_error": payload.get("error"),
            "updated_at": updated_at,
        }

    if not analysis_exists(analysis_id):
        raise HTTPException(status_code=404, detail="Analysis id not found.")

    if artifacts_ready(analysis_id):
        now = datetime.now(timezone.utc).isoformat()
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "progress": {
                "stage": "completed",
                "raw_stage": "completed",
                "pct": 100,
                "stage_label": "Completed",
                "message": "Analysis complete",
                "stage_pct": 100,
                "elapsed_sec": 0.0,
            },
            "error": None,
            "debug_error": None,
            "updated_at": now,
        }

    raise HTTPException(status_code=409, detail="Analysis is still processing.")


@router.get("/{analysis_id}/overview")
def analysis_overview(analysis_id: str) -> dict[str, Any]:
    """Return overview artifact."""

    _require_ready(analysis_id, str(overview_file(analysis_id).name))
    overview_data = read_json_file(overview_file(analysis_id))
    items_payload = read_json_file(items_file(analysis_id))
    cluster_payload = _build_cluster_payload(analysis_id=analysis_id, requested_k=None)
    timing = _load_timing_payload(analysis_id)

    top_clusters = sorted(cluster_payload["clusters"], key=lambda cluster: int(cluster["size"]), reverse=True)[:5]
    top_aspects = _extract_top_aspects(analysis_data=overview_data)

    items = _extract_items(items_payload)
    return {
        "counts": {
            "items": len(items),
            "clusters": len(cluster_payload["clusters"]),
            "aspects": len(top_aspects),
        },
        "top_clusters": top_clusters,
        "top_aspects": top_aspects,
        "runtime": {
            "embedding_method": str(overview_data.get("embedding_method") or ""),
            "llm_model": _load_analysis_llm_model(analysis_id, overview_data=overview_data),
        },
        "timing": timing,
    }


@router.get("/{analysis_id}/granulate")
def analysis_granulate(
    analysis_id: str,
    include_items: bool = Query(default=True),
) -> Any:
    """Return granulated items artifact."""

    _require_ready(analysis_id, str(items_file(analysis_id).name))
    data = read_json_file(items_file(analysis_id))
    items = _extract_items(data)

    if include_items:
        return [
            {
                "id": item.get("id"),
                "preview": build_preview(str(item.get("text", ""))),
                "result": {
                    "text": str(item.get("text", "")),
                    "units": [str(item.get("text", ""))] if str(item.get("text", "")).strip() else [],
                    "granules": [],
                    "taxonomy": [],
                    "detected_taxonomy": "general",
                    "taxonomy_candidates": [],
                    "detection_margin": 0.0,
                    "aspect_summary": _item_aspect_summary(item),
                    "highlights": [],
                },
            }
            for item in items
        ]

    mode = "csv" if any(bool(item.get("metadata")) for item in items if isinstance(item, dict)) else "text"
    item_ids = [str(item.get("id")) for item in items if isinstance(item, dict) and item.get("id") is not None]
    return {
        "mode": mode,
        "aggregate_aspect_summary": [],
        "per_cluster_aggregate": [],
        "items_included": len(item_ids),
        "items_total": len(item_ids),
        "item_ids_included": item_ids,
    }


@router.get("/{analysis_id}/hierarchy", response_model=HierarchyResponse)
def analysis_hierarchy(analysis_id: str) -> HierarchyResponse:
    """Return hierarchy artifact."""

    _require_ready(analysis_id, str(hierarchy_file(analysis_id).name))
    hierarchy_data = read_json_file(hierarchy_file(analysis_id))
    if not isinstance(hierarchy_data, dict):
        raise HTTPException(status_code=500, detail="Invalid hierarchy artifact format: expected object.")
    hierarchy_data = _ensure_hierarchy_enriched(analysis_id=analysis_id, hierarchy_data=hierarchy_data)
    return _build_hierarchy_response(analysis_id=analysis_id, hierarchy_data=hierarchy_data)


@router.post("/{analysis_id}/hierarchy/labels")
def analysis_hierarchy_labels(
    analysis_id: str,
    payload: dict[str, Any] | None = Body(default=None),
) -> dict[str, Any]:
    """Refine labels for selected hierarchy nodes using cached LLM calls."""

    _require_ready(analysis_id, str(hierarchy_file(analysis_id).name))
    _require_ready(analysis_id, str(items_file(analysis_id).name))

    node_ids_raw = payload.get("node_ids") if isinstance(payload, dict) else None
    if not isinstance(node_ids_raw, list):
        raise HTTPException(status_code=422, detail="'node_ids' must be a list of node id strings.")

    node_ids = [str(node_id).strip() for node_id in node_ids_raw if str(node_id).strip()]
    if not node_ids:
        return {"labels": {}, "updated": 0}

    hierarchy_data = read_json_file(hierarchy_file(analysis_id))
    if not isinstance(hierarchy_data, dict):
        raise HTTPException(status_code=500, detail="Invalid hierarchy artifact format: expected object.")
    hierarchy_data = _ensure_hierarchy_enriched(analysis_id=analysis_id, hierarchy_data=hierarchy_data)

    items_payload = read_json_file(items_file(analysis_id))
    items = _extract_items(items_payload)
    labels = refine_hierarchy_labels_for_nodes(
        hierarchy_data=hierarchy_data,
        items=items,
        node_ids=node_ids,
        cache_path=hierarchy_label_cache_file(analysis_id),
        model_name=_load_analysis_llm_model(analysis_id),
        max_nodes=8,
    )
    write_json_file(hierarchy_file(analysis_id), hierarchy_data)
    return {"labels": labels, "updated": len(labels)}


@router.get("/{analysis_id}/insights")
def analysis_insights(analysis_id: str) -> dict[str, Any]:
    """Return insights artifact."""

    _require_ready(analysis_id, str(insights_file(analysis_id).name))
    data = read_json_file(insights_file(analysis_id))
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="Invalid insights artifact format.")

    cluster_payload = _build_cluster_payload(analysis_id=analysis_id, requested_k=None)
    top_themes = sorted(cluster_payload["clusters"], key=lambda cluster: int(cluster["size"]), reverse=True)[:5]

    theme_summary = [
        {
            "label": theme["label"],
            "size": int(theme["size"]),
            "top_terms": theme.get("top_terms", [])[:8],
            "examples": [rep.get("preview", "") for rep in theme.get("representatives", [])[:2]],
        }
        for theme in top_themes
    ]

    key_findings = data.get("key_findings")
    if not isinstance(key_findings, list):
        key_findings = []

    quality_warnings = data.get("quality_warnings")
    if not isinstance(quality_warnings, list):
        quality_warnings = []

    overall_summary = str(data.get("overall_summary") or "").strip()
    overall_summary_source = str(data.get("overall_summary_source") or "").strip() or "heuristic"
    if not overall_summary or overall_summary.endswith("..."):
        overall_summary, overall_summary_source = build_overall_summary(
            total_items=len(cluster_payload["item_cluster_map"]),
            clusters=cluster_payload["clusters"],
            quality_warnings=quality_warnings,
            llm_model=_load_analysis_llm_model(analysis_id),
        )
        if not overall_summary:
            overall_summary = build_overall_summary_fallback(
                total_items=len(cluster_payload["item_cluster_map"]),
                clusters=cluster_payload["clusters"],
                quality_warnings=quality_warnings,
            )
            overall_summary_source = "heuristic"
        data["overall_summary"] = overall_summary
        data["overall_summary_source"] = overall_summary_source
        write_json_file(insights_file(analysis_id), data)

    return {
        "key_findings": key_findings,
        "theme_summary": theme_summary,
        "quality_warnings": quality_warnings,
        "overall_summary": overall_summary,
        "overall_summary_source": overall_summary_source,
    }


@router.get("/{analysis_id}/clusters")
def analysis_clusters(
    analysis_id: str,
    k_clusters: int | None = Query(default=None, ge=2, le=100),
) -> dict[str, Any]:
    """Return flat cluster summaries for requested k."""

    _require_ready(analysis_id, str(clusters_file(analysis_id).name))
    resolved_k = _coerce_optional_int(k_clusters)
    payload = _build_cluster_payload(analysis_id=analysis_id, requested_k=resolved_k)
    return {
        "clusters": payload["clusters"],
        "item_cluster_map": payload["item_cluster_map"],
    }


@router.get("/{analysis_id}/map")
def analysis_map(
    analysis_id: str,
    k_clusters: int | None = Query(default=None, ge=2, le=100),
) -> dict[str, Any]:
    """Return 2D map points with cluster assignment and labels."""

    _require_ready(analysis_id, str(umap_file(analysis_id).name))
    _require_ready(analysis_id, str(clusters_file(analysis_id).name))

    items_data = read_json_file(items_file(analysis_id))
    umap_data = read_json_file(umap_file(analysis_id))
    resolved_k = _coerce_optional_int(k_clusters)
    cluster_payload = _build_cluster_payload(analysis_id=analysis_id, requested_k=resolved_k)

    items = _extract_items(items_data)
    points_base = umap_data.get("points", [])
    cluster_labels = cluster_payload["cluster_labels"]
    clusters = cluster_payload["clusters"]

    if len(points_base) != len(items) or len(cluster_labels) != len(items):
        raise HTTPException(status_code=500, detail="Artifact mismatch between items and map coordinates.")

    label_by_cluster = {
        int(cluster["cluster_id"]): str(cluster.get("label") or f"Cluster {cluster['cluster_id']}")
        for cluster in clusters
    }

    points: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        point = points_base[idx]
        cluster_id = int(cluster_labels[idx])
        points.append(
            {
                "id": item["id"],
                "x": float(point["x"]),
                "y": float(point["y"]),
                "x_raw": float(point["x_raw"]),
                "y_raw": float(point["y_raw"]),
                "cluster_id": cluster_id,
                "cluster_label": label_by_cluster.get(cluster_id, f"Cluster {cluster_id}"),
                "preview": build_preview(str(item.get("text", ""))),
                "metadata": item.get("metadata", {}),
            }
        )

    return {
        "points": points,
        "clusters": clusters,
        "advanced": {
            "umap_scaled": True,
            "scale_clamp": 1.0,
        },
    }


def _parse_options(raw_options: str | None) -> AnalysisOptions:
    """Parse JSON options field from multipart form request."""

    if raw_options is None or not raw_options.strip():
        options_data: dict[str, Any] = {}
    else:
        try:
            loaded = json.loads(raw_options)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="Invalid JSON in 'options' field.") from exc
        if not isinstance(loaded, dict):
            raise HTTPException(status_code=422, detail="'options' field must be a JSON object.")
        options_data = loaded

    try:
        options = AnalysisOptions.model_validate(options_data)
        options.llm_model = validate_requested_ollama_model(options.llm_model)
    except LabelingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid options: {exc}") from exc

    options.granulate_return_items = True
    return options


def _require_ready(analysis_id: str, artifact_name: str) -> None:
    """Enforce artifact availability and 409-not-ready contract."""

    if not analysis_exists(analysis_id):
        raise HTTPException(status_code=404, detail="Analysis id not found.")

    target_path = _artifact_lookup(analysis_id, artifact_name)
    if target_path.exists():
        return

    job = worker.get_job(analysis_id)
    if job is not None and job.status in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail=f"Artifact '{artifact_name}' is not ready yet.")

    if job is not None and job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error or "Analysis failed.")

    raise HTTPException(status_code=409, detail=f"Artifact '{artifact_name}' is not ready yet.")


def _artifact_lookup(analysis_id: str, artifact_name: str):
    """Resolve artifact filename to on-disk path."""

    mapping = {
        "items.json": items_file(analysis_id),
        "embeddings.npy": embeddings_file(analysis_id),
        "umap.json": umap_file(analysis_id),
        "clusters.json": clusters_file(analysis_id),
        "hierarchy.json": hierarchy_file(analysis_id),
        "overview.json": overview_file(analysis_id),
        "insights.json": insights_file(analysis_id),
    }
    if artifact_name not in mapping:
        raise HTTPException(status_code=500, detail=f"Unknown artifact: {artifact_name}")
    return mapping[artifact_name]


def _build_hierarchy_response(analysis_id: str, hierarchy_data: dict[str, Any]) -> HierarchyResponse:
    """Transform stored hierarchy artifact into frontend response contract."""

    raw_nodes = hierarchy_data.get("nodes")
    if not isinstance(raw_nodes, list):
        raise HTTPException(status_code=500, detail="Invalid hierarchy artifact: 'nodes' must be a list.")

    nodes: dict[str, HierarchyNodeResponse] = {}
    for entry in raw_nodes:
        if not isinstance(entry, dict):
            continue
        node_id = str(entry.get("node_id") or "").strip()
        if not node_id:
            continue

        node = HierarchyNodeResponse(
            node_id=node_id,
            parent_id=str(entry["parent_id"]) if entry.get("parent_id") is not None else None,
            children_ids=[str(child_id) for child_id in entry.get("children_ids", []) if child_id is not None],
            size=int(entry.get("size", 0)),
            height=float(entry.get("height", 0.0)),
            label=(
                str(entry["label"])
                if entry.get("label") is not None
                else ("Item" if node_id.startswith("leaf_") else "Theme")
            ),
            cohesion=float(entry["cohesion"]) if entry.get("cohesion") is not None else None,
            similarity=float(entry["similarity"]) if entry.get("similarity") is not None else None,
            descendant_leaf_count=(
                int(entry["descendant_leaf_count"])
                if entry.get("descendant_leaf_count") is not None
                else None
            ),
            dominant_cluster_id=(
                int(entry["dominant_cluster_id"]) if entry.get("dominant_cluster_id") is not None else None
            ),
            dominant_cluster_share=(
                float(entry["dominant_cluster_share"])
                if entry.get("dominant_cluster_share") is not None
                else None
            ),
            summary=str(entry["summary"]) if entry.get("summary") is not None else None,
        )
        nodes[node_id] = node

    if not nodes:
        raise HTTPException(status_code=500, detail="Invalid hierarchy artifact: no nodes found.")

    roots = [node for node in nodes.values() if node.parent_id is None]
    if not roots:
        raise HTTPException(status_code=500, detail="Invalid hierarchy artifact: no root node with parent_id=null.")
    root_id = max(roots, key=lambda node: node.size).node_id

    leaf_id_by_index, cluster_by_index = _load_leaf_mappings(analysis_id)

    leaves: list[HierarchyLeafResponse] = []
    for node in nodes.values():
        is_leaf = node.node_id.startswith("leaf_") or (len(node.children_ids) == 0 and node.size == 1)
        if not is_leaf:
            continue

        leaf_index = _extract_leaf_index(node.node_id)
        leaf_item_id = node.node_id
        cluster_id = -1
        if leaf_index is not None:
            leaf_item_id = leaf_id_by_index.get(leaf_index, node.node_id)
            cluster_id = int(cluster_by_index.get(leaf_index, -1))

        leaves.append(
            HierarchyLeafResponse(
                id=str(leaf_item_id),
                node_id=node.node_id,
                cluster_id=cluster_id,
            )
        )

    leaves.sort(
        key=lambda leaf: (
            _extract_leaf_index(leaf.node_id) is None,
            _extract_leaf_index(leaf.node_id) or 0,
            leaf.node_id,
        )
    )

    return HierarchyResponse(
        root_id=root_id,
        nodes=nodes,
        leaves=leaves,
    )


def _ensure_hierarchy_enriched(analysis_id: str, hierarchy_data: dict[str, Any]) -> dict[str, Any]:
    """Backfill hierarchy labels/summaries for analyses generated before enrichment existed."""

    raw_nodes = hierarchy_data.get("nodes")
    if not isinstance(raw_nodes, list) or not _hierarchy_needs_enrichment(raw_nodes):
        return hierarchy_data

    items_payload = read_json_file(items_file(analysis_id))
    items = _extract_items(items_payload)
    if not items:
        return hierarchy_data

    cluster_payload = _build_cluster_payload(analysis_id=analysis_id, requested_k=None)
    cluster_labels = np.asarray(cluster_payload.get("cluster_labels", []), dtype=int)
    clusters = cluster_payload.get("clusters", [])
    if cluster_labels.size != len(items):
        return hierarchy_data

    enriched = enrich_hierarchy_nodes(
        hierarchy_data=hierarchy_data,
        items=items,
        cluster_labels=cluster_labels,
        clusters=clusters if isinstance(clusters, list) else [],
    )
    write_json_file(hierarchy_file(analysis_id), enriched)
    return enriched


def _hierarchy_needs_enrichment(raw_nodes: list[Any]) -> bool:
    """Detect stale hierarchy payloads that were persisted without node labels."""

    valid_nodes = [entry for entry in raw_nodes if isinstance(entry, dict)]
    if not valid_nodes:
        return False

    labeled = 0
    for entry in valid_nodes:
        label = str(entry.get("label") or "").strip()
        if label:
            labeled += 1
    return labeled < max(1, int(0.8 * len(valid_nodes)))


def _extract_leaf_index(node_id: str) -> int | None:
    """Extract integer index from a leaf node id."""

    if not node_id.startswith("leaf_"):
        return None
    suffix = node_id[len("leaf_") :]
    return int(suffix) if suffix.isdigit() else None


def _load_leaf_mappings(analysis_id: str) -> tuple[dict[int, str], dict[int, int]]:
    """Load optional mappings from leaf index to original item id and cluster id."""

    leaf_id_by_index: dict[int, str] = {}
    cluster_by_index: dict[int, int] = {}

    items_path = items_file(analysis_id)
    if items_path.exists():
        items_payload = read_json_file(items_path)
        if isinstance(items_payload, dict):
            items = items_payload.get("items")
            if isinstance(items, list):
                for idx, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    item_id = item.get("id")
                    if item_id is not None:
                        leaf_id_by_index[idx] = str(item_id)

    clusters_path = clusters_file(analysis_id)
    if clusters_path.exists():
        clusters_payload = read_json_file(clusters_path)
        if isinstance(clusters_payload, dict):
            labels = clusters_payload.get("cluster_labels")
            if isinstance(labels, list):
                numeric_labels: list[int] = []
                for label in labels:
                    try:
                        numeric_labels.append(int(label))
                    except (TypeError, ValueError):
                        continue
                normalized = _normalize_cluster_ids(numeric_labels)
                mapping = normalized["id_map"]
                for idx, label in enumerate(labels):
                    try:
                        cluster_by_index[idx] = int(mapping.get(int(label), -1))
                    except (TypeError, ValueError):
                        continue

    return leaf_id_by_index, cluster_by_index


def _build_cluster_payload(analysis_id: str, requested_k: int | None) -> dict[str, Any]:
    """Build response-friendly cluster payload from stored artifacts."""

    cluster_data = _resolve_clusters(analysis_id=analysis_id, requested_k=requested_k)
    items_data = read_json_file(items_file(analysis_id))
    items = _extract_items(items_data)

    raw_labels: list[int] = []
    for value in cluster_data.get("cluster_labels", []):
        try:
            raw_labels.append(int(value))
        except (TypeError, ValueError):
            raw_labels.append(-1)
    normalized = _normalize_cluster_ids(raw_labels)
    id_map: dict[int, int] = normalized["id_map"]
    normalized_labels: list[int] = normalized["labels"]

    item_by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if item_id is not None:
            item_by_id[str(item_id)] = item

    clusters: list[dict[str, Any]] = []
    raw_clusters = cluster_data.get("clusters", [])
    for cluster in raw_clusters:
        if not isinstance(cluster, dict):
            continue
        raw_cluster_id = int(cluster.get("cluster_id", -1))
        display_cluster_id = int(id_map.get(raw_cluster_id, raw_cluster_id))

        representatives: list[dict[str, Any]] = []
        rep_ids = cluster.get("representative_ids", [])
        if isinstance(rep_ids, list) and rep_ids:
            for rep_id in rep_ids[:5]:
                source = item_by_id.get(str(rep_id))
                if source is None:
                    continue
                representatives.append(
                    {
                        "id": str(source.get("id", rep_id)),
                        "preview": build_preview(str(source.get("text", ""))),
                        "metadata": source.get("metadata", {}),
                    }
                )
        else:
            for index, text in enumerate(cluster.get("representatives", [])[:5]):
                fallback_id = f"cluster_{display_cluster_id}_rep_{index}"
                if isinstance(text, dict):
                    representatives.append(
                        {
                            "id": str(text.get("id", fallback_id)),
                            "preview": str(text.get("preview", "")),
                            "metadata": text.get("metadata", {}),
                        }
                    )
                    continue
                representatives.append(
                    {
                        "id": fallback_id,
                        "preview": build_preview(str(text)),
                        "metadata": {},
                    }
                )

        clusters.append(
            {
                "cluster_id": display_cluster_id,
                "label": str(cluster.get("label") or f"Cluster {display_cluster_id}"),
                "size": int(cluster.get("size", 0)),
                "top_terms": [str(term) for term in cluster.get("top_terms", [])],
                "representatives": representatives,
            }
        )

    clusters.sort(key=lambda cluster: int(cluster["cluster_id"]))

    item_cluster_map: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        item_id = str(item.get("id", f"item_{idx}"))
        cluster_id = normalized_labels[idx] if idx < len(normalized_labels) else -1
        item_cluster_map.append({"id": item_id, "cluster_id": cluster_id})

    return {
        "clusters": clusters,
        "item_cluster_map": item_cluster_map,
        "cluster_labels": normalized_labels,
    }


def _normalize_cluster_ids(raw_labels: list[int]) -> dict[str, Any]:
    """Normalize cluster ids to consecutive 0-based ids for frontend compatibility."""

    unique = sorted(set(raw_labels))
    id_map = {raw_id: index for index, raw_id in enumerate(unique)}
    normalized = [id_map.get(raw_id, -1) for raw_id in raw_labels]
    return {"id_map": id_map, "labels": normalized}


def _load_timing_payload(analysis_id: str) -> dict[str, float]:
    """Load timing payload or return a default shape."""

    default = {
        "embeddings_sec": 0.0,
        "hierarchy_sec": 0.0,
        "clusters_sec": 0.0,
        "umap_sec": 0.0,
        "labeling_sec": 0.0,
        "granulate_sec": 0.0,
    }

    candidate = hierarchy_file(analysis_id).parent / "timing.json"
    if not candidate.exists():
        return default

    data = read_json_file(candidate)
    if not isinstance(data, dict):
        return default

    for key in default:
        value = data.get(key)
        try:
            default[key] = float(value)
        except (TypeError, ValueError):
            continue
    return default


def _load_analysis_llm_model(
    analysis_id: str,
    *,
    cluster_data: dict[str, Any] | None = None,
    overview_data: dict[str, Any] | None = None,
) -> str:
    """Load the persisted LLM model for one analysis, falling back to backend default."""

    if isinstance(cluster_data, dict):
        value = cluster_data.get("llm_model")
        if isinstance(value, str) and value.strip():
            return normalize_requested_ollama_model(value)

    if overview_data is None:
        overview_path = overview_file(analysis_id)
        if overview_path.exists():
            loaded = read_json_file(overview_path)
            if isinstance(loaded, dict):
                overview_data = loaded

    if isinstance(overview_data, dict):
        value = overview_data.get("llm_model")
        if isinstance(value, str) and value.strip():
            return normalize_requested_ollama_model(value)

    return default_ollama_model()


def _extract_top_aspects(analysis_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract top aspects when available; return an empty list otherwise."""

    top_aspects = analysis_data.get("top_aspects")
    if isinstance(top_aspects, list):
        return [entry for entry in top_aspects if isinstance(entry, dict)]
    return []


def _extract_items(items_payload: Any) -> list[dict[str, Any]]:
    """Extract a list of item objects from multiple artifact layouts."""

    if isinstance(items_payload, dict):
        items = items_payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return []
    if isinstance(items_payload, list):
        return [item for item in items_payload if isinstance(item, dict)]
    return []


def _coerce_optional_int(value: Any) -> int | None:
    """Convert endpoint parameter values into optional integer."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _item_aspect_summary(item: dict[str, Any]) -> dict[str, Any]:
    """Expose inferred aspect/polarity from item metadata when available."""

    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    analysis_meta = metadata.get("_analysis")
    if not isinstance(analysis_meta, dict):
        return {}

    aspect = str(analysis_meta.get("aspect") or "").strip()
    polarity = str(analysis_meta.get("polarity") or "").strip()
    if not aspect and not polarity:
        return {}
    payload: dict[str, Any] = {}
    if aspect:
        payload["aspect"] = aspect
    if polarity:
        payload["polarity"] = polarity
    return payload


def _normalize_run_stage(stage: Any) -> str:
    """Normalize backend stages to frontend-safe stage ids."""

    value = str(stage or "").strip().lower()
    aliases = {
        "ingestion": "queued",
        "granulation": "granulate",
        "summaries": "overview",
        "insights": "overview",
        "ai_summary": "overview",
        "failed": "failed",
    }
    normalized = aliases.get(value, value)

    allowed = {
        "queued",
        "embeddings",
        "hierarchy",
        "clusters",
        "umap",
        "labeling",
        "granulate",
        "overview",
        "completed",
        "failed",
    }
    if normalized not in allowed:
        return "queued"
    return normalized


def _resolve_clusters(analysis_id: str, requested_k: int | None) -> dict[str, Any]:
    """Load or recompute cluster summaries for requested k."""

    cluster_artifact = read_json_file(clusters_file(analysis_id))
    items_data = read_json_file(items_file(analysis_id))
    selected_llm_model = _load_analysis_llm_model(analysis_id, cluster_data=cluster_artifact)

    items = items_data.get("items", [])
    total_items = len(items)

    default_k = int(cluster_artifact.get("k_clusters", 0))
    if requested_k is None or requested_k == default_k:
        labels = [int(label) for label in cluster_artifact.get("cluster_labels", [])]
        return {
            "k_clusters": default_k,
            "total_items": total_items,
            "cluster_labels": labels,
            "clusters": cluster_artifact.get("clusters", []),
        }

    vectors = read_embeddings(embeddings_file(analysis_id))
    hierarchy_data = read_json_file(hierarchy_file(analysis_id))
    linkage_matrix = np.asarray(hierarchy_data.get("linkage_matrix", []), dtype=float)

    resolved_k = normalize_k_clusters(requested_k, total_items)
    labels_np = cut_clusters(linkage_matrix, n_items=total_items, k_clusters=resolved_k)
    cluster_summaries = build_cluster_summaries(items, vectors, labels_np)
    labeled_clusters = apply_labels(
        cluster_summaries=cluster_summaries,
        k_clusters=resolved_k,
        cache_path=label_cache_file(analysis_id),
        model_name=selected_llm_model,
    )

    return {
        "k_clusters": resolved_k,
        "total_items": total_items,
        "llm_model": selected_llm_model,
        "cluster_labels": labels_np.astype(int).tolist(),
        "clusters": labeled_clusters,
    }
