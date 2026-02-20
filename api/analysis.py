from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from services.models import (
    AnalysisOptions,
    AnalysisStatusResponse,
    ClustersResponse,
    CreateAnalysisResponse,
    GranulateResponse,
    HierarchyResponse,
    InsightsResponse,
    MapResponse,
    OverviewResponse,
    RecentAnalysisEntry,
)
from services.pipeline import load_artifact_or_404, load_granulate_payload, run_analysis_pipeline
from services.storage import create_analysis_dir, read_json, read_status, write_status, write_text
from services.validation import parse_options_json, save_uploaded_payload, validate_csv_upload, validate_text_input

router = APIRouter(prefix="/v1/analysis", tags=["analysis"])


def _analysis_dir_or_404(analysis_id: str) -> Path:
    path = Path("uploads") / analysis_id
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail="analysis_id not found")
    return path


def _ensure_completed_or_raise(analysis_dir: Path) -> None:
    status = read_status(analysis_dir)
    state = status.get("status")
    if state == "completed":
        return
    if state == "failed":
        raise HTTPException(status_code=400, detail=status.get("error") or "analysis failed")
    raise HTTPException(status_code=409, detail="analysis is still processing")


@router.post("", response_model=CreateAnalysisResponse)
async def create_analysis(
    background_tasks: BackgroundTasks,
    input_type: Literal["text", "csv"] = Form(...),
    text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    options: str | None = Form(default=None),
):
    opts_raw = parse_options_json(options)
    opts = AnalysisOptions.model_validate(opts_raw).model_dump()

    analysis_id, analysis_dir = create_analysis_dir()
    created_at = datetime.now(timezone.utc)

    if input_type == "text":
        cleaned_text = validate_text_input(text)
        texts = [cleaned_text]
        input_items = [{"text": cleaned_text, "metadata": {}}]
        write_text(analysis_dir / "input.txt", cleaned_text)
    else:
        texts, payload, csv_rows = validate_csv_upload(file)
        input_items = csv_rows
        save_uploaded_payload(analysis_dir / "input.csv", payload)

    write_status(analysis_dir, status="queued", stage="queued", pct=0)
    background_tasks.add_task(
        run_analysis_pipeline,
        analysis_id=analysis_id,
        analysis_dir=analysis_dir,
        mode=input_type,
        texts=texts,
        options=opts,
        input_items=input_items,
    )

    base = f"/v1/analysis/{analysis_id}"
    return CreateAnalysisResponse(
        analysis_id=analysis_id,
        status="queued",
        created_at=created_at,
        tabs={
            "overview": f"{base}/overview",
            "map": f"{base}/map",
            "clusters": f"{base}/clusters",
            "granulate": f"{base}/granulate",
            "hierarchy": f"{base}/hierarchy",
            "insights": f"{base}/insights",
            "status": f"{base}/status",
        },
    )


@router.get("/recent", response_model=list[RecentAnalysisEntry])
def list_recent_analyses(limit: int = 20):
    capped_limit = max(1, min(100, int(limit)))
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        return []

    out: list[RecentAnalysisEntry] = []
    dirs = [d for d in uploads_dir.iterdir() if d.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for analysis_dir in dirs[:capped_limit]:
        status = read_status(analysis_dir)
        item_count = 0
        overview_file = analysis_dir / "overview.json"
        items_file = analysis_dir / "items.json"
        if overview_file.exists():
            try:
                overview = read_json(overview_file)
                item_count = int((overview.get("counts") or {}).get("items", 0))
            except Exception:
                item_count = 0
        elif items_file.exists():
            try:
                payload = read_json(items_file)
                if isinstance(payload, list):
                    item_count = len(payload)
            except Exception:
                item_count = 0

        out.append(
            RecentAnalysisEntry(
                analysis_id=analysis_dir.name,
                status=str(status.get("status", "queued")),
                created_at=str(status.get("updated_at", "")) or None,
                item_count=item_count,
            )
        )
    return out


@router.get("/{analysis_id}/status", response_model=AnalysisStatusResponse)
def get_analysis_status(analysis_id: str):
    analysis_dir = _analysis_dir_or_404(analysis_id)
    status = read_status(analysis_dir)
    return AnalysisStatusResponse.model_validate(status)


@router.get("/{analysis_id}/overview", response_model=OverviewResponse)
def get_overview(analysis_id: str):
    analysis_dir = _analysis_dir_or_404(analysis_id)
    _ensure_completed_or_raise(analysis_dir)
    try:
        return OverviewResponse.model_validate(load_artifact_or_404(analysis_dir, "overview.json"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{analysis_id}/map", response_model=MapResponse)
def get_map(analysis_id: str):
    analysis_dir = _analysis_dir_or_404(analysis_id)
    _ensure_completed_or_raise(analysis_dir)
    try:
        return MapResponse.model_validate(load_artifact_or_404(analysis_dir, "umap_2d.json"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{analysis_id}/clusters", response_model=ClustersResponse)
def get_clusters(analysis_id: str):
    analysis_dir = _analysis_dir_or_404(analysis_id)
    _ensure_completed_or_raise(analysis_dir)
    try:
        payload = load_artifact_or_404(analysis_dir, "clusters.json")
        return ClustersResponse.model_validate({"clusters": payload.get("clusters", [])})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{analysis_id}/granulate", response_model=GranulateResponse)
def get_granulate(analysis_id: str, include_items: bool = False):
    analysis_dir = _analysis_dir_or_404(analysis_id)
    _ensure_completed_or_raise(analysis_dir)
    try:
        payload = load_granulate_payload(analysis_dir, include_items=include_items)
        return GranulateResponse.model_validate(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{analysis_id}/hierarchy", response_model=HierarchyResponse)
def get_hierarchy(analysis_id: str):
    analysis_dir = _analysis_dir_or_404(analysis_id)
    _ensure_completed_or_raise(analysis_dir)
    try:
        return HierarchyResponse.model_validate(load_artifact_or_404(analysis_dir, "hierarchy.json"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{analysis_id}/insights", response_model=InsightsResponse)
def get_insights(analysis_id: str):
    analysis_dir = _analysis_dir_or_404(analysis_id)
    _ensure_completed_or_raise(analysis_dir)
    try:
        return InsightsResponse.model_validate(load_artifact_or_404(analysis_dir, "insights.json"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
