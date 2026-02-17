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
    MapResponse,
    OverviewResponse,
)
from services.pipeline import load_artifact_or_404, load_granulate_payload, run_analysis_pipeline
from services.storage import create_analysis_dir, read_status, write_status, write_text
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
        write_text(analysis_dir / "input.txt", cleaned_text)
    else:
        texts, payload = validate_csv_upload(file)
        save_uploaded_payload(analysis_dir / "input.csv", payload)

    write_status(analysis_dir, status="queued", stage="queued", pct=0)
    background_tasks.add_task(
        run_analysis_pipeline,
        analysis_id=analysis_id,
        analysis_dir=analysis_dir,
        mode=input_type,
        texts=texts,
        options=opts,
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
            "status": f"{base}/status",
        },
    )


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
