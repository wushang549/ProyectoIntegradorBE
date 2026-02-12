# app/api/projects.py  (additions + full file)
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from db import get_conn
from models import CreateProjectResponse, ProjectResponse
from services.granulate import granulate_text
from services.ml import predict_one, train_and_save
from services.storage import new_project_id, save_upload
from services.validation import validate_extension, validate_text_column

router = APIRouter(prefix="/v1/projects", tags=["projects"])


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_project_exists(project_id: str) -> None:
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")


def set_status(project_id: str, status: str, error_message: str | None = None) -> None:
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE projects
            SET status = ?, updated_at = ?, error_message = ?
            WHERE id = ?
            """,
            (status, ts, error_message, project_id),
        )
        conn.commit()


def get_project_row(project_id: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return row


def get_input_path(project_id: str) -> Path:
    with get_conn() as conn:
        row = conn.execute("SELECT input_path FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")
    return Path(row["input_path"])


def get_project_dir(project_id: str) -> Path:
    return get_input_path(project_id).parent


def read_texts_from_csv(input_csv: Path) -> list[str]:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV is empty or missing header")
        if "text" not in reader.fieldnames:
            raise ValueError("Missing required column: text")

        texts: list[str] = []
        for row in reader:
            t = (row.get("text") or "").strip()
            if t:
                texts.append(t)

    if not texts:
        raise ValueError("No non-empty rows found in column: text")

    return texts


def results_path(project_id: str) -> Path:
    return get_project_dir(project_id) / "results.json"


@router.post("", response_model=CreateProjectResponse)
async def create_project(
    analysis_name: str = Form(...),
    file: UploadFile = File(...),
):
    try:
        validate_extension(file.filename or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    project_id = new_project_id()
    saved_path: Path = save_upload(project_id, file.filename or "input.csv", raw)

    try:
        validate_text_column(saved_path)
    except ValueError as e:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO projects (id, name, status, input_path, created_at, updated_at, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (project_id, analysis_name, "queued", str(saved_path), ts, ts, None),
        )
        conn.commit()

    return CreateProjectResponse(project_id=project_id, status="queued")


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str):
    row = get_project_row(project_id)
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProjectResponse(
        project_id=row["id"],
        name=row["name"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        input_path=row["input_path"],
        error_message=row["error_message"],
    )


@router.post("/{project_id}/run")
def run_project(project_id: str, background: BackgroundTasks):
    ensure_project_exists(project_id)

    row = get_project_row(project_id)
    if row and row["status"] == "processing":
        return {"project_id": project_id, "status": "processing"}

    set_status(project_id, "processing", None)
    background.add_task(ml_pipeline, project_id)
    return {"project_id": project_id, "status": "processing"}


def ml_pipeline(project_id: str) -> None:
    try:
        input_csv = get_input_path(project_id)
        pdir = input_csv.parent

        texts = read_texts_from_csv(input_csv)

        out = train_and_save(texts, pdir, n_clusters=8)

        results = {
            "project_id": project_id,
            "stats": {
                "total_texts": len(texts),
                "clusters": len(out.clusters),
                "input_file": input_csv.name,
                "model": "tfidf+kmeans+svd2d",
            },
            "points": out.points,
            "clusters": out.clusters,
        }

        results_path(project_id).write_text(
            json.dumps(results, ensure_ascii=False),
            encoding="utf-8",
        )

        set_status(project_id, "completed", None)
    except Exception as e:
        set_status(project_id, "failed", str(e))


@router.get("/{project_id}/results")
def get_results(project_id: str):
    ensure_project_exists(project_id)

    path = results_path(project_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Results not found. Run the project first.")

    return json.loads(path.read_text(encoding="utf-8"))


class ClassifyRequest(BaseModel):
    text: str


@router.post("/{project_id}/classify")
def classify_text(project_id: str, body: ClassifyRequest):
    ensure_project_exists(project_id)

    pdir = get_project_dir(project_id)

    if not (pdir / "vectorizer.joblib").exists():
        raise HTTPException(status_code=404, detail="Model not found. Run the project first.")

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    return predict_one(text, pdir)


class GranulateRequest(BaseModel):
    text: str
    taxonomy: dict[str, list[str]] | None = None
    top_k_evidence: int = Field(default=6, ge=1, le=12)
    min_similarity: float = Field(default=0.08, ge=0.0, le=1.0)


@router.post("/{project_id}/granulate")
def granulate(project_id: str, body: GranulateRequest):
    ensure_project_exists(project_id)

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        return granulate_text(
            text=text,
            aspects=body.taxonomy,
            top_k_evidence=body.top_k_evidence,
            min_similarity=body.min_similarity,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
