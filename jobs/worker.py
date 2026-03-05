"""In-process async job manager using ThreadPoolExecutor."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from services.storage import now_utc_iso, upsert_index_entry


@dataclass(slots=True)
class JobState:
    """Mutable status for one background analysis job."""

    analysis_id: str
    status: str
    stage: str
    pct: int
    stage_label: str
    message: str
    created_at: str
    updated_at: str
    started_monotonic: float
    error: str | None = None
    total_records: int = 0
    total_items: int = 0


_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="analysis-worker")
_LOCK = threading.Lock()
_JOBS: dict[str, JobState] = {}
_FUTURES: dict[str, Future] = {}


def create_job(analysis_id: str) -> JobState:
    """Register a queued job."""

    now = now_utc_iso()
    job = JobState(
        analysis_id=analysis_id,
        status="queued",
        stage="queued",
        pct=0,
        stage_label="Queued",
        message="Analysis queued.",
        created_at=now,
        updated_at=now,
        started_monotonic=time.monotonic(),
    )
    with _LOCK:
        _JOBS[analysis_id] = job
    upsert_index_entry(
        {
            "analysis_id": analysis_id,
            "created_at": now,
            "updated_at": now,
            "status": "queued",
            "stage": "queued",
            "pct": 0,
            "total_records": 0,
            "total_items": 0,
        }
    )
    return job


def submit_job(analysis_id: str, fn: Callable[[], None]) -> None:
    """Submit a background task for an existing job id."""

    future = _EXECUTOR.submit(fn)
    with _LOCK:
        _FUTURES[analysis_id] = future


def set_processing(analysis_id: str) -> None:
    """Mark job as processing."""

    _update(
        analysis_id,
        status="processing",
        stage="ingestion",
        pct=1,
        stage_label="Ingestion",
        message="Starting ingestion stage.",
    )


def update_progress(
    analysis_id: str,
    *,
    stage: str,
    pct: int,
    stage_label: str,
    message: str,
    total_records: int | None = None,
    total_items: int | None = None,
) -> None:
    """Update progress of a running job."""

    _update(
        analysis_id,
        status="processing",
        stage=stage,
        pct=pct,
        stage_label=stage_label,
        message=message,
        total_records=total_records,
        total_items=total_items,
    )


def set_completed(analysis_id: str) -> None:
    """Mark job as completed."""

    _update(
        analysis_id,
        status="completed",
        stage="completed",
        pct=100,
        stage_label="Completed",
        message="Analysis completed.",
    )


def set_failed(analysis_id: str, error: str) -> None:
    """Mark job as failed."""

    _update(
        analysis_id,
        status="failed",
        stage="failed",
        pct=100,
        stage_label="Failed",
        message="Analysis failed.",
        error=error,
    )


def get_job(analysis_id: str) -> JobState | None:
    """Get job state by analysis id."""

    with _LOCK:
        return _JOBS.get(analysis_id)


def get_job_payload(analysis_id: str) -> dict[str, object] | None:
    """Return job as API-ready dictionary."""

    job = get_job(analysis_id)
    if job is None:
        return None

    elapsed = max(0.0, time.monotonic() - job.started_monotonic)
    payload = {
        "analysis_id": job.analysis_id,
        "status": job.status,
        "stage": job.stage,
        "pct": int(job.pct),
        "stage_label": job.stage_label,
        "message": job.message,
        "elapsed_sec": round(float(elapsed), 3),
        "created_at": datetime.fromisoformat(job.created_at),
        "updated_at": datetime.fromisoformat(job.updated_at),
        "error": job.error,
    }
    return payload


def _update(
    analysis_id: str,
    *,
    status: str,
    stage: str,
    pct: int,
    stage_label: str,
    message: str,
    error: str | None = None,
    total_records: int | None = None,
    total_items: int | None = None,
) -> None:
    """Apply thread-safe job state updates and persist to index."""

    with _LOCK:
        job = _JOBS.get(analysis_id)
        if job is None:
            return

        job.status = status
        job.stage = stage
        job.pct = max(0, min(100, int(pct)))
        job.stage_label = stage_label
        job.message = message
        job.updated_at = now_utc_iso()
        if error is not None:
            job.error = error
        if total_records is not None:
            job.total_records = int(total_records)
        if total_items is not None:
            job.total_items = int(total_items)

        upsert_index_entry(
            {
                "analysis_id": analysis_id,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "status": job.status,
                "stage": job.stage,
                "pct": job.pct,
                "total_records": job.total_records,
                "total_items": job.total_items,
            }
        )
