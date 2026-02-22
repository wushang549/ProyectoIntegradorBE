from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UPLOADS_DIR = Path("uploads")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_analysis_dir(analysis_id: str) -> Path:
    target = UPLOADS_DIR / analysis_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def create_analysis_dir() -> tuple[str, Path]:
    analysis_id = uuid.uuid4().hex
    return analysis_id, ensure_analysis_dir(analysis_id)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(5):
            try:
                tmp_path.replace(path)
                break
            except PermissionError:
                if attempt >= 4:
                    # Last-resort fallback for cloud-sync/AV lock edge cases.
                    # Readers may briefly observe partial content, so read_status
                    # is defensive against transient decode failures.
                    with path.open("w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    break
                time.sleep(0.02 * (attempt + 1))
            except OSError:
                if attempt >= 4:
                    raise
                time.sleep(0.02 * (attempt + 1))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_status(analysis_dir: Path) -> dict[str, Any]:
    status_path = analysis_dir / "status.json"
    default_status = {
        "analysis_id": analysis_dir.name,
        "status": "queued",
        "progress": {"stage": "queued", "pct": 0},
        "error": None,
        "debug_error": None,
    }
    processing_status = {
        "analysis_id": analysis_dir.name,
        "status": "processing",
        "progress": {"stage": "processing", "pct": 0},
        "error": None,
        "debug_error": None,
    }
    if not status_path.exists():
        return default_status

    candidates = [status_path, status_path.with_suffix(f"{status_path.suffix}.tmp")]
    for attempt in range(4):
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                data = read_json(candidate)
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                continue
        time.sleep(0.02 * (attempt + 1))
    return processing_status


def write_status(
    analysis_dir: Path,
    status: str,
    stage: str,
    pct: int,
    error: str | None = None,
    debug_error: str | None = None,
) -> None:
    payload = {
        "analysis_id": analysis_dir.name,
        "status": status,
        "progress": {"stage": stage, "pct": int(max(0, min(100, pct)))},
        "error": error,
        "debug_error": debug_error,
        "updated_at": utc_now_iso(),
    }
    write_json(analysis_dir / "status.json", payload)
