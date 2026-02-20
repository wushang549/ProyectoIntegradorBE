from __future__ import annotations

import json
import os
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
        try:
            tmp_path.replace(path)
        except PermissionError:
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
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
    if not status_path.exists():
        return {
            "analysis_id": analysis_dir.name,
            "status": "queued",
            "progress": {"stage": "queued", "pct": 0},
            "error": None,
        }
    return read_json(status_path)


def write_status(
    analysis_dir: Path,
    status: str,
    stage: str,
    pct: int,
    error: str | None = None,
) -> None:
    payload = {
        "analysis_id": analysis_dir.name,
        "status": status,
        "progress": {"stage": stage, "pct": int(max(0, min(100, pct)))},
        "error": error,
        "updated_at": utc_now_iso(),
    }
    write_json(analysis_dir / "status.json", payload)
