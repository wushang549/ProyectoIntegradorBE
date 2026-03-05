"""Disk persistence helpers for analysis artifacts and index."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path("data")
INDEX_FILE = DATA_DIR / "index.json"

_ITEMS_FILE = "items.json"
_EMBEDDINGS_FILE = "embeddings.npy"
_UMAP_FILE = "umap.json"
_CLUSTERS_FILE = "clusters.json"
_HIERARCHY_FILE = "hierarchy.json"
_OVERVIEW_FILE = "overview.json"
_INSIGHTS_FILE = "insights.json"
_LABEL_CACHE_FILE = "labels_cache.json"
_HIERARCHY_LABEL_CACHE_FILE = "hierarchy_labels_cache.json"

_INDEX_LOCK = threading.Lock()


def ensure_data_dir() -> Path:
    """Ensure base data directory exists."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        write_json_file(INDEX_FILE, [])
    return DATA_DIR


def ensure_analysis_dir(analysis_id: str) -> Path:
    """Ensure analysis artifact directory exists."""

    ensure_data_dir()
    target = DATA_DIR / analysis_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def analysis_dir(analysis_id: str) -> Path:
    """Get directory path for one analysis."""

    return DATA_DIR / analysis_id


def analysis_exists(analysis_id: str) -> bool:
    """Check if analysis directory exists on disk."""

    return analysis_dir(analysis_id).exists()


def artifact_path(analysis_id: str, filename: str) -> Path:
    """Build a path to one analysis artifact file."""

    return analysis_dir(analysis_id) / filename


def artifacts_ready(analysis_id: str) -> bool:
    """Check whether all required artifacts are present."""

    target = analysis_dir(analysis_id)
    required = [_ITEMS_FILE, _EMBEDDINGS_FILE, _UMAP_FILE, _CLUSTERS_FILE, _HIERARCHY_FILE, _OVERVIEW_FILE, _INSIGHTS_FILE]
    return all((target / name).exists() for name in required)


def items_file(analysis_id: str) -> Path:
    """Path for items artifact."""

    return artifact_path(analysis_id, _ITEMS_FILE)


def embeddings_file(analysis_id: str) -> Path:
    """Path for embeddings artifact."""

    return artifact_path(analysis_id, _EMBEDDINGS_FILE)


def umap_file(analysis_id: str) -> Path:
    """Path for UMAP artifact."""

    return artifact_path(analysis_id, _UMAP_FILE)


def clusters_file(analysis_id: str) -> Path:
    """Path for clusters artifact."""

    return artifact_path(analysis_id, _CLUSTERS_FILE)


def hierarchy_file(analysis_id: str) -> Path:
    """Path for hierarchy artifact."""

    return artifact_path(analysis_id, _HIERARCHY_FILE)


def overview_file(analysis_id: str) -> Path:
    """Path for overview artifact."""

    return artifact_path(analysis_id, _OVERVIEW_FILE)


def insights_file(analysis_id: str) -> Path:
    """Path for insights artifact."""

    return artifact_path(analysis_id, _INSIGHTS_FILE)


def label_cache_file(analysis_id: str) -> Path:
    """Path for cluster label cache file."""

    return artifact_path(analysis_id, _LABEL_CACHE_FILE)


def hierarchy_label_cache_file(analysis_id: str) -> Path:
    """Path for hierarchy-node label cache file."""

    return artifact_path(analysis_id, _HIERARCHY_LABEL_CACHE_FILE)


def write_json_file(path: Path, data: Any) -> None:
    """Write JSON data with numpy-safe conversion."""

    safe = json_safe(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json_file(path: Path) -> Any:
    """Read JSON data from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_embeddings(path: Path, vectors: np.ndarray) -> None:
    """Persist embedding matrix to .npy file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)


def read_embeddings(path: Path) -> np.ndarray:
    """Load embedding matrix from .npy file."""

    return np.load(path)


def upsert_index_entry(entry: dict[str, Any], keep: int = 500) -> None:
    """Insert/update one index entry and keep recent order."""

    ensure_data_dir()
    with _INDEX_LOCK:
        current = []
        if INDEX_FILE.exists():
            loaded = read_json_file(INDEX_FILE)
            if isinstance(loaded, list):
                current = loaded

        filtered = [item for item in current if item.get("analysis_id") != entry.get("analysis_id")]
        filtered.insert(0, json_safe(entry))
        write_json_file(INDEX_FILE, filtered[:keep])


def list_recent(limit: int = 10) -> list[dict[str, Any]]:
    """Return recent analyses from index file."""

    ensure_data_dir()
    with _INDEX_LOCK:
        if not INDEX_FILE.exists():
            return []
        loaded = read_json_file(INDEX_FILE)
        if not isinstance(loaded, list):
            return []
        return loaded[: max(1, limit)]


def now_utc_iso() -> str:
    """Current UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Recursively convert numpy values into JSON-serializable values."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
