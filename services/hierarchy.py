"""Hierarchy stage based on scipy linkage clustering."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage

from services.storage import read_json_file, write_json_file

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{1,}")
_STOP_TOKENS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "with",
    "for",
    "from",
    "into",
    "very",
    "really",
    "was",
    "were",
    "is",
    "are",
    "this",
    "that",
    "these",
    "those",
    "food",
    "service",
    "restaurant",
    "place",
}


def build_hierarchy(vectors: np.ndarray) -> dict[str, Any]:
    """Build hierarchical dendrogram nodes from embedding vectors."""

    n_items = int(vectors.shape[0])
    if n_items == 0:
        return {"nodes": [], "linkage_matrix": []}

    if n_items == 1:
        node = {
            "node_id": "leaf_0",
            "parent_id": None,
            "children_ids": [],
            "size": 1,
            "height": 0.0,
            "cohesion": 1.0,
            "similarity": 1.0,
            "descendant_leaf_count": 1,
        }
        return {"nodes": [node], "linkage_matrix": []}

    linkage_matrix = _build_linkage(vectors)
    nodes = _build_nodes(linkage_matrix, n_items)

    return {
        "nodes": nodes,
        "linkage_matrix": linkage_matrix.tolist(),
    }


def enrich_hierarchy_nodes(
    hierarchy_data: dict[str, Any],
    items: list[dict[str, Any]],
    cluster_labels: np.ndarray,
    clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach heuristic labels/summaries to hierarchy nodes."""

    nodes = hierarchy_data.get("nodes")
    if not isinstance(nodes, list):
        return hierarchy_data

    cluster_label_by_id: dict[int, str] = {}
    for cluster in clusters:
        try:
            cid = int(cluster.get("cluster_id", -1))
        except Exception:
            continue
        cluster_label_by_id[cid] = str(cluster.get("label") or f"Cluster {cid}")

    cluster_labels_arr = np.asarray(cluster_labels, dtype=int)
    node_by_id = {
        str(entry.get("node_id")): entry
        for entry in nodes
        if isinstance(entry, dict) and entry.get("node_id")
    }
    descendants = _descendants_by_node(node_by_id)

    for node_id, entry in node_by_id.items():
        leaf_indices = descendants.get(node_id, [])
        if not leaf_indices:
            continue

        size = len(leaf_indices)
        entry["descendant_leaf_count"] = size

        if node_id.startswith("leaf_"):
            leaf_idx = leaf_indices[0]
            label = _leaf_label(items, leaf_idx)
            entry["label"] = label
            entry["summary"] = _leaf_summary(items, leaf_idx)
            if 0 <= leaf_idx < cluster_labels_arr.size:
                entry["dominant_cluster_id"] = int(cluster_labels_arr[leaf_idx])
                entry["dominant_cluster_share"] = 1.0
            continue

        counts: Counter[int] = Counter()
        for idx in leaf_indices:
            if 0 <= idx < cluster_labels_arr.size:
                counts[int(cluster_labels_arr[idx])] += 1

        dominant_id = None
        dominant_share = 0.0
        if counts:
            dominant_id, dominant_count = counts.most_common(1)[0]
            dominant_share = float(dominant_count / max(1, size))
            entry["dominant_cluster_id"] = int(dominant_id)
            entry["dominant_cluster_share"] = round(dominant_share, 4)

        texts = [
            str(items[idx].get("text", ""))
            for idx in leaf_indices
            if 0 <= idx < len(items) and isinstance(items[idx], dict)
        ]
        top_terms = _top_terms(texts, top_n=6)
        base_label = _compose_node_label(
            counts=counts,
            cluster_label_by_id=cluster_label_by_id,
            dominant_cluster_id=dominant_id,
            dominant_share=dominant_share,
            top_terms=top_terms,
        )
        entry["label"] = base_label or "Theme"
        if top_terms:
            entry["summary"] = ", ".join(top_terms[:4])
        else:
            entry["summary"] = str(entry.get("label") or "Theme")

    hierarchy_data["nodes"] = list(node_by_id.values())
    return hierarchy_data


def refine_hierarchy_labels_for_nodes(
    hierarchy_data: dict[str, Any],
    items: list[dict[str, Any]],
    node_ids: list[str],
    cache_path: Path,
    model_name: str | None = None,
    max_nodes: int = 8,
) -> dict[str, str]:
    """Refine selected node labels using LLM with per-node caching."""

    nodes = hierarchy_data.get("nodes")
    if not isinstance(nodes, list):
        return {}

    node_by_id = {
        str(entry.get("node_id")): entry
        for entry in nodes
        if isinstance(entry, dict) and entry.get("node_id")
    }
    descendants = _descendants_by_node(node_by_id)
    cache = _load_label_cache(cache_path)
    selected = []
    for node_id in node_ids:
        normalized = str(node_id).strip()
        if not normalized or normalized in selected:
            continue
        if normalized not in node_by_id:
            continue
        if normalized.startswith("leaf_"):
            continue
        selected.append(normalized)
        if len(selected) >= max_nodes:
            break

    if not selected:
        return {}

    # Local import avoids coupling for call sites that do not need LLM.
    from services.labeling import generate_label_from_context
    from services.openai_text import resolve_openai_text_model

    resolved_model_name = resolve_openai_text_model(model_name)

    output: dict[str, str] = {}
    for node_id in selected:
        cache_key = f"model::{resolved_model_name.lower()}::node::{node_id}"
        node_entry = node_by_id[node_id]
        node_size = int(node_entry.get("size", 0))
        dominant_share = float(node_entry.get("dominant_cluster_share") or 0.0)
        if node_size >= 20 and dominant_share < 0.6:
            stable = str(node_entry.get("label") or "Mixed Themes")
            cache[cache_key] = stable
            output[node_id] = stable
            continue

        cached = cache.get(cache_key)
        if isinstance(cached, str) and cached.strip():
            node_by_id[node_id]["label"] = cached
            output[node_id] = cached
            continue

        leaf_indices = descendants.get(node_id, [])
        texts = [
            str(items[idx].get("text", ""))
            for idx in leaf_indices
            if 0 <= idx < len(items) and isinstance(items[idx], dict)
        ]
        if not texts:
            continue

        top_terms = _top_terms(texts, top_n=10)
        representatives = _representative_texts(texts, limit=5)
        fallback = str(node_by_id[node_id].get("label") or "Theme")
        label = generate_label_from_context(
            top_terms=top_terms,
            representatives=representatives,
            fallback=fallback,
            model_name=resolved_model_name,
        )
        node_by_id[node_id]["label"] = label
        if top_terms and not node_by_id[node_id].get("summary"):
            node_by_id[node_id]["summary"] = ", ".join(top_terms[:4])
        cache[cache_key] = label
        output[node_id] = label

    hierarchy_data["nodes"] = list(node_by_id.values())
    _save_label_cache(cache_path, cache)
    return output


def _build_linkage(vectors: np.ndarray) -> np.ndarray:
    """Compute linkage matrix with cosine distance."""

    try:
        return linkage(vectors, method="average", metric="cosine")
    except Exception:
        return linkage(vectors, method="average", metric="euclidean")


def _build_nodes(linkage_matrix: np.ndarray, n_items: int) -> list[dict[str, Any]]:
    """Convert linkage rows into node list for dendrogram rendering."""

    by_index: dict[int, dict[str, Any]] = {}

    for idx in range(n_items):
        by_index[idx] = {
            "node_id": f"leaf_{idx}",
            "parent_id": None,
            "children_ids": [],
            "size": 1,
            "height": 0.0,
            "cohesion": 1.0,
            "similarity": 1.0,
            "descendant_leaf_count": 1,
        }

    for row_idx, row in enumerate(linkage_matrix):
        left = int(row[0])
        right = int(row[1])
        dist = float(row[2])
        size = int(row[3])

        node_index = n_items + row_idx
        left_id = by_index[left]["node_id"]
        right_id = by_index[right]["node_id"]

        similarity = max(0.0, 1.0 - dist)
        internal_node = {
            "node_id": f"node_{node_index}",
            "parent_id": None,
            "children_ids": [left_id, right_id],
            "size": size,
            "height": dist,
            "cohesion": similarity,
            "similarity": similarity,
            "descendant_leaf_count": size,
        }
        by_index[node_index] = internal_node

        by_index[left]["parent_id"] = internal_node["node_id"]
        by_index[right]["parent_id"] = internal_node["node_id"]

    ordered_indices = sorted(by_index.keys())
    return [by_index[index] for index in ordered_indices]


def _descendants_by_node(node_by_id: dict[str, dict[str, Any]]) -> dict[str, list[int]]:
    """Return leaf-index descendants for each node."""

    memo: dict[str, list[int]] = {}

    def collect(node_id: str) -> list[int]:
        if node_id in memo:
            return memo[node_id]

        if node_id.startswith("leaf_"):
            suffix = node_id[len("leaf_") :]
            if suffix.isdigit():
                memo[node_id] = [int(suffix)]
            else:
                memo[node_id] = []
            return memo[node_id]

        node = node_by_id.get(node_id)
        if node is None:
            memo[node_id] = []
            return memo[node_id]

        leaf_ids: list[int] = []
        for child_id in node.get("children_ids", []):
            leaf_ids.extend(collect(str(child_id)))
        memo[node_id] = sorted(set(leaf_ids))
        return memo[node_id]

    for node_id in node_by_id:
        collect(node_id)
    return memo


def _leaf_label(items: list[dict[str, Any]], leaf_idx: int) -> str:
    """Build a short leaf label from item text."""

    if leaf_idx < 0 or leaf_idx >= len(items):
        return "Item"
    item = items[leaf_idx]
    if not isinstance(item, dict):
        return "Item"
    text = str(item.get("text", ""))
    terms = _top_terms([text], top_n=2)
    if not terms:
        return "Item"
    return " ".join(token.title() for token in terms[:2])


def _leaf_summary(items: list[dict[str, Any]], leaf_idx: int, max_len: int = 120) -> str:
    """Return a short summary for one leaf item."""

    if leaf_idx < 0 or leaf_idx >= len(items):
        return ""
    item = items[leaf_idx]
    if not isinstance(item, dict):
        return ""
    text = str(item.get("text", "")).strip()
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3].rstrip()}..."


def _compose_node_label(
    counts: Counter[int],
    cluster_label_by_id: dict[int, str],
    dominant_cluster_id: int | None,
    dominant_share: float,
    top_terms: list[str],
) -> str:
    """Compose one node label from cluster composition and terms."""

    if counts and dominant_share < 0.55 and len(counts) >= 3:
        return "Mixed Themes"

    if dominant_cluster_id is not None and dominant_share >= 0.74:
        return str(cluster_label_by_id.get(dominant_cluster_id, f"Cluster {dominant_cluster_id}"))

    ranked = counts.most_common(2)
    if len(ranked) >= 2:
        first_id, first_count = ranked[0]
        second_id, second_count = ranked[1]
        total = max(1, sum(counts.values()))
        first_share = first_count / total
        second_share = second_count / total
        if first_share >= 0.25 and second_share >= 0.2:
            first_label = str(cluster_label_by_id.get(first_id, f"Cluster {first_id}"))
            second_label = str(cluster_label_by_id.get(second_id, f"Cluster {second_id}"))
            return _sanitize_node_label(f"{first_label} + {second_label}")

    if dominant_cluster_id is not None and dominant_share >= 0.55:
        base = str(cluster_label_by_id.get(dominant_cluster_id, f"Cluster {dominant_cluster_id}"))
        return _sanitize_node_label(f"{base} Mixed")

    if top_terms:
        return " ".join(token.title() for token in top_terms[:2])

    return "Theme"


def _top_terms(texts: list[str], top_n: int = 6) -> list[str]:
    """Extract simple frequency-based terms for hierarchy labels."""

    counts: Counter[str] = Counter()
    for text in texts:
        tokens = [token.lower() for token in _WORD_RE.findall(str(text).lower())]
        for token in tokens:
            if token in _STOP_TOKENS:
                continue
            counts[token] += 1
    return [term for term, _ in counts.most_common(top_n)]


def _representative_texts(texts: list[str], limit: int = 5) -> list[str]:
    """Pick a few representative snippets for LLM prompts."""

    unique = []
    seen: set[str] = set()
    for text in sorted(texts, key=lambda value: len(value), reverse=True):
        normalized = text.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
        if len(unique) >= limit:
            break
    return unique


def _sanitize_node_label(label: str) -> str:
    """Clean noisy node labels to short deterministic text."""

    cleaned = re.sub(r"[^a-zA-Z0-9\s\+\-]", "", str(label or "")).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return "Theme"
    words = cleaned.split()
    if len(words) > 6:
        words = words[:6]
    return " ".join(token.title() for token in words)


def _load_label_cache(cache_path: Path) -> dict[str, str]:
    """Load hierarchy label cache."""

    if not cache_path.exists():
        return {}
    data = read_json_file(cache_path)
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _save_label_cache(cache_path: Path, cache: dict[str, str]) -> None:
    """Persist hierarchy label cache."""

    write_json_file(cache_path, cache)
