from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_LINKAGE_ITEMS = 2000
MAX_LARGE_MODE_CLUSTERS = 64


@dataclass
class HierarchyResult:
    root_id: str
    nodes: dict[str, dict[str, Any]]
    leaves: list[dict[str, Any]]
    cluster_ids: list[int]


def _safe_cluster_count(requested_k: int, n_items: int) -> int:
    if n_items <= 1:
        return 1
    try:
        k = int(requested_k)
    except (TypeError, ValueError):
        k = 2
    return max(2, min(k, n_items))


def _remap_cluster_ids(raw_ids: np.ndarray | list[int]) -> list[int]:
    cluster_map: dict[int, int] = {}
    next_id = 0
    out: list[int] = []
    for cid in raw_ids:
        c = int(cid)
        if c not in cluster_map:
            cluster_map[c] = next_id
            next_id += 1
        out.append(cluster_map[c])
    return out


def _build_members_arrays(linkage_matrix: np.ndarray, leaf_members: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    members: dict[int, np.ndarray] = {k: v.astype(np.int32, copy=False) for k, v in leaf_members.items()}
    n_leaves = len(leaf_members)
    for row_idx, row in enumerate(linkage_matrix):
        left = int(row[0])
        right = int(row[1])
        node_id = n_leaves + row_idx
        members[node_id] = np.concatenate((members[left], members[right]))
    return members


def _build_parent_and_children(linkage_matrix: np.ndarray, n_leaves: int) -> tuple[dict[int, int], dict[int, list[int]]]:
    parent_by_node: dict[int, int] = {}
    children: dict[int, list[int]] = defaultdict(list)
    for row_idx, row in enumerate(linkage_matrix):
        left = int(row[0])
        right = int(row[1])
        node_idx = n_leaves + row_idx
        children[node_idx] = [left, right]
        parent_by_node[left] = node_idx
        parent_by_node[right] = node_idx
    return parent_by_node, children


def _annotate_node_metrics(nodes: dict[str, dict[str, Any]]) -> None:
    heights: list[float] = []
    for node in nodes.values():
        if node.get("children_ids"):
            heights.append(float(node.get("height", 0.0)))
    max_height = max(heights) if heights else 0.0

    for node in nodes.values():
        size = int(node.get("size", 1))
        node["descendant_leaf_count"] = size
        if not node.get("children_ids"):
            node["cohesion"] = 1.0
            continue
        height = float(node.get("height", 0.0))
        if max_height <= 1e-9:
            cohesion = 1.0
        else:
            cohesion = 1.0 - (height / max_height)
        node["cohesion"] = float(max(0.0, min(1.0, round(cohesion, 4))))


def _build_standard_hierarchy(embeddings: np.ndarray, item_ids: list[str], k_clusters: int) -> HierarchyResult:
    from scipy.cluster.hierarchy import fcluster, linkage

    n_items = embeddings.shape[0]
    linkage_matrix = linkage(embeddings, method="ward", metric="euclidean")
    requested_k = _safe_cluster_count(k_clusters, n_items)
    clusters_raw = fcluster(linkage_matrix, t=requested_k, criterion="maxclust")
    cluster_ids = _remap_cluster_ids(clusters_raw)

    leaf_members = {i: np.array([i], dtype=np.int32) for i in range(n_items)}
    members = _build_members_arrays(linkage_matrix, leaf_members)
    parent_by_node, children = _build_parent_and_children(linkage_matrix, n_items)

    root_idx = n_items + linkage_matrix.shape[0] - 1
    root_id = "node_root"
    nodes: dict[str, dict[str, Any]] = {}
    for i in range(n_items):
        pid = parent_by_node.get(i)
        node_id = f"leaf_{i}"
        nodes[node_id] = {
            "node_id": node_id,
            "parent_id": root_id if pid == root_idx else (f"node_{pid}" if pid is not None else root_id),
            "children_ids": [],
            "size": 1,
            "height": 0.0,
            "label": "Item",
            "member_indices": [i],
        }

    for row_idx, row in enumerate(linkage_matrix):
        node_idx = n_items + row_idx
        node_id = root_id if node_idx == root_idx else f"node_{node_idx}"
        pid = parent_by_node.get(node_idx)
        child_ids: list[str] = []
        for ch in children[node_idx]:
            if ch < n_items:
                child_ids.append(f"leaf_{ch}")
            else:
                child_ids.append(root_id if ch == root_idx else f"node_{ch}")
        nodes[node_id] = {
            "node_id": node_id,
            "parent_id": None if node_idx == root_idx else (root_id if pid == root_idx else f"node_{pid}"),
            "children_ids": child_ids,
            "size": int(row[3]),
            "height": float(row[2]),
            "label": "Topic",
            "member_indices": members[node_idx].tolist(),
        }

    _annotate_node_metrics(nodes)
    leaves = [{"id": item_ids[i], "node_id": f"leaf_{i}", "cluster_id": cluster_ids[i]} for i in range(n_items)]
    return HierarchyResult(root_id=root_id, nodes=nodes, leaves=leaves, cluster_ids=cluster_ids)


def _centroid_node_id(centroid_idx: int) -> str:
    return f"node_centroid_{centroid_idx}"


def _build_large_hierarchy(embeddings: np.ndarray, item_ids: list[str], k_clusters: int) -> HierarchyResult:
    from scipy.cluster.hierarchy import linkage
    from sklearn.cluster import MiniBatchKMeans

    n_items = embeddings.shape[0]
    requested_k = _safe_cluster_count(k_clusters, n_items)
    n_centroids_target = min(requested_k, MAX_LARGE_MODE_CLUSTERS, n_items)
    kmeans = MiniBatchKMeans(
        n_clusters=n_centroids_target,
        random_state=42,
        batch_size=min(4096, max(256, n_centroids_target * 32)),
        n_init=10,
        reassignment_ratio=0.01,
    )
    raw_labels = kmeans.fit_predict(embeddings)
    cluster_ids = _remap_cluster_ids(raw_labels)
    labels_arr = np.array(cluster_ids, dtype=np.int32)

    centroid_members: dict[int, np.ndarray] = {}
    centroid_vectors: list[np.ndarray] = []
    unique_cluster_ids = sorted(set(cluster_ids))
    for cid in unique_cluster_ids:
        member_idx = np.where(labels_arr == cid)[0].astype(np.int32, copy=False)
        centroid_members[cid] = member_idx
        centroid_vectors.append(embeddings[member_idx].mean(axis=0))

    n_centroids = len(unique_cluster_ids)
    root_id = "node_root"
    nodes: dict[str, dict[str, Any]] = {}

    for i in range(n_items):
        cid = cluster_ids[i]
        node_id = f"leaf_{i}"
        nodes[node_id] = {
            "node_id": node_id,
            "parent_id": _centroid_node_id(cid),
            "children_ids": [],
            "size": 1,
            "height": 0.0,
            "label": "Item",
            "member_indices": [i],
        }

    if n_centroids <= 1:
        only_members = centroid_members.get(0, np.arange(n_items, dtype=np.int32))
        member_list = only_members.tolist()
        centroid_node = _centroid_node_id(0)
        nodes[centroid_node] = {
            "node_id": centroid_node,
            "parent_id": root_id,
            "children_ids": [f"leaf_{int(i)}" for i in only_members],
            "size": int(only_members.size),
            "height": 0.0,
            "label": "Topic",
            "member_indices": member_list,
        }
        nodes[root_id] = {
            "node_id": root_id,
            "parent_id": None,
            "children_ids": [centroid_node],
            "size": int(only_members.size),
            "height": 0.0,
            "label": "Topic",
            "member_indices": member_list,
        }
        _annotate_node_metrics(nodes)
        leaves = [{"id": item_ids[i], "node_id": f"leaf_{i}", "cluster_id": cluster_ids[i]} for i in range(n_items)]
        return HierarchyResult(root_id=root_id, nodes=nodes, leaves=leaves, cluster_ids=cluster_ids)

    centroid_matrix = np.vstack(centroid_vectors).astype(np.float32, copy=False)
    linkage_matrix = linkage(centroid_matrix, method="ward", metric="euclidean")

    leaf_members = {i: centroid_members[i] for i in range(n_centroids)}
    members = _build_members_arrays(linkage_matrix, leaf_members)
    parent_by_node, children = _build_parent_and_children(linkage_matrix, n_centroids)
    root_idx = n_centroids + linkage_matrix.shape[0] - 1

    for centroid_idx in range(n_centroids):
        pid = parent_by_node.get(centroid_idx)
        member_idx = centroid_members[centroid_idx]
        node_id = _centroid_node_id(centroid_idx)
        nodes[node_id] = {
            "node_id": node_id,
            "parent_id": root_id if pid == root_idx else (f"node_{pid}" if pid is not None else root_id),
            "children_ids": [f"leaf_{int(i)}" for i in member_idx],
            "size": int(member_idx.size),
            "height": 0.0,
            "label": "Topic",
            "member_indices": member_idx.tolist(),
        }

    for row_idx, row in enumerate(linkage_matrix):
        node_idx = n_centroids + row_idx
        node_id = root_id if node_idx == root_idx else f"node_{node_idx}"
        pid = parent_by_node.get(node_idx)
        child_ids: list[str] = []
        for ch in children[node_idx]:
            if ch < n_centroids:
                child_ids.append(_centroid_node_id(ch))
            else:
                child_ids.append(root_id if ch == root_idx else f"node_{ch}")
        member_idx = members[node_idx]
        nodes[node_id] = {
            "node_id": node_id,
            "parent_id": None if node_idx == root_idx else (root_id if pid == root_idx else f"node_{pid}"),
            "children_ids": child_ids,
            "size": int(member_idx.size),
            "height": float(row[2]),
            "label": "Topic",
            "member_indices": member_idx.tolist(),
        }

    _annotate_node_metrics(nodes)
    leaves = [{"id": item_ids[i], "node_id": f"leaf_{i}", "cluster_id": cluster_ids[i]} for i in range(n_items)]
    return HierarchyResult(root_id=root_id, nodes=nodes, leaves=leaves, cluster_ids=cluster_ids)


def build_hierarchy(
    embeddings: np.ndarray,
    item_ids: list[str],
    k_clusters: int,
) -> HierarchyResult:
    n_items = embeddings.shape[0]
    if n_items != len(item_ids):
        raise ValueError("item_ids length must match embeddings")

    if n_items == 1:
        root_id = "node_root"
        nodes = {
            root_id: {
                "node_id": root_id,
                "parent_id": None,
                "children_ids": ["leaf_0"],
                "size": 1,
                "height": 0.0,
                "label": "Single",
                "member_indices": [0],
                "descendant_leaf_count": 1,
                "cohesion": 1.0,
            },
            "leaf_0": {
                "node_id": "leaf_0",
                "parent_id": root_id,
                "children_ids": [],
                "size": 1,
                "height": 0.0,
                "label": "Item",
                "member_indices": [0],
                "descendant_leaf_count": 1,
                "cohesion": 1.0,
            },
        }
        leaves = [{"id": item_ids[0], "node_id": "leaf_0", "cluster_id": 0}]
        return HierarchyResult(root_id=root_id, nodes=nodes, leaves=leaves, cluster_ids=[0])

    if n_items > MAX_LINKAGE_ITEMS:
        return _build_large_hierarchy(embeddings=embeddings, item_ids=item_ids, k_clusters=k_clusters)
    return _build_standard_hierarchy(embeddings=embeddings, item_ids=item_ids, k_clusters=k_clusters)
