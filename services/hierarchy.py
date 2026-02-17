from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class HierarchyResult:
    root_id: str
    nodes: dict[str, dict[str, Any]]
    leaves: list[dict[str, Any]]
    cluster_ids: list[int]


def _safe_cluster_count(requested_k: int, n_items: int) -> int:
    if n_items <= 1:
        return 1
    return max(2, min(int(requested_k), n_items))


def _build_members(linkage_matrix: np.ndarray, n_items: int) -> dict[int, list[int]]:
    members: dict[int, list[int]] = {i: [i] for i in range(n_items)}
    for row_idx, row in enumerate(linkage_matrix):
        left = int(row[0])
        right = int(row[1])
        node_id = n_items + row_idx
        members[node_id] = members[left] + members[right]
    return members


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
            },
            "leaf_0": {
                "node_id": "leaf_0",
                "parent_id": root_id,
                "children_ids": [],
                "size": 1,
                "height": 0.0,
                "label": "Item",
            },
        }
        leaves = [{"id": item_ids[0], "node_id": "leaf_0", "cluster_id": 0}]
        return HierarchyResult(root_id=root_id, nodes=nodes, leaves=leaves, cluster_ids=[0])

    from scipy.cluster.hierarchy import fcluster, linkage

    linkage_matrix = linkage(embeddings, method="ward", metric="euclidean")
    requested_k = _safe_cluster_count(k_clusters, n_items)
    clusters_raw = fcluster(linkage_matrix, t=requested_k, criterion="maxclust")

    cluster_map: dict[int, int] = {}
    cluster_ids: list[int] = []
    next_id = 0
    for cid in clusters_raw:
        c = int(cid)
        if c not in cluster_map:
            cluster_map[c] = next_id
            next_id += 1
        cluster_ids.append(cluster_map[c])

    members = _build_members(linkage_matrix, n_items)
    nodes: dict[str, dict[str, Any]] = {}
    parent_by_node: dict[int, int] = {}
    children: dict[int, list[int]] = defaultdict(list)

    for row_idx, row in enumerate(linkage_matrix):
        left = int(row[0])
        right = int(row[1])
        node_idx = n_items + row_idx
        children[node_idx] = [left, right]
        parent_by_node[left] = node_idx
        parent_by_node[right] = node_idx

    root_idx = n_items + linkage_matrix.shape[0] - 1
    root_id = "node_root"

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
            "member_indices": members[node_idx],
        }

    leaves = [{"id": item_ids[i], "node_id": f"leaf_{i}", "cluster_id": cluster_ids[i]} for i in range(n_items)]
    return HierarchyResult(root_id=root_id, nodes=nodes, leaves=leaves, cluster_ids=cluster_ids)
