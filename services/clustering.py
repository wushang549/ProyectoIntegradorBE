"""Flat clustering utilities using scipy fcluster."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score

from services.thematics import bucket_key, theme_from_item


def normalize_k_clusters(k_clusters: int | None, n_items: int, default_k: int = 8) -> int:
    """Clamp requested cluster count to valid bounds for current data."""

    if n_items <= 0:
        return 0
    if n_items == 1:
        return 1

    if k_clusters is None:
        k_clusters = default_k
    return max(2, min(int(k_clusters), min(100, n_items)))


def cut_clusters(linkage_matrix: np.ndarray, n_items: int, k_clusters: int) -> np.ndarray:
    """Cut hierarchical tree into flat clusters."""

    if n_items == 0:
        return np.array([], dtype=int)
    if n_items == 1:
        return np.array([1], dtype=int)
    if linkage_matrix.size == 0:
        return np.ones((n_items,), dtype=int)

    labels = fcluster(linkage_matrix, t=k_clusters, criterion="maxclust")
    return labels.astype(int)


def choose_auto_k(
    linkage_matrix: np.ndarray,
    vectors: np.ndarray,
    n_items: int,
    min_k: int = 2,
    max_k: int = 16,
) -> dict[str, Any]:
    """Select k automatically from candidate cuts using quality-aware scoring."""

    resolved_default = normalize_k_clusters(None, n_items=n_items)
    if n_items <= 2:
        labels = cut_clusters(linkage_matrix, n_items=n_items, k_clusters=resolved_default)
        quality = evaluate_cluster_partition(vectors=vectors, cluster_labels=labels)
        quality["tested_k"] = int(resolved_default)
        quality["score"] = float(_score_candidate(resolved_default, resolved_default, quality))
        return {
            "k_clusters": int(resolved_default),
            "target_k": int(resolved_default),
            "mode": "auto",
            "selected_quality": quality,
            "candidates": [quality],
        }

    upper_bound = max(min_k, min(max_k, n_items))
    lower_bound = min(min_k, upper_bound)
    target_k = max(lower_bound, min(upper_bound, int(round(np.sqrt(n_items)))))

    diagnostics: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for k in range(lower_bound, upper_bound + 1):
        labels = cut_clusters(linkage_matrix, n_items=n_items, k_clusters=k)
        quality = evaluate_cluster_partition(vectors=vectors, cluster_labels=labels)
        quality["tested_k"] = int(k)
        quality["score"] = float(_score_candidate(k_clusters=k, target_k=target_k, quality=quality))
        diagnostics.append(quality)

        if best is None:
            best = quality
            continue

        if quality["score"] > best["score"] + 1e-9:
            best = quality
            continue

        if abs(quality["score"] - best["score"]) <= 1e-9:
            if quality["largest_cluster_share"] < best["largest_cluster_share"] - 1e-9:
                best = quality
                continue
            if abs(quality["largest_cluster_share"] - best["largest_cluster_share"]) <= 1e-9:
                if abs(quality["tested_k"] - target_k) < abs(best["tested_k"] - target_k):
                    best = quality

    if best is None:
        labels = cut_clusters(linkage_matrix, n_items=n_items, k_clusters=resolved_default)
        quality = evaluate_cluster_partition(vectors=vectors, cluster_labels=labels)
        quality["tested_k"] = int(resolved_default)
        quality["score"] = float(_score_candidate(resolved_default, resolved_default, quality))
        return {
            "k_clusters": int(resolved_default),
            "target_k": int(resolved_default),
            "mode": "auto",
            "selected_quality": quality,
            "candidates": [quality],
        }

    ordered_candidates = sorted(
        diagnostics,
        key=lambda item: (
            float(item.get("score", -1.0)),
            -float(item.get("largest_cluster_share", 1.0)),
            -abs(int(item.get("tested_k", target_k)) - target_k),
        ),
        reverse=True,
    )
    return {
        "k_clusters": int(best["tested_k"]),
        "target_k": int(target_k),
        "mode": "auto",
        "selected_quality": best,
        "candidates": ordered_candidates[:8],
    }


def recommended_min_cluster_size(n_items: int) -> int:
    """Return the minimum cluster size used for auto-mode cleanup."""

    if n_items >= 80:
        return 3
    if n_items >= 30:
        return 2
    return 1


def merge_small_clusters(
    cluster_labels: np.ndarray,
    vectors: np.ndarray,
    min_size: int = 2,
) -> np.ndarray:
    """Merge tiny clusters into nearest larger clusters using cosine similarity."""

    labels = np.asarray(cluster_labels, dtype=int).copy()
    if labels.size == 0 or min_size <= 1:
        return labels

    counts = Counter(labels.tolist())
    large_ids = [cluster_id for cluster_id, count in counts.items() if count >= min_size]
    small_ids = [cluster_id for cluster_id, count in counts.items() if count < min_size]
    if not small_ids or not large_ids:
        return labels

    large_ids_arr = np.asarray(sorted(large_ids), dtype=int)
    centroids = np.stack(
        [
            vectors[np.where(labels == cluster_id)[0]].mean(axis=0)
            for cluster_id in large_ids_arr.tolist()
        ],
        axis=0,
    )
    centroid_norms = np.linalg.norm(centroids, axis=1)
    centroid_norms[centroid_norms == 0.0] = 1.0

    for cluster_id in sorted(small_ids):
        member_idx = np.where(labels == cluster_id)[0]
        if member_idx.size == 0:
            continue

        members = vectors[member_idx]
        member_norms = np.linalg.norm(members, axis=1)
        member_norms[member_norms == 0.0] = 1.0

        similarities = (members @ centroids.T) / (member_norms[:, None] * centroid_norms[None, :])
        target_positions = np.argmax(similarities, axis=1)
        labels[member_idx] = large_ids_arr[target_positions]

    return labels.astype(int)


def auto_cluster_with_buckets(
    vectors: np.ndarray,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Cluster items by aspect/polarity buckets, then split each bucket semantically."""

    n_items = int(vectors.shape[0])
    if n_items == 0:
        return {
            "cluster_labels": np.array([], dtype=int),
            "k_clusters": 0,
            "bucket_diagnostics": [],
        }
    if n_items == 1:
        return {
            "cluster_labels": np.array([1], dtype=int),
            "k_clusters": 1,
            "bucket_diagnostics": [{"bucket": "general:neutral", "size": 1, "local_k": 1}],
        }

    buckets = _build_theme_buckets(items=items, n_items=n_items)
    min_bucket_size = max(2, recommended_min_cluster_size(n_items))
    buckets = _merge_tiny_buckets(buckets=buckets, vectors=vectors, min_size=min_bucket_size)

    labels = np.full((n_items,), fill_value=-1, dtype=int)
    next_cluster_id = 1
    diagnostics: list[dict[str, Any]] = []

    ordered_buckets = sorted(buckets.items(), key=lambda pair: (-len(pair[1]), pair[0]))
    for bucket, indices in ordered_buckets:
        if not indices:
            continue

        local_vectors = vectors[indices]
        local_labels, local_k = _cluster_bucket_subset(local_vectors)

        local_ids = sorted(np.unique(local_labels).tolist())
        remap = {int(local_id): int(next_cluster_id + offset) for offset, local_id in enumerate(local_ids)}
        for local_pos, global_idx in enumerate(indices):
            labels[global_idx] = remap[int(local_labels[local_pos])]

        next_cluster_id += len(local_ids)
        diagnostics.append(
            {
                "bucket": bucket,
                "size": int(len(indices)),
                "local_k": int(len(local_ids)),
                "requested_local_k": int(local_k),
            }
        )

    labels = _fill_unassigned_labels(labels)
    labels = _purity_reassign(labels=labels, vectors=vectors, items=items)
    labels = _compact_partition(labels=labels, vectors=vectors, items=items)
    labels = _renumber_cluster_ids(labels)

    return {
        "cluster_labels": labels.astype(int),
        "k_clusters": int(np.unique(labels).size),
        "bucket_diagnostics": diagnostics,
    }


def evaluate_cluster_partition(vectors: np.ndarray, cluster_labels: np.ndarray) -> dict[str, Any]:
    """Compute lightweight quality metrics for one cluster partition."""

    labels = np.asarray(cluster_labels, dtype=int)
    if labels.size == 0:
        return {
            "k_clusters": 0,
            "cluster_count": 0,
            "largest_cluster_share": 0.0,
            "singleton_share": 0.0,
            "silhouette": -1.0,
        }

    counts = np.asarray(list(Counter(labels.tolist()).values()), dtype=np.int32)
    total_items = int(labels.size)
    largest_cluster_share = float(counts.max() / total_items) if counts.size else 0.0
    singleton_share = float((counts == 1).sum() / total_items) if total_items > 0 else 0.0
    silhouette = _safe_silhouette(vectors=vectors, cluster_labels=labels)

    return {
        "k_clusters": int(counts.size),
        "cluster_count": int(counts.size),
        "largest_cluster_share": largest_cluster_share,
        "singleton_share": singleton_share,
        "silhouette": silhouette,
    }


def cluster_to_indices(cluster_labels: Iterable[int]) -> dict[int, list[int]]:
    """Group item indexes by cluster id."""

    mapping: dict[int, list[int]] = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_labels):
        mapping[int(cluster_id)].append(idx)
    return dict(sorted(mapping.items(), key=lambda pair: pair[0]))


def _safe_silhouette(vectors: np.ndarray, cluster_labels: np.ndarray) -> float:
    """Compute cosine silhouette with bounded runtime and fallback safety."""

    unique_labels = np.unique(cluster_labels)
    if unique_labels.size < 2 or unique_labels.size >= cluster_labels.size:
        return -1.0

    labels = cluster_labels
    subset = vectors
    if cluster_labels.size > 600:
        rng = np.random.default_rng(42)
        sample_idx = np.sort(rng.choice(cluster_labels.size, size=600, replace=False))
        labels = cluster_labels[sample_idx]
        subset = vectors[sample_idx]
        if np.unique(labels).size < 2:
            return -1.0

    try:
        score = float(silhouette_score(subset, labels, metric="cosine"))
    except Exception:
        return -1.0

    if np.isnan(score) or np.isinf(score):
        return -1.0
    return score


def _score_candidate(k_clusters: int, target_k: int, quality: dict[str, Any]) -> float:
    """Convert quality metrics into one ranking score for auto-k selection."""

    silhouette = float(quality.get("silhouette", -1.0))
    largest_share = float(quality.get("largest_cluster_share", 1.0))
    singleton_share = float(quality.get("singleton_share", 1.0))
    cluster_count = int(quality.get("cluster_count", 0))

    dominance_penalty = max(0.0, largest_share - 0.45) * 1.2
    singleton_penalty = singleton_share * 0.35
    fragmentation_penalty = max(0, int(k_clusters) - int(target_k)) * 0.012
    degenerate_penalty = 0.08 if cluster_count < 2 else 0.0

    return silhouette - dominance_penalty - singleton_penalty - fragmentation_penalty - degenerate_penalty


def _build_theme_buckets(items: list[dict[str, Any]], n_items: int) -> dict[str, list[int]]:
    """Group item indices by inferred aspect/polarity."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx in range(n_items):
        item = items[idx] if idx < len(items) else {}
        if not isinstance(item, dict):
            item = {}
        tags = theme_from_item(item)
        key = bucket_key(tags["aspect"], tags["polarity"])
        grouped[key].append(idx)
    return {str(key): sorted(indices) for key, indices in grouped.items()}


def _merge_tiny_buckets(
    buckets: dict[str, list[int]],
    vectors: np.ndarray,
    min_size: int,
) -> dict[str, list[int]]:
    """Merge tiny buckets into nearest compatible larger buckets."""

    if min_size <= 1:
        return buckets

    large = {key: sorted(indices) for key, indices in buckets.items() if len(indices) >= min_size}
    tiny = {key: sorted(indices) for key, indices in buckets.items() if 0 < len(indices) < min_size}
    if not tiny or not large:
        return {key: sorted(indices) for key, indices in buckets.items() if indices}

    centroids = {key: vectors[np.asarray(indices, dtype=int)].mean(axis=0) for key, indices in large.items() if indices}
    centroid_norm = {key: float(np.linalg.norm(value)) or 1.0 for key, value in centroids.items()}

    for tiny_key, tiny_indices in sorted(tiny.items(), key=lambda pair: len(pair[1])):
        src_aspect, src_polarity = _split_bucket_key(tiny_key)
        for idx in tiny_indices:
            member = vectors[idx]
            member_norm = float(np.linalg.norm(member)) or 1.0
            best_key = None
            best_score = -1.0
            for dst_key, centroid in centroids.items():
                dot = float(member @ centroid)
                sim = dot / (member_norm * centroid_norm[dst_key])
                dst_aspect, dst_polarity = _split_bucket_key(dst_key)
                bonus = 0.0
                if src_aspect == dst_aspect:
                    bonus += 0.08
                if src_polarity == dst_polarity:
                    bonus += 0.05
                score = sim + bonus
                if score > best_score:
                    best_score = score
                    best_key = dst_key
            if best_key is None:
                continue
            large[best_key].append(int(idx))

    for key, indices in large.items():
        large[key] = sorted(indices)
    return large


def _cluster_bucket_subset(local_vectors: np.ndarray) -> tuple[np.ndarray, int]:
    """Find local partition for one bucket."""

    n_items = int(local_vectors.shape[0])
    if n_items <= 1:
        return np.ones((n_items,), dtype=int), 1
    if n_items <= 3:
        return np.ones((n_items,), dtype=int), 1

    mean_sim = _mean_cosine_similarity(local_vectors)
    if n_items <= 6 and mean_sim >= 0.58:
        return np.ones((n_items,), dtype=int), 1

    local_linkage = _safe_linkage(local_vectors)
    if n_items <= 8:
        max_local_k = 2
    elif n_items <= 16:
        max_local_k = 3
    else:
        max_local_k = 4
    max_local_k = min(max_local_k, n_items)
    auto = choose_auto_k(
        linkage_matrix=local_linkage,
        vectors=local_vectors,
        n_items=n_items,
        min_k=2,
        max_k=max_local_k,
    )
    local_k = max(1, min(int(auto.get("k_clusters", 2)), max_local_k))
    if local_k <= 1:
        return np.ones((n_items,), dtype=int), 1

    local_labels = cut_clusters(local_linkage, n_items=n_items, k_clusters=local_k)
    local_labels = merge_small_clusters(
        cluster_labels=local_labels,
        vectors=local_vectors,
        min_size=2 if n_items >= 10 else 1,
    )
    resolved_k = int(np.unique(local_labels).size)
    return local_labels.astype(int), resolved_k


def _mean_cosine_similarity(vectors: np.ndarray) -> float:
    """Approximate average pairwise cosine similarity."""

    if vectors.shape[0] <= 1:
        return 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = vectors / norms
    sim = normalized @ normalized.T
    n = sim.shape[0]
    upper = sim[np.triu_indices(n, k=1)]
    if upper.size == 0:
        return 1.0
    return float(np.mean(upper))


def _safe_linkage(vectors: np.ndarray) -> np.ndarray:
    """Compute linkage matrix with cosine fallback to euclidean."""

    try:
        return linkage(vectors, method="average", metric="cosine")
    except Exception:
        return linkage(vectors, method="average", metric="euclidean")


def _fill_unassigned_labels(labels: np.ndarray) -> np.ndarray:
    """Fill any unassigned labels with singleton cluster ids."""

    output = labels.astype(int).copy()
    current_max = int(np.max(output)) if output.size and np.max(output) > 0 else 0
    for idx, value in enumerate(output):
        if int(value) > 0:
            continue
        current_max += 1
        output[idx] = current_max
    return output


def _purity_reassign(
    labels: np.ndarray,
    vectors: np.ndarray,
    items: list[dict[str, Any]],
    max_move_ratio: float = 0.14,
) -> np.ndarray:
    """Reassign likely outliers to better matching clusters."""

    output = labels.astype(int).copy()
    if output.size <= 3:
        return output

    cluster_members = cluster_to_indices(output.tolist())
    if len(cluster_members) <= 1:
        return output

    theme_by_cluster = {
        int(cluster_id): _cluster_theme_signature(items=items, indices=indices)
        for cluster_id, indices in cluster_members.items()
    }
    centroids = {cluster_id: vectors[np.asarray(indices, dtype=int)].mean(axis=0) for cluster_id, indices in cluster_members.items()}
    centroid_norm = {cluster_id: float(np.linalg.norm(value)) or 1.0 for cluster_id, value in centroids.items()}

    max_moves = max(1, int(round(output.size * max_move_ratio)))
    moves = 0
    for idx in range(output.size):
        if moves >= max_moves:
            break
        current_cluster = int(output[idx])
        members = cluster_members.get(current_cluster, [])
        if len(members) <= 2:
            continue

        item_theme = theme_from_item(items[idx] if idx < len(items) and isinstance(items[idx], dict) else {})
        signature = theme_by_cluster.get(current_cluster, {})
        mismatch = _theme_mismatch(item_theme=item_theme, cluster_theme=signature)
        if mismatch <= 0:
            continue

        current_sim = _cosine_similarity_to_centroid(vectors[idx], centroids[current_cluster], centroid_norm[current_cluster])
        best_cluster = current_cluster
        best_sim = current_sim

        for candidate_cluster, candidate_theme in theme_by_cluster.items():
            if candidate_cluster == current_cluster:
                continue
            compatible = _theme_compatibility(item_theme=item_theme, cluster_theme=candidate_theme)
            if compatible <= 0.0:
                continue

            candidate_sim = _cosine_similarity_to_centroid(
                vectors[idx],
                centroids[candidate_cluster],
                centroid_norm[candidate_cluster],
            )
            score = candidate_sim + compatible
            if score > (best_sim + 1e-9):
                best_sim = score
                best_cluster = candidate_cluster

        margin = 0.05 if mismatch >= 2 else 0.08
        if best_cluster != current_cluster and best_sim >= current_sim + margin:
            output[idx] = int(best_cluster)
            moves += 1

    return output


def _compact_partition(
    labels: np.ndarray,
    vectors: np.ndarray,
    items: list[dict[str, Any]],
) -> np.ndarray:
    """Merge micro-clusters and cap total cluster count for readability."""

    output = labels.astype(int).copy()
    if output.size <= 3:
        return output

    n_items = int(output.size)
    min_size = 3 if n_items >= 45 else (2 if n_items >= 20 else 1)
    output = _merge_small_clusters_theme(
        labels=output,
        vectors=vectors,
        items=items,
        min_size=min_size,
    )
    output = _merge_near_duplicate_clusters(
        labels=output,
        vectors=vectors,
        items=items,
    )

    target_k = max(4, min(24, int(round(np.sqrt(n_items) * 1.6))))
    for _ in range(128):
        cluster_members = cluster_to_indices(output.tolist())
        if len(cluster_members) <= target_k:
            break
        smallest_cluster = min(cluster_members.items(), key=lambda pair: (len(pair[1]), pair[0]))[0]
        target_cluster = _best_cluster_merge_target(
            source_cluster=int(smallest_cluster),
            labels=output,
            vectors=vectors,
            items=items,
        )
        if target_cluster is None or target_cluster == smallest_cluster:
            break
        output[output == int(smallest_cluster)] = int(target_cluster)
        output = _renumber_cluster_ids(output)

    return output


def _merge_near_duplicate_clusters(
    labels: np.ndarray,
    vectors: np.ndarray,
    items: list[dict[str, Any]],
    strict_similarity: float = 0.965,
    small_similarity: float = 0.94,
) -> np.ndarray:
    """Merge clusters that are almost identical by theme signature + centroid similarity."""

    output = labels.astype(int).copy()
    if output.size <= 3:
        return output

    for _ in range(64):
        members = cluster_to_indices(output.tolist())
        if len(members) <= 1:
            break

        themes = {int(cluster_id): _cluster_theme_signature(items=items, indices=idxs) for cluster_id, idxs in members.items()}
        centroids = {int(cluster_id): vectors[np.asarray(idxs, dtype=int)].mean(axis=0) for cluster_id, idxs in members.items() if idxs}
        norms = {cluster_id: float(np.linalg.norm(value)) or 1.0 for cluster_id, value in centroids.items()}

        merge_pair: tuple[int, int] | None = None
        best_score = -1.0
        cluster_ids = sorted(members.keys())
        for i, left_id in enumerate(cluster_ids):
            left_members = members.get(left_id, [])
            if not left_members:
                continue
            for right_id in cluster_ids[i + 1 :]:
                right_members = members.get(right_id, [])
                if not right_members:
                    continue
                if not _can_merge_as_duplicate(themes.get(int(left_id), {}), themes.get(int(right_id), {})):
                    continue

                sim = _cosine_similarity_to_centroid(
                    centroids[int(left_id)],
                    centroids[int(right_id)],
                    norms[int(right_id)],
                )
                min_size = min(len(left_members), len(right_members))
                threshold = small_similarity if min_size <= 3 else strict_similarity
                if sim < threshold:
                    continue
                if sim > best_score:
                    best_score = sim
                    merge_pair = (int(left_id), int(right_id))

        if merge_pair is None:
            break

        left_id, right_id = merge_pair
        left_size = len(members.get(left_id, []))
        right_size = len(members.get(right_id, []))
        source = left_id if left_size <= right_size else right_id
        target = right_id if source == left_id else left_id
        output[output == int(source)] = int(target)
        output = _renumber_cluster_ids(output)

    return output


def _merge_small_clusters_theme(
    labels: np.ndarray,
    vectors: np.ndarray,
    items: list[dict[str, Any]],
    min_size: int,
) -> np.ndarray:
    """Merge clusters below min size into most compatible larger cluster."""

    output = labels.astype(int).copy()
    if min_size <= 1:
        return output

    for _ in range(64):
        members = cluster_to_indices(output.tolist())
        small = [cluster_id for cluster_id, idxs in members.items() if len(idxs) < min_size]
        if not small:
            break
        moved = False
        for cluster_id in sorted(small):
            target = _best_cluster_merge_target(
                source_cluster=int(cluster_id),
                labels=output,
                vectors=vectors,
                items=items,
            )
            if target is None or target == cluster_id:
                continue
            output[output == int(cluster_id)] = int(target)
            moved = True
        if not moved:
            break
        output = _renumber_cluster_ids(output)

    return output


def _best_cluster_merge_target(
    source_cluster: int,
    labels: np.ndarray,
    vectors: np.ndarray,
    items: list[dict[str, Any]],
) -> int | None:
    """Find best target cluster for merging one source cluster."""

    members = cluster_to_indices(labels.tolist())
    source_members = members.get(int(source_cluster), [])
    if not source_members:
        return None

    source_centroid = vectors[np.asarray(source_members, dtype=int)].mean(axis=0)
    source_norm = float(np.linalg.norm(source_centroid)) or 1.0
    source_theme = _cluster_theme_signature(items=items, indices=source_members)

    best_id = None
    best_score = -1.0
    for cluster_id, idxs in members.items():
        if int(cluster_id) == int(source_cluster) or not idxs:
            continue
        target_centroid = vectors[np.asarray(idxs, dtype=int)].mean(axis=0)
        target_norm = float(np.linalg.norm(target_centroid)) or 1.0
        sim = float((source_centroid @ target_centroid) / (source_norm * target_norm))
        target_theme = _cluster_theme_signature(items=items, indices=idxs)
        bonus = _theme_compatibility(
            item_theme={"aspect": source_theme.get("aspect", "general"), "polarity": source_theme.get("polarity", "neutral")},
            cluster_theme=target_theme,
        )
        score = sim + bonus
        if score > best_score:
            best_score = score
            best_id = int(cluster_id)

    return best_id


def _cluster_theme_signature(items: list[dict[str, Any]], indices: list[int]) -> dict[str, str]:
    """Compute dominant aspect and polarity labels for a cluster."""

    if not indices:
        return {"aspect": "general", "polarity": "neutral"}

    aspect_counter: Counter[str] = Counter()
    polarity_counter: Counter[str] = Counter()
    for idx in indices:
        if idx < 0 or idx >= len(items):
            continue
        item = items[idx] if isinstance(items[idx], dict) else {}
        tags = theme_from_item(item)
        aspect_counter[str(tags["aspect"])] += 1
        polarity_counter[str(tags["polarity"])] += 1

    dominant_aspect = _dominant_counter_value(aspect_counter, default="general")
    dominant_polarity = _dominant_counter_value(polarity_counter, default="neutral")
    return {"aspect": dominant_aspect, "polarity": dominant_polarity}


def _theme_mismatch(item_theme: dict[str, str], cluster_theme: dict[str, str]) -> int:
    """Count hard mismatches for aspect/polarity."""

    mismatch = 0
    item_aspect = str(item_theme.get("aspect", "general"))
    item_polarity = str(item_theme.get("polarity", "neutral"))
    cluster_aspect = str(cluster_theme.get("aspect", "general"))
    cluster_polarity = str(cluster_theme.get("polarity", "neutral"))

    if cluster_aspect != "general" and item_aspect != "general" and item_aspect != cluster_aspect:
        mismatch += 1
    if cluster_polarity in {"positive", "negative"} and item_polarity in {"positive", "negative"} and item_polarity != cluster_polarity:
        mismatch += 1
    return mismatch


def _theme_compatibility(item_theme: dict[str, str], cluster_theme: dict[str, str]) -> float:
    """Return compatibility bonus between item theme and cluster signature."""

    bonus = 0.0
    item_aspect = str(item_theme.get("aspect", "general"))
    item_polarity = str(item_theme.get("polarity", "neutral"))
    cluster_aspect = str(cluster_theme.get("aspect", "general"))
    cluster_polarity = str(cluster_theme.get("polarity", "neutral"))

    if item_aspect == cluster_aspect and item_aspect != "general":
        bonus += 0.12
    if item_polarity == cluster_polarity and item_polarity in {"positive", "negative"}:
        bonus += 0.08
    return bonus


def _can_merge_as_duplicate(left_theme: dict[str, str], right_theme: dict[str, str]) -> bool:
    """Return True when two clusters are semantically the same theme family."""

    left_aspect = str(left_theme.get("aspect", "general"))
    right_aspect = str(right_theme.get("aspect", "general"))
    left_polarity = str(left_theme.get("polarity", "neutral"))
    right_polarity = str(right_theme.get("polarity", "neutral"))

    if left_aspect != right_aspect:
        return False
    if left_polarity != right_polarity:
        return False

    # Avoid collapsing broad general-neutral clusters aggressively.
    if left_aspect == "general" and left_polarity in {"neutral", "mixed"}:
        return False
    return True


def _cosine_similarity_to_centroid(vector: np.ndarray, centroid: np.ndarray, centroid_norm: float) -> float:
    """Cosine similarity between one vector and a centroid."""

    vector_norm = float(np.linalg.norm(vector)) or 1.0
    return float((vector @ centroid) / (vector_norm * (centroid_norm or 1.0)))


def _renumber_cluster_ids(labels: np.ndarray) -> np.ndarray:
    """Normalize cluster ids to stable consecutive ids starting at 1."""

    unique = sorted(np.unique(labels).tolist())
    mapping = {int(raw): int(pos + 1) for pos, raw in enumerate(unique)}
    return np.asarray([mapping.get(int(value), 1) for value in labels], dtype=int)


def _split_bucket_key(value: str) -> tuple[str, str]:
    parts = str(value or "").split(":", maxsplit=1)
    if len(parts) != 2:
        return "general", "neutral"
    return parts[0], parts[1]


def _dominant_counter_value(counter: Counter[str], default: str) -> str:
    if not counter:
        return default
    max_count = max(counter.values())
    tied = sorted(key for key, value in counter.items() if value == max_count)
    if default in tied:
        return default
    return tied[0]
