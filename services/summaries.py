"""Cluster summaries: top terms and representative items."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from services.clustering import cluster_to_indices
from services.thematics import dominant_theme

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{1,}")
_GENERIC_TERMS = {
    "food",
    "service",
    "restaurant",
    "staff",
    "place",
    "good",
    "great",
    "nice",
    "friendly",
    "amazing",
}
_LOW_SIGNAL_TERMS = {
    "arrived",
    "came",
    "especially",
    "loved",
    "perfectly",
    "really",
    "super",
    "tasted",
    "very",
}


def build_cluster_summaries(
    items: list[dict[str, Any]],
    vectors: np.ndarray,
    cluster_labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Build summaries for each cluster."""

    grouped = cluster_to_indices(cluster_labels.tolist())
    all_texts = [str(item.get("text", "")) for item in items]
    terms_by_cluster = _extract_distinctive_terms(all_texts=all_texts, grouped_indices=grouped, top_n=8)

    summaries: list[dict[str, Any]] = []
    for cluster_id, indices in grouped.items():
        cluster_texts = [items[i]["text"] for i in indices]
        top_terms = terms_by_cluster.get(cluster_id, [])
        if not top_terms:
            top_terms = _fallback_terms(cluster_texts, top_n=8)

        rep_ids, representatives = _representatives(items, vectors, indices, limit=5)
        theme_info = dominant_theme(items=items, indices=indices)
        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "size": len(indices),
                "top_terms": top_terms,
                "representatives": representatives,
                "representative_ids": rep_ids,
                "dominant_aspect": str(theme_info.get("dominant_aspect", "general")),
                "dominant_polarity": str(theme_info.get("dominant_polarity", "neutral")),
                "aspect_purity": float(theme_info.get("aspect_purity", 0.0)),
                "polarity_purity": float(theme_info.get("polarity_purity", 0.0)),
                "label": "",
            }
        )

    return summaries


def _extract_distinctive_terms(
    all_texts: list[str],
    grouped_indices: dict[int, list[int]],
    top_n: int = 8,
) -> dict[int, list[str]]:
    """Extract cluster-distinctive terms by contrasting cluster vs corpus usage."""

    if not all_texts or not grouped_indices:
        return {}

    min_df = 2 if len(all_texts) >= 40 else 1
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=2048,
            ngram_range=(1, 2),
            min_df=min_df,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(all_texts)
    except Exception:
        return {}

    if matrix.shape[1] == 0:
        return {}

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    global_scores = np.asarray(matrix.mean(axis=0)).ravel()
    cluster_terms: dict[int, list[str]] = {}

    for cluster_id, indices in grouped_indices.items():
        if not indices:
            cluster_terms[int(cluster_id)] = []
            continue

        local_scores = np.asarray(matrix[indices].mean(axis=0)).ravel()
        distinctive = local_scores - (0.35 * global_scores)
        order = np.argsort(distinctive)[::-1]

        selected: list[str] = []
        for feature_idx in order:
            if local_scores[feature_idx] <= 0:
                continue
            term = str(feature_names[feature_idx]).strip().lower()
            if not term or _is_generic_term(term) or _is_low_signal_term(term):
                continue
            if term not in selected:
                selected.append(term)
            if len(selected) >= top_n:
                break

        if not selected:
            order = np.argsort(local_scores)[::-1]
            for feature_idx in order:
                if local_scores[feature_idx] <= 0:
                    continue
                term = str(feature_names[feature_idx]).strip().lower()
                if not term:
                    continue
                if term not in selected:
                    selected.append(term)
                if len(selected) >= top_n:
                    break

        cluster_terms[int(cluster_id)] = selected[:top_n]

    return cluster_terms


def _is_generic_term(term: str) -> bool:
    """Filter generic terms that frequently produce vague labels."""

    if term in _GENERIC_TERMS:
        return True
    if " " in term:
        tokens = [token for token in term.split() if token]
        if tokens and all(token in _GENERIC_TERMS for token in tokens):
            return True
    return False


def _is_low_signal_term(term: str) -> bool:
    """Filter terms that are too vague to describe a stable theme."""

    tokens = [token for token in term.split() if token]
    if not tokens:
        return True
    if len(tokens) == 1:
        return tokens[0] in _LOW_SIGNAL_TERMS
    return all(token in _LOW_SIGNAL_TERMS for token in tokens)


def _fallback_terms(texts: list[str], top_n: int = 8) -> list[str]:
    """Extract frequent tokens when TF-IDF is unavailable."""

    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(token.lower() for token in _WORD_RE.findall(text.lower()))
    return [term for term, _ in counts.most_common(top_n)]


def _representatives(
    items: list[dict[str, Any]],
    vectors: np.ndarray,
    indices: list[int],
    limit: int = 5,
) -> tuple[list[str], list[str]]:
    """Return item ids/texts closest to the cluster centroid."""

    if not indices:
        return [], []

    subset = vectors[indices]
    centroid = subset.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    subset_norm = np.linalg.norm(subset, axis=1)

    denom = subset_norm * centroid_norm
    denom[denom == 0.0] = 1.0
    similarities = (subset @ centroid) / denom

    local_order = np.argsort(similarities)[::-1][:limit]
    selected_indices = [indices[int(local_idx)] for local_idx in local_order]

    ids = [str(items[idx]["id"]) for idx in selected_indices]
    texts = [str(items[idx]["text"]) for idx in selected_indices]
    return ids, texts
