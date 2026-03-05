"""High-level insights generated from cluster distribution."""

from __future__ import annotations

from typing import Any

_WARNING_TERMS = {
    "slow",
    "cold",
    "bad",
    "poor",
    "dirty",
    "rude",
    "expensive",
    "wait",
    "late",
    "burnt",
    "raw",
    "complaint",
    "delay",
}


def build_insights(
    total_items: int,
    clusters: list[dict[str, Any]],
    embedding_method: str | None = None,
    cluster_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate heuristic insights for the insights tab."""

    if total_items <= 0:
        return {
            "key_findings": ["No analyzable comments were found."],
            "theme_summary": "No themes available.",
            "quality_warnings": ["Input dataset is empty after normalization."],
        }

    key_findings: list[str] = []
    quality_warnings: list[str] = []

    sorted_clusters = sorted(clusters, key=lambda cluster: cluster["size"], reverse=True)
    for cluster in sorted_clusters[:5]:
        size = int(cluster["size"])
        pct = (size / total_items) * 100.0
        label = cluster.get("label") or f"Cluster {cluster['cluster_id']}"
        key_findings.append(f"{label} appears in {pct:.1f}% of comments.")

    warning_share = 0.0
    for cluster in clusters:
        text_blob = " ".join(
            [
                str(cluster.get("label", "")).lower(),
                " ".join(str(term).lower() for term in cluster.get("top_terms", [])),
            ]
        )
        if any(term in text_blob for term in _WARNING_TERMS):
            warning_share += int(cluster["size"]) / total_items

    if warning_share > 0.0:
        quality_warnings.append(
            f"Potential quality issues appear in {warning_share * 100.0:.1f}% of comments."
        )
    else:
        quality_warnings.append("No major quality-risk pattern was detected in the current clusters.")

    if embedding_method and "fallback" in embedding_method.lower():
        quality_warnings.append(
            "Semantic embedding model was unavailable, so lexical fallback embeddings were used."
        )

    if cluster_quality:
        largest_share = float(cluster_quality.get("largest_cluster_share", 0.0))
        singleton_share = float(cluster_quality.get("singleton_share", 0.0))
        silhouette = float(cluster_quality.get("silhouette", -1.0))
        if largest_share >= 0.55:
            quality_warnings.append(
                f"One cluster dominates {largest_share * 100.0:.1f}% of items; themes may be overly broad."
            )
        if singleton_share >= 0.12:
            quality_warnings.append(
                f"Singleton clusters represent {singleton_share * 100.0:.1f}% of items; granularity may be too high."
            )
        if silhouette >= -1.0 and silhouette < 0.02:
            quality_warnings.append(
                "Cluster separation is weak for this dataset; theme boundaries should be interpreted carefully."
            )

    purity_values = [
        float(cluster.get("aspect_purity", 0.0))
        for cluster in clusters
        if cluster.get("aspect_purity") is not None
    ]
    if purity_values:
        avg_purity = sum(purity_values) / max(1, len(purity_values))
        if avg_purity < 0.62:
            quality_warnings.append(
                f"Average theme coherence is low ({avg_purity * 100.0:.1f}% aspect purity); some items may still be mixed."
            )

    top_labels = [
        str(cluster.get("label") or f"Cluster {cluster['cluster_id']}")
        for cluster in sorted_clusters[:3]
    ]
    theme_summary = "Top themes: " + ", ".join(top_labels) + "."

    return {
        "key_findings": key_findings,
        "theme_summary": theme_summary,
        "quality_warnings": quality_warnings,
    }
