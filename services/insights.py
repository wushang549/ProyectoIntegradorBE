"""High-level insights generated from cluster distribution."""

from __future__ import annotations

import logging
import re
from typing import Any

from services.labeling import LabelingError
from services.openai_text import OpenAITextError, request_openai_text, resolve_openai_text_model

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
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_LOGGER = logging.getLogger(__name__)


def build_insights(
    total_items: int,
    clusters: list[dict[str, Any]],
    embedding_method: str | None = None,
    cluster_quality: dict[str, Any] | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Generate heuristic insights for the insights tab."""

    heuristics = build_insight_heuristics(
        total_items=total_items,
        clusters=clusters,
        embedding_method=embedding_method,
        cluster_quality=cluster_quality,
    )
    overall_summary, summary_source = build_overall_summary(
        total_items=total_items,
        clusters=clusters,
        quality_warnings=heuristics["quality_warnings"],
        llm_model=llm_model,
    )
    heuristics["overall_summary"] = overall_summary
    heuristics["overall_summary_source"] = summary_source
    return heuristics


def build_insight_heuristics(
    total_items: int,
    clusters: list[dict[str, Any]],
    embedding_method: str | None = None,
    cluster_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate non-LLM insights used by the insights tab."""

    if total_items <= 0:
        empty_summary = "No analyzable comments were found."
        return {
            "key_findings": [empty_summary],
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


def build_overall_summary(
    total_items: int,
    clusters: list[dict[str, Any]],
    quality_warnings: list[str] | None = None,
    llm_model: str | None = None,
) -> tuple[str, str]:
    """Build one general summary covering the full analysis."""

    llm_summary = _build_overall_summary_with_llm(
        total_items=total_items,
        clusters=clusters,
        quality_warnings=quality_warnings or [],
        llm_model=llm_model,
    )
    if llm_summary:
        return llm_summary, "llm"

    return build_overall_summary_fallback(
        total_items=total_items,
        clusters=clusters,
        quality_warnings=quality_warnings or [],
    ), "heuristic"


def build_overall_summary_fallback(
    total_items: int,
    clusters: list[dict[str, Any]],
    quality_warnings: list[str] | None = None,
) -> str:
    """Build a deterministic general summary when the LLM is unavailable."""

    if total_items <= 0 or not clusters:
        return "No analyzable comments were found."

    sorted_clusters = sorted(clusters, key=lambda cluster: int(cluster.get("size", 0)), reverse=True)
    top_clusters = sorted_clusters[:3]
    theme_bits: list[str] = []
    for cluster in top_clusters:
        label = str(cluster.get("label") or f"Cluster {cluster.get('cluster_id', 0)}").strip()
        size = int(cluster.get("size", 0))
        pct = (size / total_items) * 100.0 if total_items > 0 else 0.0
        theme_bits.append(f"{label} ({pct:.1f}%)")

    positive_count = 0
    negative_count = 0
    for cluster in sorted_clusters[:6]:
        polarity = str(cluster.get("dominant_polarity") or "").strip().lower()
        size = int(cluster.get("size", 0))
        if polarity == "positive":
            positive_count += size
        elif polarity == "negative":
            negative_count += size

    sentiment_line = ""
    if positive_count > negative_count * 1.25:
        sentiment_line = "Overall sentiment leans positive across the most common themes."
    elif negative_count > positive_count * 1.25:
        sentiment_line = "Overall sentiment leans negative across the most common themes."
    else:
        sentiment_line = "Overall sentiment is mixed across the most common themes."

    warning_line = ""
    normalized_warnings = [str(entry).strip() for entry in quality_warnings or [] if str(entry).strip()]
    if normalized_warnings:
        warning_line = " " + normalized_warnings[0]

    return (
        f"The analysis is mainly driven by {', '.join(theme_bits)}. "
        f"{sentiment_line}{warning_line}"
    ).strip()


def _build_overall_summary_with_llm(
    total_items: int,
    clusters: list[dict[str, Any]],
    quality_warnings: list[str],
    llm_model: str | None,
) -> str:
    """Generate one concise overall summary from clustered evidence."""

    if total_items <= 0 or not clusters or not str(llm_model or "").strip():
        return ""

    prompt = _build_overall_summary_prompt(
        total_items=total_items,
        clusters=clusters,
        quality_warnings=quality_warnings,
    )
    try:
        raw = _call_openai_summary(prompt, model_name=resolve_openai_text_model(llm_model))
    except LabelingError as exc:
        _LOGGER.warning("OpenAI summary generation failed; using heuristic summary. model=%s error=%s", llm_model, exc)
        return ""
    except Exception:
        return ""

    cleaned = _sanitize_summary(raw)
    if len(cleaned.split()) < 18:
        return ""
    return cleaned


def _build_overall_summary_prompt(
    total_items: int,
    clusters: list[dict[str, Any]],
    quality_warnings: list[str],
) -> str:
    """Build a compact prompt grounded in top cluster evidence."""

    lines = [
        "You are summarizing clustered customer feedback.",
        "Write one plain-text executive summary of 90 to 140 words.",
        "No bullets. No headings. No speculation. No reasoning trace.",
        "Mention the main positive patterns, the main negative patterns, and the overall balance.",
        f"Total comments: {total_items}",
        "Themes:",
    ]

    for index, cluster in enumerate(clusters[:6], start=1):
        label = str(cluster.get("label") or f"Cluster {cluster.get('cluster_id', index)}").strip()
        size = int(cluster.get("size", 0))
        share = (size / total_items) * 100.0 if total_items > 0 else 0.0
        top_terms = ", ".join(str(term).strip() for term in cluster.get("top_terms", [])[:5] if str(term).strip())
        aspect = str(cluster.get("dominant_aspect") or "").strip()
        polarity = str(cluster.get("dominant_polarity") or "").strip()
        examples = _format_cluster_examples(cluster.get("representatives", []))

        lines.append(f"{index}. {label} | {size} comments | {share:.1f}%")
        if aspect or polarity:
            lines.append(f"   aspect={aspect or 'unknown'}; polarity={polarity or 'unknown'}")
        if top_terms:
            lines.append(f"   top_terms={top_terms}")
        if examples:
            lines.append(f"   examples={examples}")

    if quality_warnings:
        lines.append("Caveats:")
        for warning in quality_warnings[:2]:
            normalized = str(warning).strip()
            if normalized:
                lines.append(f"- {normalized}")

    lines.append("Executive summary:")
    return "\n".join(lines)


def _format_cluster_examples(raw_examples: Any) -> str:
    """Normalize representative examples for prompts."""

    examples: list[str] = []
    if isinstance(raw_examples, list):
        for entry in raw_examples[:2]:
            if isinstance(entry, dict):
                value = str(entry.get("preview") or entry.get("text") or "").strip()
            else:
                value = str(entry or "").strip()
            if not value:
                continue
            examples.append(_clip_text(value, limit=110))
    return " | ".join(examples)


def _clip_text(value: str, limit: int = 160) -> str:
    """Trim prompt text without cutting words too aggressively."""

    cleaned = " ".join(str(value or "").split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rsplit(" ", 1)[0].rstrip(" ,.;:") + "..."


def _sanitize_summary(value: str) -> str:
    """Normalize LLM output into one display-ready paragraph."""

    cleaned = _THINK_BLOCK_RE.sub(" ", str(value or ""))
    cleaned = cleaned.replace("\r", "\n")
    lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\u2022]+\s*", "", line)
        line = re.sub(r"^(summary|executive summary)\s*:\s*", "", line, flags=re.IGNORECASE)
        if line:
            lines.append(line)

    collapsed = " ".join(lines)
    collapsed = re.sub(r"\s+", " ", collapsed).strip()
    return collapsed


def _call_openai_summary(prompt: str, model_name: str) -> str:
    """Call OpenAI for one longer-form summary."""

    try:
        return request_openai_text(
            prompt,
            model_name=model_name,
            instructions=(
                "You summarize clustered customer feedback for a product analytics interface. "
                "Return a single plain-text executive summary with no bullets, no heading, and no reasoning trace."
            ),
            max_output_tokens=220,
            timeout_sec=60,
        )
    except OpenAITextError as exc:
        raise LabelingError(str(exc)) from exc
