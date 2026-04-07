"""Grounded chat responses for one completed analysis."""

from __future__ import annotations

import re
from typing import Any

from services.openai_text import request_openai_text, resolve_openai_text_model
from services.thematics import theme_from_item

MAX_HISTORY_MESSAGES = 8
MAX_THEME_COUNT = 5
MAX_THEME_EXAMPLES = 2
MAX_KEY_FINDINGS = 5
MAX_WARNINGS = 3
MAX_BRANCHES = 4
MAX_RETRIEVED_ITEMS = 6

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{1,}")
_ROW_LEVEL_HINTS = {
    "comment",
    "comments",
    "review",
    "reviews",
    "row",
    "rows",
    "quote",
    "quoted",
    "example",
    "examples",
    "exact",
    "specific",
    "worst",
    "best",
    "negative",
    "positive",
    "complaint",
    "complaints",
}
_NEGATIVE_HINTS = {"negative", "worst", "bad", "complaint", "complaints", "angry", "upset"}
_POSITIVE_HINTS = {"positive", "best", "good", "great", "praise", "praised", "happy"}


def generate_analysis_chat_answer(
    *,
    analysis_id: str,
    messages: list[dict[str, str]],
    items_payload: list[dict[str, Any]],
    overview_payload: dict[str, Any],
    insights_payload: dict[str, Any],
    clusters_payload: dict[str, Any],
    map_payload: dict[str, Any],
    hierarchy_payload: dict[str, Any],
    selection: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Build grounded context and return one model answer."""

    cleaned_messages = _normalize_messages(messages)
    if not cleaned_messages:
        raise ValueError("At least one chat message is required.")
    if cleaned_messages[-1]["role"] != "user":
        raise ValueError("The last chat message must be from the user.")

    current_question = cleaned_messages[-1]["content"]
    context_text, sections = build_analysis_chat_context(
        current_question=current_question,
        items_payload=items_payload,
        overview_payload=overview_payload,
        insights_payload=insights_payload,
        clusters_payload=clusters_payload,
        map_payload=map_payload,
        hierarchy_payload=hierarchy_payload,
        selection=selection,
    )

    prior_history = cleaned_messages[:-1][-MAX_HISTORY_MESSAGES:]
    prompt_parts = [
        "Analysis context:",
        context_text,
        "Conversation so far:",
        _format_history(prior_history),
        "Current user question:",
        current_question,
    ]
    prompt = "\n\n".join(part for part in prompt_parts if part.strip())
    resolved_model = resolve_openai_text_model(model_name)
    answer = request_openai_text(
        prompt,
        model_name=resolved_model,
        instructions=(
            "You are an analytical assistant for one dataset analysis. "
            "Answer only with support from the provided analysis context. "
            "Be concise, concrete, and transparent. "
            "If the answer is not supported by the available context, say so clearly. "
            "Do not invent metrics, rows, labels, or findings that are not present."
        ),
        max_output_tokens=420,
        timeout_sec=60,
    )

    return {
        "answer": answer.strip(),
        "model": resolved_model,
        "grounding": {
            "analysis_id": analysis_id,
            "sections": sections,
            "selection_applied": _selection_applied(selection),
        },
    }


def build_analysis_chat_context(
    *,
    current_question: str,
    items_payload: list[dict[str, Any]],
    overview_payload: dict[str, Any],
    insights_payload: dict[str, Any],
    clusters_payload: dict[str, Any],
    map_payload: dict[str, Any],
    hierarchy_payload: dict[str, Any],
    selection: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Build one compact, stable text context for analysis chat."""

    sections: list[tuple[str, str]] = []

    overview_text = _format_overview_section(overview_payload)
    if overview_text:
        sections.append(("overview", overview_text))

    insights_text = _format_insights_section(insights_payload)
    if insights_text:
        sections.append(("insights", insights_text))

    clusters_text = _format_clusters_section(clusters_payload)
    if clusters_text:
        sections.append(("clusters", clusters_text))

    map_text = _format_map_section(map_payload)
    if map_text:
        sections.append(("map", map_text))

    hierarchy_text = _format_hierarchy_section(hierarchy_payload)
    if hierarchy_text:
        sections.append(("hierarchy", hierarchy_text))

    selection_text = _format_selection_section(
        selection=selection,
        clusters_payload=clusters_payload,
        map_payload=map_payload,
        hierarchy_payload=hierarchy_payload,
    )
    if selection_text:
        sections.append(("selection", selection_text))

    raw_examples_text = _format_retrieved_items_section(
        current_question=current_question,
        items_payload=items_payload,
        clusters_payload=clusters_payload,
        selection=selection,
    )
    if raw_examples_text:
        sections.append(("raw_examples", raw_examples_text))

    section_names = [name for name, _content in sections]
    context_text = "\n\n".join(f"{name.upper()}\n{content}" for name, content in sections if content.strip())
    return context_text.strip(), section_names


def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def _format_history(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "No prior conversation."
    lines = []
    for message in messages:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines)


def _format_overview_section(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    top_aspects = payload.get("top_aspects") if isinstance(payload.get("top_aspects"), list) else []
    top_clusters = payload.get("top_clusters") if isinstance(payload.get("top_clusters"), list) else []

    parts = [
        f"Items: {int(counts.get('items') or 0)}",
        f"Clusters: {int(counts.get('clusters') or 0)}",
        f"Aspects: {int(counts.get('aspects') or 0)}",
    ]
    if top_aspects:
        aspects_text = ", ".join(
            f"{str(entry.get('aspect') or '').strip()} ({int(entry.get('count') or 0)})"
            for entry in top_aspects[:MAX_THEME_COUNT]
            if str(entry.get("aspect") or "").strip()
        )
        if aspects_text:
            parts.append(f"Top aspects: {aspects_text}")

    if top_clusters:
        cluster_text = ", ".join(
            f"{str(cluster.get('label') or '').strip()} ({int(cluster.get('size') or 0)})"
            for cluster in top_clusters[:MAX_THEME_COUNT]
            if str(cluster.get("label") or "").strip()
        )
        if cluster_text:
            parts.append(f"Largest themes: {cluster_text}")

    return "\n".join(parts)


def _format_insights_section(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    overall = str(payload.get("overall_summary") or "").strip()
    if overall:
        parts.append(f"Overall summary: {overall}")

    findings = payload.get("key_findings") if isinstance(payload.get("key_findings"), list) else []
    cleaned_findings = [str(item).strip() for item in findings if str(item).strip()]
    if cleaned_findings:
        parts.append("Key findings:")
        parts.extend(f"- {item}" for item in cleaned_findings[:MAX_KEY_FINDINGS])

    warnings = payload.get("quality_warnings") if isinstance(payload.get("quality_warnings"), list) else []
    cleaned_warnings = [str(item).strip() for item in warnings if str(item).strip()]
    if cleaned_warnings:
        parts.append("Quality notes:")
        parts.extend(f"- {item}" for item in cleaned_warnings[:MAX_WARNINGS])

    theme_summary = payload.get("theme_summary") if isinstance(payload.get("theme_summary"), list) else []
    if theme_summary:
        parts.append("Theme summary:")
        for theme in theme_summary[:MAX_THEME_COUNT]:
            label = str(theme.get("label") or "").strip()
            if not label:
                continue
            size = int(theme.get("size") or 0)
            terms = ", ".join(str(term).strip() for term in theme.get("top_terms", [])[:5] if str(term).strip())
            examples = [str(example).strip() for example in theme.get("examples", [])[:MAX_THEME_EXAMPLES] if str(example).strip()]
            line = f"- {label} ({size})"
            if terms:
                line += f" | terms: {terms}"
            if examples:
                line += f" | examples: {' ; '.join(examples)}"
            parts.append(line)

    return "\n".join(parts)


def _format_clusters_section(payload: dict[str, Any]) -> str:
    clusters = payload.get("clusters") if isinstance(payload.get("clusters"), list) else []
    if not clusters:
        return ""

    lines = ["Top clusters:"]
    for cluster in clusters[:MAX_THEME_COUNT]:
        label = str(cluster.get("label") or "").strip()
        if not label:
            continue
        size = int(cluster.get("size") or 0)
        cluster_id = int(cluster.get("cluster_id") or 0)
        terms = ", ".join(str(term).strip() for term in cluster.get("top_terms", [])[:5] if str(term).strip())
        reps = []
        for rep in cluster.get("representatives", [])[:MAX_THEME_EXAMPLES]:
            if not isinstance(rep, dict):
                continue
            preview = str(rep.get("preview") or "").strip()
            if preview:
                reps.append(preview)
        line = f"- Cluster {cluster_id}: {label} ({size})"
        if terms:
            line += f" | terms: {terms}"
        if reps:
            line += f" | examples: {' ; '.join(reps)}"
        lines.append(line)

    return "\n".join(lines)


def _format_map_section(payload: dict[str, Any]) -> str:
    points = payload.get("points") if isinstance(payload.get("points"), list) else []
    clusters = payload.get("clusters") if isinstance(payload.get("clusters"), list) else []
    if not points and not clusters:
        return ""

    xs = [float(point.get("x")) for point in points if isinstance(point, dict) and _is_number(point.get("x"))]
    ys = [float(point.get("y")) for point in points if isinstance(point, dict) and _is_number(point.get("y"))]

    parts = [
        f"Map points: {len(points)}",
        f"Map clusters: {len(clusters)}",
    ]
    if xs and ys:
        parts.append(
            f"Coordinate ranges: x {min(xs):.2f} to {max(xs):.2f}, y {min(ys):.2f} to {max(ys):.2f}"
        )
    return "\n".join(parts)


def _format_hierarchy_section(payload: dict[str, Any]) -> str:
    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), dict) else {}
    root_id = str(payload.get("root_id") or "").strip()
    if not nodes:
        return ""

    root = nodes.get(root_id) if isinstance(nodes.get(root_id), dict) else None
    parts = []
    if root:
        parts.append(
            f"Root theme: {str(root.get('label') or 'Root').strip()} ({int(root.get('descendant_leaf_count') or root.get('size') or 0)} items)"
        )

    branches = []
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        if str(node_id).startswith("leaf_"):
            continue
        if str(node.get("parent_id") or "").strip() != root_id:
            continue
        branches.append(node)

    branches.sort(
        key=lambda entry: int(entry.get("descendant_leaf_count") or entry.get("size") or 0),
        reverse=True,
    )
    if branches:
        parts.append("Main branches:")
        for branch in branches[:MAX_BRANCHES]:
            label = str(branch.get("label") or "Theme").strip()
            count = int(branch.get("descendant_leaf_count") or branch.get("size") or 0)
            summary = str(branch.get("summary") or "").strip()
            line = f"- {label} ({count})"
            if summary:
                line += f" | {summary}"
            parts.append(line)

    return "\n".join(parts)


def _format_selection_section(
    *,
    selection: dict[str, Any] | None,
    clusters_payload: dict[str, Any],
    map_payload: dict[str, Any],
    hierarchy_payload: dict[str, Any],
) -> str:
    if not _selection_applied(selection):
        return ""

    parts = ["Current UI selection:"]

    selected_cluster_id = _coerce_optional_int((selection or {}).get("selected_cluster_id"))
    if selected_cluster_id is not None:
        cluster = next(
            (
                entry
                for entry in clusters_payload.get("clusters", [])
                if isinstance(entry, dict) and _coerce_optional_int(entry.get("cluster_id")) == selected_cluster_id
            ),
            None,
        )
        if isinstance(cluster, dict):
            label = str(cluster.get("label") or f"Cluster {selected_cluster_id}").strip()
            size = int(cluster.get("size") or 0)
            parts.append(f"- Selected cluster: {label} ({size})")

    selected_point_id = str((selection or {}).get("selected_point_id") or "").strip()
    if selected_point_id:
        point = next(
            (
                entry
                for entry in map_payload.get("points", [])
                if isinstance(entry, dict) and str(entry.get("id") or "").strip() == selected_point_id
            ),
            None,
        )
        if isinstance(point, dict):
            preview = str(point.get("preview") or "").strip()
            cluster_label = str(point.get("cluster_label") or "").strip()
            parts.append(f"- Selected point: {selected_point_id} | theme: {cluster_label} | preview: {preview}")

    selected_node_id = str((selection or {}).get("selected_node_id") or "").strip()
    if selected_node_id:
        node = hierarchy_payload.get("nodes", {}).get(selected_node_id)
        if isinstance(node, dict):
            label = str(node.get("label") or selected_node_id).strip()
            count = int(node.get("descendant_leaf_count") or node.get("size") or 0)
            summary = str(node.get("summary") or "").strip()
            line = f"- Selected hierarchy node: {label} ({count})"
            if summary:
                line += f" | {summary}"
            parts.append(line)

    return "\n".join(parts)


def _selection_applied(selection: dict[str, Any] | None) -> bool:
    if not isinstance(selection, dict):
        return False
    return any(
        value not in {None, "", -1}
        for value in (
            selection.get("selected_cluster_id"),
            selection.get("selected_point_id"),
            selection.get("selected_node_id"),
        )
    )


def _format_retrieved_items_section(
    *,
    current_question: str,
    items_payload: list[dict[str, Any]],
    clusters_payload: dict[str, Any],
    selection: dict[str, Any] | None,
) -> str:
    if not _should_include_raw_examples(current_question):
        return ""

    retrieved = _retrieve_relevant_items(
        current_question=current_question,
        items_payload=items_payload,
        clusters_payload=clusters_payload,
        selection=selection,
    )
    if not retrieved:
        return ""

    lines = ["Relevant raw comment snippets:"]
    for item in retrieved:
        item_id = str(item.get("id") or "").strip()
        text = str(item.get("text") or "").strip()
        theme = theme_from_item(item)
        cluster_id = item.get("_cluster_id")
        prefix = f"- {item_id}" if item_id else "- Item"
        if cluster_id is not None:
            prefix += f" | cluster {cluster_id}"
        prefix += f" | polarity={theme.get('polarity', 'unknown')} | aspect={theme.get('aspect', 'general')}"
        lines.append(prefix)
        lines.append(f"  {text}")
    return "\n".join(lines)


def _retrieve_relevant_items(
    *,
    current_question: str,
    items_payload: list[dict[str, Any]],
    clusters_payload: dict[str, Any],
    selection: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    question_tokens = {token.lower() for token in _WORD_RE.findall(current_question)}
    wants_negative = any(token in _NEGATIVE_HINTS for token in question_tokens)
    wants_positive = any(token in _POSITIVE_HINTS for token in question_tokens)
    selected_cluster_id = _coerce_optional_int((selection or {}).get("selected_cluster_id"))
    cluster_by_item_id = {
        str(entry.get("id") or "").strip(): entry.get("cluster_id")
        for entry in clusters_payload.get("item_cluster_map", [])
        if isinstance(entry, dict) and str(entry.get("id") or "").strip()
    }

    scored: list[tuple[float, dict[str, Any]]] = []
    for item in items_payload:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        text = str(item.get("text") or "").strip()
        if not text:
            continue

        cluster_id = _coerce_optional_int(cluster_by_item_id.get(item_id))
        if selected_cluster_id is not None and cluster_id != selected_cluster_id:
            continue

        theme = theme_from_item(item)
        item_tokens = {token.lower() for token in _WORD_RE.findall(text)}
        overlap = len(question_tokens.intersection(item_tokens))

        score = float(overlap)
        polarity = str(theme.get("polarity") or "neutral").lower()
        if wants_negative:
            if polarity == "negative":
                score += 8
            elif polarity == "mixed":
                score += 3
        if wants_positive:
            if polarity == "positive":
                score += 8
            elif polarity == "mixed":
                score += 3
        if any(token in {"most", "worst", "best"} for token in question_tokens):
            confidence = _coerce_float((item.get("metadata") or {}).get("_analysis", {}).get("polarity_confidence"))
            score += confidence * 4
        if selected_cluster_id is not None:
            score += 2
        if score <= 0:
            continue

        enriched = dict(item)
        enriched["_cluster_id"] = cluster_id
        scored.append((score, enriched))

    scored.sort(
        key=lambda pair: (
            pair[0],
            _coerce_float((pair[1].get("metadata") or {}).get("_analysis", {}).get("polarity_confidence")),
            len(str(pair[1].get("text") or "")),
        ),
        reverse=True,
    )
    return [item for _score, item in scored[:MAX_RETRIEVED_ITEMS]]


def _should_include_raw_examples(question: str) -> bool:
    tokens = {token.lower() for token in _WORD_RE.findall(question)}
    return any(token in _ROW_LEVEL_HINTS for token in tokens)


def _coerce_optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
