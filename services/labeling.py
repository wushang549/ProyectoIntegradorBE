"""Cluster labeling with local Ollama and deterministic fallback."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from services.storage import read_json_file, write_json_file

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{1,}")
_GENERIC_TOKENS = {"food", "service", "restaurant", "place", "staff", "good", "great", "nice"}
_STOPWORD_TOKENS = {
    "the",
    "our",
    "your",
    "their",
    "this",
    "that",
    "these",
    "those",
    "was",
    "were",
    "is",
    "are",
    "and",
    "but",
    "with",
    "for",
    "from",
    "too",
    "very",
}
_LOW_SIGNAL_TOKENS = {
    "amazing",
    "arrived",
    "beautiful",
    "came",
    "fresh",
    "loved",
    "tasted",
    "super",
    "outstanding",
    "perfectly",
    "excellent",
    "delicious",
}
_ASPECT_KEYWORDS = {
    "Food": {
        "food",
        "dish",
        "pizza",
        "pasta",
        "burger",
        "tacos",
        "sushi",
        "salad",
        "dessert",
        "desserts",
        "ramen",
        "steak",
        "chicken",
        "rice",
        "soup",
        "sandwich",
        "flavor",
        "toppings",
    },
    "Service": {"service", "waiter", "staff", "order", "seating", "friendly", "rude", "customer"},
    "Atmosphere": {"atmosphere", "cozy", "noisy", "vibe", "interior", "restaurant"},
    "Value": {"price", "overpriced", "value", "portions", "money", "tiny", "generous"},
    "Cleanliness": {"dirty", "clean", "smelled", "table"},
    "Speed": {"slow", "quick", "quickly", "wait", "arrived"},
}
_POSITIVE_TOKENS = {
    "great",
    "excellent",
    "amazing",
    "friendly",
    "delicious",
    "fresh",
    "perfect",
    "outstanding",
    "cozy",
    "wonderful",
    "authentic",
}
_NEGATIVE_TOKENS = {
    "slow",
    "cold",
    "bad",
    "dirty",
    "rude",
    "overpriced",
    "bland",
    "stale",
    "undercooked",
    "dry",
    "soggy",
    "mushy",
    "noisy",
    "weird",
    "mediocre",
}
_ASPECT_DISPLAY = {
    "food": "Food",
    "service": "Service",
    "value": "Value",
    "atmosphere": "Atmosphere",
    "cleanliness": "Cleanliness",
    "speed": "Speed",
    "general": "General",
}


class LabelingError(RuntimeError):
    """Raised when label generation cannot complete."""


def apply_labels(
    cluster_summaries: list[dict[str, Any]],
    k_clusters: int,
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Assign labels to clusters with cache + fallback strategy."""

    cache = _load_cache(cache_path)

    for cluster in cluster_summaries:
        cluster_id = int(cluster["cluster_id"])
        cache_key = f"k{k_clusters}_c{cluster_id}"

        cached = cache.get(cache_key)
        if isinstance(cached, str) and cached.strip():
            cluster["label"] = cached
            continue

        label = _generate_label(cluster)
        cache[cache_key] = label
        cluster["label"] = label

    _disambiguate_duplicate_labels(cluster_summaries)
    _save_cache(cache_path, cache)
    return cluster_summaries


def generate_label_from_context(
    top_terms: list[str],
    representatives: list[str],
    fallback: str = "Theme",
) -> str:
    """Generate one short label from arbitrary context snippets."""

    normalized_terms = [str(term) for term in top_terms if str(term).strip()]
    normalized_reps = [str(rep) for rep in representatives if str(rep).strip()]
    cluster_size = max(3, len(normalized_reps))

    try:
        prompt = _build_prompt(top_terms=normalized_terms, representatives=normalized_reps)
        result = _call_ollama(prompt)
        cleaned = _sanitize_label(result)
        if (
            cleaned
            and _label_overlaps_context(cleaned, normalized_terms, normalized_reps)
            and not _is_low_information_label(cleaned, cluster_size=cluster_size)
        ):
            return cleaned
    except Exception:
        pass

    heuristic = _heuristic_label(top_terms=normalized_terms, representatives=normalized_reps)
    if heuristic:
        return heuristic

    backed = _fallback_label(top_terms=normalized_terms, representatives=normalized_reps)
    if backed:
        return backed
    return fallback


def _generate_label(cluster: dict[str, Any]) -> str:
    """Generate one short label for a cluster."""

    top_terms = [str(term) for term in cluster.get("top_terms", [])]
    representatives = [str(entry) for entry in cluster.get("representatives", [])]
    cluster_size = int(cluster.get("size", 0))
    signature_label = _theme_signature_label(cluster)
    heuristic_label = _heuristic_label(top_terms=top_terms, representatives=representatives, cluster=cluster)

    if signature_label and cluster_size >= 6:
        return signature_label

    # Tiny clusters are best labeled deterministically from terms/examples.
    if cluster_size <= 2:
        return _fallback_label(top_terms=top_terms, representatives=representatives)

    prompt = _build_prompt(top_terms=top_terms, representatives=representatives)
    try:
        result = _call_ollama(prompt)
        cleaned = _sanitize_label(result)
        if (
            cleaned
            and _label_overlaps_context(cleaned, top_terms, representatives)
            and not _is_low_information_label(cleaned, cluster_size=cluster_size)
        ):
            if signature_label and not _label_matches_signature(cleaned, cluster):
                return signature_label
            return cleaned
    except Exception:
        pass

    if signature_label:
        return signature_label
    if heuristic_label:
        return heuristic_label

    return _fallback_label(top_terms=top_terms, representatives=representatives)


def _build_prompt(top_terms: list[str], representatives: list[str]) -> str:
    """Create the Ollama prompt with cluster context."""

    reps = "\n".join(f"- {entry}" for entry in representatives[:5])
    return (
        "You are labeling a text cluster. Return exactly one short label of 1-3 words. "
        "Use nouns/adjectives from provided evidence only. "
        "No punctuation, no explanation, no extra text.\n\n"
        f"Candidate terms: {', '.join(top_terms[:10])}\n"
        f"Representative items:\n{reps}\n\n"
        "Label:"
    )


def _call_ollama(prompt: str) -> str:
    """Call local Ollama /api/generate endpoint."""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
            "seed": 42,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise LabelingError("Ollama service is unavailable.") from exc

    parsed = json.loads(raw)
    result = parsed.get("response", "")
    if not isinstance(result, str):
        raise LabelingError("Ollama response is invalid.")
    return result


def _sanitize_label(label: str) -> str:
    """Normalize LLM output to a strict 1-3 word label."""

    first_line = label.strip().splitlines()[0] if label.strip() else ""
    cleaned = re.sub(r"[^a-zA-Z0-9\s\-]", "", first_line).strip()
    if not cleaned:
        return ""

    words = cleaned.split()
    if not words:
        return ""
    words = words[:3]

    normalized = " ".join(words)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.title()


def _label_overlaps_context(label: str, top_terms: list[str], representatives: list[str]) -> bool:
    """Validate that label tokens are grounded in cluster evidence."""

    label_tokens = {token.lower() for token in _TOKEN_RE.findall(label.lower())}
    if not label_tokens:
        return False

    context_blob = " ".join(top_terms + representatives).lower()
    context_tokens = {token.lower() for token in _TOKEN_RE.findall(context_blob)}
    return bool(label_tokens.intersection(context_tokens))


def _is_low_information_label(label: str, cluster_size: int) -> bool:
    """Reject labels that are too generic for medium/large clusters."""

    tokens = [token.lower() for token in _TOKEN_RE.findall(label.lower())]
    if not tokens:
        return True
    if len(tokens) >= 2:
        return False

    token = tokens[0]
    if cluster_size >= 6:
        return True
    return token in _GENERIC_TOKENS or token in _LOW_SIGNAL_TOKENS


def _heuristic_label(
    top_terms: list[str],
    representatives: list[str],
    cluster: dict[str, Any] | None = None,
) -> str:
    """Build deterministic aspect/sentiment labels when LLM output is weak."""

    if cluster:
        by_signature = _theme_signature_label(cluster)
        if by_signature:
            return by_signature

    tokens = [token.lower() for token in _TOKEN_RE.findall(" ".join(top_terms + representatives).lower())]
    if not tokens:
        return ""

    aspect_scores: dict[str, int] = {}
    for aspect, keywords in _ASPECT_KEYWORDS.items():
        aspect_scores[aspect] = sum(1 for token in tokens if token in keywords)

    best_aspect = max(aspect_scores.items(), key=lambda pair: pair[1])
    if best_aspect[1] <= 0:
        return ""

    pos = sum(1 for token in tokens if token in _POSITIVE_TOKENS)
    neg = sum(1 for token in tokens if token in _NEGATIVE_TOKENS)
    if neg >= max(2, pos + 1):
        tone = "Issues"
    elif pos >= max(2, neg + 1):
        tone = "Highlights"
    else:
        tone = "Feedback"

    return f"{best_aspect[0]} {tone}"


def _theme_signature_label(cluster: dict[str, Any]) -> str:
    """Build aspect/polarity-aware label when cluster purity is high enough."""

    aspect = str(cluster.get("dominant_aspect", "")).strip().lower()
    polarity = str(cluster.get("dominant_polarity", "")).strip().lower()
    size = int(cluster.get("size", 0))
    aspect_purity = float(cluster.get("aspect_purity", 0.0) or 0.0)
    polarity_purity = float(cluster.get("polarity_purity", 0.0) or 0.0)

    aspect_name = _ASPECT_DISPLAY.get(aspect)
    if not aspect_name:
        return ""

    # Avoid generic labels when cluster signature is weak.
    if aspect == "general" and size < 10:
        return ""
    if aspect != "general" and aspect_purity < 0.6:
        return ""

    if polarity == "negative" and polarity_purity >= 0.52:
        return f"{aspect_name} Issues"
    if polarity == "positive" and polarity_purity >= 0.52:
        return f"{aspect_name} Highlights"
    if polarity in {"mixed", "neutral"}:
        if size >= 8:
            return f"{aspect_name} Feedback"
        return f"{aspect_name} Themes"

    return f"{aspect_name} Themes"


def _label_matches_signature(label: str, cluster: dict[str, Any]) -> bool:
    """Validate whether generated label is consistent with cluster signature."""

    signature = _theme_signature_label(cluster)
    if not signature:
        return True

    label_tokens = {token.lower() for token in _TOKEN_RE.findall(label.lower())}
    signature_tokens = {token.lower() for token in _TOKEN_RE.findall(signature.lower())}
    return bool(label_tokens.intersection(signature_tokens))


def _fallback_label(top_terms: list[str], representatives: list[str]) -> str:
    """Fallback labeling based on distinctive terms and representative snippets."""

    selected: list[str] = []
    for term in top_terms:
        tokens = [token.lower() for token in _TOKEN_RE.findall(term.lower())]
        if not tokens:
            continue
        if len(tokens) == 1 and (
            tokens[0] in _GENERIC_TOKENS
            or tokens[0] in _LOW_SIGNAL_TOKENS
            or tokens[0] in _STOPWORD_TOKENS
        ):
            continue
        tokens = [
            token
            for token in tokens
            if token not in _LOW_SIGNAL_TOKENS and token not in _STOPWORD_TOKENS
        ]
        if not tokens:
            continue
        selected.append(" ".join(tokens[:2]))
        if len(selected) >= 2:
            break

    if selected:
        label = " ".join(token.title() for token in selected[0].split()[:3]).strip()
        if label:
            return label

    for text in representatives:
        tokens = [token.lower() for token in _TOKEN_RE.findall(text.lower())]
        meaningful = [
            token
            for token in tokens
            if token not in _GENERIC_TOKENS
            and token not in _LOW_SIGNAL_TOKENS
            and token not in _STOPWORD_TOKENS
        ]
        if meaningful:
            return " ".join(token.title() for token in meaningful[:2])

    return "General Feedback"


def _disambiguate_duplicate_labels(clusters: list[dict[str, Any]]) -> None:
    """Disambiguate duplicate labels while preserving canonical aspect labels."""

    buckets: dict[str, list[dict[str, Any]]] = {}
    for cluster in clusters:
        label = str(cluster.get("label", "")).strip()
        if not label:
            continue
        buckets.setdefault(label.lower(), []).append(cluster)

    for _, group in buckets.items():
        if len(group) <= 1:
            continue

        ordered = sorted(group, key=lambda item: int(item.get("size", 0)), reverse=True)
        base_label = str(ordered[0].get("label", "")).strip()
        used_labels = {base_label.lower()}

        for cluster in ordered[1:]:
            qualifier = _cluster_qualifier(cluster, base_label=base_label)
            if not qualifier:
                qualifier = f"Cluster {int(cluster.get('cluster_id', 0))}"
            candidate = _build_disambiguated_label(base_label=base_label, qualifier=qualifier, used=used_labels)
            if candidate:
                cluster["label"] = candidate
                used_labels.add(candidate.lower())


def _is_signature_style_label(label: str) -> bool:
    """Keep canonical aspect labels stable even if multiple clusters share them."""

    normalized = str(label or "").strip()
    if not normalized:
        return False
    suffixes = ("Issues", "Highlights", "Feedback", "Themes")
    if not any(normalized.endswith(suffix) for suffix in suffixes):
        return False
    tokens = {token.lower() for token in _TOKEN_RE.findall(normalized.lower())}
    aspect_tokens = {value.lower() for value in _ASPECT_DISPLAY.values() if value}
    return bool(tokens.intersection(aspect_tokens))


def _cluster_qualifier(cluster: dict[str, Any], base_label: str) -> str:
    """Extract one short distinctive qualifier from cluster evidence."""

    base_tokens = {token.lower() for token in _TOKEN_RE.findall(base_label.lower())}
    for term in cluster.get("top_terms", []):
        tokens = [token.lower() for token in _TOKEN_RE.findall(str(term).lower())]
        for token in tokens:
            if token in base_tokens or token in _GENERIC_TOKENS or token in _LOW_SIGNAL_TOKENS or token in _STOPWORD_TOKENS:
                continue
            return token.title()

    reps = [str(entry) for entry in cluster.get("representatives", [])]
    for text in reps:
        tokens = [token.lower() for token in _TOKEN_RE.findall(text.lower())]
        for token in tokens:
            if token in base_tokens or token in _GENERIC_TOKENS or token in _LOW_SIGNAL_TOKENS or token in _STOPWORD_TOKENS:
                continue
            return token.title()
    return ""


def _build_disambiguated_label(base_label: str, qualifier: str, used: set[str]) -> str:
    """Compose a stable, unique disambiguated label."""

    base = " ".join(token for token in _TOKEN_RE.findall(str(base_label))[:3]).strip()
    qual = " ".join(token for token in _TOKEN_RE.findall(str(qualifier))[:2]).strip()
    if not base or not qual:
        return ""

    candidate = f"{base} {qual}".strip()
    if candidate.lower() not in used:
        return candidate

    suffix = 2
    while True:
        fallback = f"{candidate} {suffix}"
        if fallback.lower() not in used:
            return fallback
        suffix += 1


def _load_cache(cache_path: Path) -> dict[str, str]:
    """Load cluster label cache."""

    if not cache_path.exists():
        return {}
    data = read_json_file(cache_path)
    if isinstance(data, dict):
        return {str(key): str(value) for key, value in data.items()}
    return {}


def _save_cache(cache_path: Path, cache: dict[str, str]) -> None:
    """Persist cluster label cache to disk."""

    write_json_file(cache_path, cache)
