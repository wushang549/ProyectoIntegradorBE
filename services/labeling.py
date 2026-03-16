"""Cluster labeling with OpenAI-backed naming and deterministic fallback."""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from services.openai_text import OpenAITextError, request_openai_text, resolve_openai_text_model
from services.storage import read_json_file, write_json_file

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
    "just",
}
_LOW_SIGNAL_TOKENS = {
    "amazing",
    "arrived",
    "beautiful",
    "bit",
    "came",
    "checked",
    "gave",
    "genuinely",
    "happy",
    "kind",
    "little",
    "looked",
    "loved",
    "often",
    "ordered",
    "parts",
    "positive",
    "seemed",
    "side",
    "slightly",
    "still",
    "surprisingly",
    "tasted",
    "super",
    "outstanding",
    "perfectly",
    "excellent",
    "delicious",
    "tried",
    "just",
}
_ASPECT_KEYWORDS = {
    "Food": {
        "food",
        "dish",
        "dishes",
        "pizza",
        "pasta",
        "burger",
        "burgers",
        "taco",
        "tacos",
        "sushi",
        "salad",
        "sandwiches",
        "nachos",
        "dessert",
        "desserts",
        "ramen",
        "steak",
        "chicken",
        "falafel",
        "seafood",
        "burritos",
        "burrito",
        "dumplings",
        "dumpling",
        "noodles",
        "noodle",
        "omelets",
        "omelet",
        "pancakes",
        "pancake",
        "curry",
        "ribs",
        "bbq",
        "rice",
        "soup",
        "sandwich",
        "flavor",
        "flavorful",
        "seasoning",
        "seasoned",
        "greasy",
        "salty",
        "bland",
        "lukewarm",
        "overcooked",
        "satisfying",
        "tender",
        "crispy",
        "presentation",
        "presented",
        "toppings",
    },
    "Service": {
        "service",
        "waiter",
        "staff",
        "server",
        "order",
        "seating",
        "friendly",
        "rude",
        "customer",
        "attentive",
        "recommendations",
        "recommendation",
        "polite",
        "help",
        "forgot",
        "disorganized",
        "overwhelmed",
    },
    "Atmosphere": {"atmosphere", "cozy", "noisy", "vibe", "interior", "restaurant", "stylish", "cramped", "lighting", "lively", "warm"},
    "Value": {"price", "prices", "overpriced", "value", "portions", "portion", "money", "tiny", "generous", "worth", "reasonable", "expensive"},
    "Cleanliness": {"dirty", "clean", "smelled", "table", "bathroom", "sticky"},
    "Speed": {"slow", "quick", "quickly", "wait", "arrived", "waited", "forever", "late", "fast"},
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
    "attentive",
    "clean",
    "crispy",
    "flavorful",
    "lively",
    "polite",
    "reasonable",
    "satisfying",
    "seasoned",
    "stylish",
    "tender",
    "worth",
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
    "cramped",
    "disorganized",
    "forgot",
    "greasy",
    "lukewarm",
    "missing",
    "overcooked",
    "overwhelmed",
    "salty",
    "forever",
    "late",
    "wait",
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

_LOGGER = logging.getLogger(__name__)


class LabelingError(RuntimeError):
    """Raised when label generation cannot complete."""

def apply_labels(
    cluster_summaries: list[dict[str, Any]],
    k_clusters: int,
    cache_path: Path,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Assign labels to clusters with cache + fallback strategy."""

    resolved_model_name = resolve_openai_text_model(model_name)
    cache = _load_cache(cache_path)

    for cluster in cluster_summaries:
        cluster_id = int(cluster["cluster_id"])
        cache_key = _cluster_cache_key(
            model_name=resolved_model_name,
            k_clusters=k_clusters,
            cluster_id=cluster_id,
        )

        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            cached_label = str(cached.get("label") or "").strip()
            cached_source = str(cached.get("source") or "").strip() or "unknown"
            if cached_label:
                cluster["label"] = cached_label
                cluster["label_source"] = cached_source
                continue
        if isinstance(cached, str) and cached.strip():
            cluster["label"] = cached
            cluster["label_source"] = "unknown"
            continue

        label, label_source = _generate_label(cluster, model_name=resolved_model_name)
        cache[cache_key] = {"label": label, "source": label_source}
        cluster["label"] = label
        cluster["label_source"] = label_source

    _disambiguate_duplicate_labels(cluster_summaries)
    _save_cache(cache_path, cache)
    return cluster_summaries


def generate_label_from_context(
    top_terms: list[str],
    representatives: list[str],
    fallback: str = "Theme",
    model_name: str | None = None,
) -> str:
    """Generate one short label from arbitrary context snippets."""

    normalized_terms = [str(term) for term in top_terms if str(term).strip()]
    normalized_reps = [str(rep) for rep in representatives if str(rep).strip()]
    cluster_size = max(3, len(normalized_reps))
    resolved_model_name = resolve_openai_text_model(model_name)

    try:
        prompt = _build_prompt(top_terms=normalized_terms, representatives=normalized_reps)
        result = _call_openai_label(prompt, model_name=resolved_model_name)
        cleaned = _sanitize_label(result)
        if (
            cleaned
            and _label_overlaps_context(cleaned, normalized_terms, normalized_reps)
            and not _is_low_information_label(cleaned, cluster_size=cluster_size)
        ):
            return cleaned
    except Exception as exc:
        _LOGGER.warning("OpenAI node label generation failed; using fallback. model=%s error=%s", resolved_model_name, exc)

    heuristic = _heuristic_label(top_terms=normalized_terms, representatives=normalized_reps)
    if heuristic:
        return heuristic

    backed = _fallback_label(top_terms=normalized_terms, representatives=normalized_reps)
    if backed:
        return backed
    return fallback


def _generate_label(cluster: dict[str, Any], model_name: str | None = None) -> tuple[str, str]:
    """Generate one short label for a cluster."""

    top_terms = [str(term) for term in cluster.get("top_terms", [])]
    representatives = [str(entry) for entry in cluster.get("representatives", [])]
    cluster_size = int(cluster.get("size", 0))
    signature_label = _theme_signature_label(cluster)
    contextual_label = _contextual_label(top_terms=top_terms, representatives=representatives, cluster=cluster)
    heuristic_label = _heuristic_label(top_terms=top_terms, representatives=representatives, cluster=cluster)
    resolved_model_name = resolve_openai_text_model(model_name)

    # Tiny clusters are best labeled deterministically from terms/examples.
    if cluster_size <= 2:
        if contextual_label:
            return contextual_label, "contextual"
        return _fallback_label(top_terms=top_terms, representatives=representatives), "fallback"

    prompt = _build_prompt(top_terms=top_terms, representatives=representatives)
    try:
        result = _call_openai_label(prompt, model_name=resolved_model_name)
        cleaned = _sanitize_label(result)
        if (
            cleaned
            and _label_overlaps_context(cleaned, top_terms, representatives)
            and not _is_low_information_label(cleaned, cluster_size=cluster_size)
        ):
            if signature_label and not _label_matches_signature(cleaned, cluster):
                if contextual_label:
                    return contextual_label, "contextual"
                return signature_label, "signature"
            return cleaned, "openai"
    except Exception as exc:
        _LOGGER.warning("OpenAI cluster labeling failed; using fallback. model=%s error=%s", resolved_model_name, exc)

    if contextual_label:
        return contextual_label, "contextual"
    if signature_label:
        return signature_label, "signature"
    if heuristic_label:
        return heuristic_label, "heuristic"

    return _fallback_label(top_terms=top_terms, representatives=representatives), "fallback"


def _build_prompt(top_terms: list[str], representatives: list[str]) -> str:
    """Create the prompt with cluster context."""

    reps = "\n".join(f"- {entry}" for entry in representatives[:5])
    return (
        "You are labeling a text cluster. Return exactly one short label of 1-3 words. "
        "Use nouns/adjectives from provided evidence only. "
        "No punctuation, no explanation, no extra text.\n\n"
        f"Candidate terms: {', '.join(top_terms[:10])}\n"
        f"Representative items:\n{reps}\n\n"
        "Label:"
    )


def _call_openai_label(prompt: str, model_name: str | None = None) -> str:
    """Call OpenAI for one short cluster label."""

    try:
        return request_openai_text(
            prompt,
            model_name=model_name,
            instructions=(
                "You generate short labels for clustered feedback. "
                "Return exactly one label of 1 to 3 words, with no punctuation and no explanation."
            ),
            max_output_tokens=64,
            timeout_sec=30,
        )
    except OpenAITextError as exc:
        raise LabelingError(str(exc)) from exc


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
    if aspect == "general" and size < 12:
        return ""
    if aspect != "general" and aspect_purity < 0.72:
        return ""

    if polarity == "negative" and polarity_purity >= 0.62:
        return f"{aspect_name} Issues"
    if polarity == "positive" and polarity_purity >= 0.62:
        return f"{aspect_name} Highlights"
    if polarity in {"mixed", "neutral"} and size >= 10 and aspect_purity >= 0.8:
        if size >= 10:
            return f"{aspect_name} Feedback"

    return ""


def _label_matches_signature(label: str, cluster: dict[str, Any]) -> bool:
    """Validate whether generated label is consistent with cluster signature."""

    signature = _theme_signature_label(cluster)
    if not signature:
        return True

    label_tokens = {token.lower() for token in _TOKEN_RE.findall(label.lower())}
    signature_tokens = {token.lower() for token in _TOKEN_RE.findall(signature.lower())}
    if label_tokens.intersection(signature_tokens):
        return True

    aspect = str(cluster.get("dominant_aspect", "")).strip().lower()
    if not aspect or aspect == "general":
        return False

    aspect_name = _ASPECT_DISPLAY.get(aspect, "")
    aspect_keywords = {
        token.lower()
        for token in _ASPECT_KEYWORDS.get(aspect_name, set())
        if str(token).strip()
    }
    if aspect_name and aspect_name.lower() in label_tokens:
        return True
    return bool(label_tokens.intersection(aspect_keywords))


def _fallback_label(top_terms: list[str], representatives: list[str]) -> str:
    """Fallback labeling based on distinctive terms and representative snippets."""

    selected: list[str] = []
    for term in top_terms:
        tokens = _meaningful_tokens(term)
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
        meaningful = _meaningful_tokens(text)
        if meaningful:
            return " ".join(token.title() for token in meaningful[:2])

    return "General Feedback"


def _contextual_label(
    top_terms: list[str],
    representatives: list[str],
    cluster: dict[str, Any],
) -> str:
    """Prefer short, aspect-aware labels grounded in cleaned cluster evidence."""

    aspect = str(cluster.get("dominant_aspect", "")).strip().lower()
    polarity = str(cluster.get("dominant_polarity", "")).strip().lower()
    polarity_purity = float(cluster.get("polarity_purity", 0.0) or 0.0)
    if not aspect:
        return ""

    token_counter = Counter(_meaningful_tokens(" ".join(top_terms + representatives)))
    tokens = set(token_counter.keys())
    if not tokens:
        return ""

    if aspect == "service":
        if {"waiter", "recommendations"}.issubset(tokens) or "recommendations" in tokens:
            return "Waiter Recommendations"
        if polarity == "negative" and ({"slow", "service"}.issubset(tokens) or {"wait", "forever"}.intersection(tokens)):
            return "Slow Service"
        if polarity == "negative" and ({"forgot", "dishes"}.issubset(tokens) or {"disorganized", "overwhelmed"}.intersection(tokens)):
            return "Service Issues"
        if polarity != "negative" and polarity_purity >= 0.55 and ({"attentive", "server"}.issubset(tokens) or "attentive" in tokens):
            return "Attentive Service"
        if polarity != "negative" and polarity_purity >= 0.55 and ({"friendly", "staff"}.issubset(tokens) or "friendly" in tokens):
            return "Friendly Service"
        if {"help", "happy", "friendly"}.intersection(tokens):
            return "Staff Helpfulness"

    if aspect == "value":
        if {"worth", "price"}.issubset(tokens) or {"worth", "expensive"}.issubset(tokens):
            return "Worth The Price"
        if polarity != "positive" and ({"small", "portions"}.issubset(tokens) or {"portion", "small"}.issubset(tokens)):
            return "Small Portions"
        if polarity != "positive" and ({"price", "concerns"}.issubset(tokens) or {"price", "prices", "expensive", "higher"}.intersection(tokens)):
            return "Price Concerns"
        if polarity != "negative" and ({"value", "money"}.issubset(tokens) or {"worth", "reasonable", "value", "money"}.intersection(tokens)):
            return "Good Value"

    if aspect == "atmosphere":
        if {"cozy", "atmosphere"}.issubset(tokens) or {"cozy", "lighting", "warm"}.intersection(tokens):
            return "Cozy Atmosphere"
        if "noisy" in tokens:
            return "Noisy Atmosphere"
        if "lively" in tokens:
            return "Lively Atmosphere"
        if "cramped" in tokens:
            return "Cramped Interior"
        if "stylish" in tokens:
            return "Stylish Interior"

    if aspect == "cleanliness":
        if "clean" in tokens:
            return "Clean Dining Area"
        if {"dirty", "sticky", "bathroom"}.intersection(tokens):
            return "Cleanliness Issues"

    if aspect == "speed":
        if polarity != "positive" and {"slow", "wait", "forever"}.intersection(tokens):
            return "Slow Service"
        if polarity != "negative" and {"quick", "fast", "polite"}.intersection(tokens):
            return "Quick Service"

    if aspect == "food":
        if polarity != "positive" and "greasy" in tokens:
            return "Greasy Food"
        if polarity != "positive" and "salty" in tokens:
            return "Salty Food"
        if polarity != "positive" and ("bland" in tokens or ({"missing", "seasoning"}.issubset(tokens))):
            return "Underseasoned Food"
        if polarity != "positive" and "lukewarm" in tokens:
            return "Lukewarm Food"
        if polarity != "positive" and "overcooked" in tokens:
            return "Overcooked Food"
        if polarity != "negative" and "fresh" in tokens:
            return "Fresh Food"
        if polarity != "negative" and "flavorful" in tokens:
            return "Flavorful Food"
        if polarity != "negative" and "seasoned" in tokens:
            return "Well Seasoned"
        if polarity != "negative" and "tender" in tokens:
            return "Tender Food"
        if polarity != "negative" and "satisfying" in tokens:
            return "Satisfying Food"
        if polarity != "negative" and "crispy" in tokens:
            return "Crispy Texture"
        if {"presentation", "presented"}.intersection(tokens):
            return "Food Presentation"

    return ""


def _meaningful_tokens(text: str) -> list[str]:
    """Normalize one evidence string into semantically useful tokens."""

    tokens = [token.lower() for token in _TOKEN_RE.findall(str(text or "").lower())]
    return [
        token
        for token in tokens
        if token not in _GENERIC_TOKENS
        and token not in _LOW_SIGNAL_TOKENS
        and token not in _STOPWORD_TOKENS
    ]


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
        if not _is_signature_style_label(base_label):
            continue
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
        tokens = _meaningful_tokens(str(term))
        for token in tokens:
            if token in base_tokens:
                continue
            return token.title()

    reps = [str(entry) for entry in cluster.get("representatives", [])]
    for text in reps:
        tokens = _meaningful_tokens(text)
        for token in tokens:
            if token in base_tokens:
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


def _load_cache(cache_path: Path) -> dict[str, Any]:
    """Load cluster label cache."""

    if not cache_path.exists():
        return {}
    data = read_json_file(cache_path)
    if isinstance(data, dict):
        normalized: dict[str, Any] = {}
        for key, value in data.items():
            cache_key = str(key)
            if isinstance(value, dict):
                normalized[cache_key] = {
                    "label": str(value.get("label") or "").strip(),
                    "source": str(value.get("source") or "").strip() or "unknown",
                }
            else:
                normalized[cache_key] = str(value)
        return normalized
    return {}


def _save_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    """Persist cluster label cache to disk."""

    write_json_file(cache_path, cache)


def _cluster_cache_key(model_name: str, k_clusters: int, cluster_id: int) -> str:
    """Build one cache key scoped by model and partition size."""

    normalized_model = resolve_openai_text_model(model_name).lower()
    return f"model::{normalized_model}::k{k_clusters}_c{cluster_id}"
