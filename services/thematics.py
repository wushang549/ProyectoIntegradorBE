"""Heuristic aspect/polarity classification for text items and clusters."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{1,}")

ASPECTS = ("food", "service", "value", "atmosphere", "cleanliness", "speed", "general")
POLARITIES = ("positive", "negative", "mixed", "neutral")

_ASPECT_PRIORITY = ("food", "speed", "service", "value", "cleanliness", "atmosphere", "general")
_ASPECT_KEYWORDS = {
    "food": {
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
        "sandwich",
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
        "flavor",
        "flavorful",
        "toppings",
        "seasoning",
        "seasoned",
        "fresh",
        "greasy",
        "salty",
        "bland",
        "lukewarm",
        "overcooked",
        "tender",
        "crispy",
        "satisfying",
        "presented",
        "presentation",
        "ingredients",
        "menu",
    },
    "service": {
        "service",
        "waiter",
        "waitress",
        "staff",
        "order",
        "orders",
        "seating",
        "friendly",
        "rude",
        "customer",
        "host",
        "server",
        "attentive",
        "recommendation",
        "recommendations",
        "checked",
        "help",
        "helpful",
        "water",
        "forgot",
        "disorganized",
        "overwhelmed",
    },
    "value": {
        "price",
        "prices",
        "overpriced",
        "value",
        "money",
        "cost",
        "cheap",
        "expensive",
        "worth",
        "reasonable",
        "portion",
        "portions",
    },
    "atmosphere": {
        "atmosphere",
        "cozy",
        "noisy",
        "vibe",
        "interior",
        "ambience",
        "music",
        "environment",
        "stylish",
        "cramped",
        "lighting",
        "lively",
        "warm",
    },
    "cleanliness": {"dirty", "clean", "smelled", "smell", "table", "bathroom", "sticky"},
    "speed": {"slow", "quick", "quickly", "wait", "waited", "waiting", "arrived", "late", "fast", "forever"},
}

_POSITIVE_WORDS = {
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
    "helpful",
    "juicy",
    "fluffy",
    "tasty",
    "incredible",
    "crispy",
    "generous",
    "best",
    "clean",
    "lively",
    "flavorful",
    "seasoned",
    "satisfying",
    "tender",
    "polite",
    "attentive",
    "reasonable",
    "worth",
    "stylish",
}
_NEGATIVE_WORDS = {
    "slow",
    "cold",
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
    "tiny",
    "forgot",
    "annoyed",
    "forever",
    "fell",
    "apart",
    "late",
    "messy",
    "disappointing",
    "lacked",
    "greasy",
    "salty",
    "lukewarm",
    "overcooked",
    "cramped",
    "overwhelmed",
    "disorganized",
    "missing",
    "wait",
}
_NEGATION_TOKENS = {"not", "never", "no", "hardly", "barely"}


def classify_text_theme(text: str) -> dict[str, Any]:
    """Infer lightweight aspect and polarity from text."""

    tokens = _tokens(text)
    if not tokens:
        return {
            "aspect": "general",
            "polarity": "neutral",
            "aspect_confidence": 0.0,
            "polarity_confidence": 0.0,
        }

    aspect_scores = _aspect_scores(tokens)
    aspect = _pick_aspect(aspect_scores)
    aspect_best = int(aspect_scores.get(aspect, 0))
    aspect_total = sum(int(value) for value in aspect_scores.values())
    aspect_conf = float(aspect_best / max(1, aspect_total)) if aspect != "general" else 0.0

    pos_score, neg_score = _polarity_scores(tokens)
    polarity = _pick_polarity(pos_score, neg_score)
    polarity_conf = float(abs(pos_score - neg_score) / max(1, pos_score + neg_score))
    if polarity in {"neutral", "mixed"}:
        polarity_conf = min(0.55, polarity_conf)

    return {
        "aspect": aspect,
        "polarity": polarity,
        "aspect_confidence": round(aspect_conf, 4),
        "polarity_confidence": round(polarity_conf, 4),
    }


def theme_from_item(item: dict[str, Any]) -> dict[str, str]:
    """Read theme tags from item metadata or classify from item text."""

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        analysis_meta = metadata.get("_analysis")
        if isinstance(analysis_meta, dict):
            aspect = _normalize_aspect(analysis_meta.get("aspect"))
            polarity = _normalize_polarity(analysis_meta.get("polarity"))
            if aspect and polarity:
                return {"aspect": aspect, "polarity": polarity}

    detected = classify_text_theme(str(item.get("text", "")))
    return {
        "aspect": str(detected.get("aspect") or "general"),
        "polarity": str(detected.get("polarity") or "neutral"),
    }


def attach_theme_metadata(metadata: dict[str, Any], text: str) -> dict[str, Any]:
    """Return metadata merged with inferred _analysis tags."""

    safe_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    analysis_meta = dict(safe_metadata.get("_analysis", {}))
    inferred = classify_text_theme(text)
    analysis_meta["aspect"] = inferred["aspect"]
    analysis_meta["polarity"] = inferred["polarity"]
    analysis_meta["aspect_confidence"] = inferred["aspect_confidence"]
    analysis_meta["polarity_confidence"] = inferred["polarity_confidence"]
    safe_metadata["_analysis"] = analysis_meta
    return safe_metadata


def bucket_key(aspect: str, polarity: str) -> str:
    """Build normalized clustering bucket key from aspect and polarity."""

    normalized_aspect = _normalize_aspect(aspect) or "general"
    normalized_polarity = _normalize_polarity(polarity) or "neutral"
    return f"{normalized_aspect}:{normalized_polarity}"


def dominant_theme(items: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    """Compute dominant aspect/polarity and purity for one set of item indices."""

    if not indices:
        return {
            "dominant_aspect": "general",
            "dominant_polarity": "neutral",
            "aspect_purity": 0.0,
            "polarity_purity": 0.0,
        }

    aspect_counter: Counter[str] = Counter()
    polarity_counter: Counter[str] = Counter()
    for idx in indices:
        if idx < 0 or idx >= len(items):
            continue
        detected = theme_from_item(items[idx])
        aspect_counter[str(detected["aspect"])] += 1
        polarity_counter[str(detected["polarity"])] += 1

    total = max(1, sum(aspect_counter.values()))
    dominant_aspect, aspect_count = _counter_mode(aspect_counter, default="general")
    dominant_polarity, polarity_count = _counter_mode(polarity_counter, default="neutral")
    return {
        "dominant_aspect": dominant_aspect,
        "dominant_polarity": dominant_polarity,
        "aspect_purity": round(float(aspect_count / total), 4),
        "polarity_purity": round(float(polarity_count / total), 4),
    }


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(str(text or "").lower())]


def _aspect_scores(tokens: list[str]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for aspect, keywords in _ASPECT_KEYWORDS.items():
        scores[aspect] = sum(1 for token in tokens if token in keywords)
    return scores


def _pick_aspect(scores: dict[str, int]) -> str:
    if not scores:
        return "general"

    best_score = max(int(value) for value in scores.values())
    if best_score <= 0:
        return "general"

    tied = {aspect for aspect, value in scores.items() if int(value) == best_score}
    for aspect in _ASPECT_PRIORITY:
        if aspect in tied:
            return aspect
    return "general"


def _polarity_scores(tokens: list[str]) -> tuple[int, int]:
    if not tokens:
        return 0, 0

    pos = 0
    neg = 0
    for idx, token in enumerate(tokens):
        prev = tokens[idx - 1] if idx > 0 else ""
        inverted = prev in _NEGATION_TOKENS
        if token in _POSITIVE_WORDS:
            if inverted:
                neg += 1
            else:
                pos += 1
            continue
        if token in _NEGATIVE_WORDS:
            if inverted:
                pos += 1
            else:
                neg += 1
    return pos, neg


def _pick_polarity(pos_score: int, neg_score: int) -> str:
    if pos_score <= 0 and neg_score <= 0:
        return "neutral"
    if pos_score > 0 and neg_score > 0:
        if abs(pos_score - neg_score) <= 1:
            return "mixed"
        return "positive" if pos_score > neg_score else "negative"
    if pos_score > 0:
        return "positive"
    return "negative"


def _normalize_aspect(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in ASPECTS else None


def _normalize_polarity(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in POLARITIES else None


def _counter_mode(counter: Counter[str], default: str) -> tuple[str, int]:
    if not counter:
        return default, 0
    max_count = max(counter.values())
    tied = [key for key, value in counter.items() if value == max_count]
    if len(tied) == 1:
        return str(tied[0]), int(max_count)
    if default in tied:
        return default, int(max_count)
    return sorted(tied)[0], int(max_count)
