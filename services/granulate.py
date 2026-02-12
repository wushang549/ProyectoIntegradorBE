# app/services/granulate.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


DEFAULT_ASPECTS: dict[str, list[str]] = {
    "PRODUCT_QUALITY": [
        "quality",
        "build quality",
        "materials",
        "durable",
        "sturdy",
        "flimsy",
        "defect",
        "broken",
        "well made",
        "poor quality",
        "finish",
        "craftsmanship",
    ],
    "FEATURES_FUNCTIONALITY": [
        "feature",
        "features",
        "functionality",
        "capability",
        "capabilities",
        "option",
        "options",
        "settings",
        "works",
        "does not work",
        "doesn't work",
        "missing feature",
        "use case",
        "integration",
        "integrates",
        "compatibility",
        "compatible",
        "incompatible",
    ],
    "USABILITY_UX": [
        "easy to use",
        "hard to use",
        "intuitive",
        "confusing",
        "workflow",
        "navigation",
        "interface",
        "ui",
        "ux",
        "design",
        "layout",
        "looks",
        "look and feel",
        "visual",
        "accessibility",
        "learning curve",
        "discoverability",
        "onboarding",
    ],
    "PERFORMANCE_SPEED": [
        "fast",
        "slow",
        "lag",
        "laggy",
        "latency",
        "performance",
        "responsive",
        "unresponsive",
        "stutter",
        "stuttering",
        "freeze",
        "freezes",
        "load time",
        "loading",
        "speed",
        "optimization",
        "resource usage",
        "memory",
        "cpu",
    ],
    "RELIABILITY_STABILITY": [
        "reliable",
        "unreliable",
        "stability",
        "stable",
        "unstable",
        "crash",
        "crashes",
        "crashing",
        "bug",
        "bugs",
        "issue",
        "issues",
        "glitch",
        "glitches",
        "error",
        "errors",
        "fails",
        "failed",
        "failure",
        "downtime",
        "outage",
        "broken",
        "works consistently",
    ],
    "CUSTOMER_SUPPORT_SERVICE": [
        "support",
        "customer service",
        "agent",
        "representative",
        "helpful",
        "unhelpful",
        "rude",
        "ignored",
        "response time",
        "ticket",
        "refund",
        "replacement",
        "warranty",
        "escalation",
    ],
    "PRICE_VALUE": [
        "price",
        "pricing",
        "expensive",
        "cheap",
        "cost",
        "overpriced",
        "affordable",
        "value",
        "worth it",
        "not worth",
        "plan",
        "roi",
    ],
    "BILLING_PAYMENTS": [
        "billing",
        "charged",
        "invoice",
        "payment",
        "refund",
        "subscription",
        "renewal",
        "cancellation",
        "trial",
        "upgrade",
        "downgrade",
        "credit card",
        "unexpected charge",
    ],
    "SHIPPING_DELIVERY_LOGISTICS": [
        "shipping",
        "delivery",
        "arrived",
        "late",
        "on time",
        "packaging",
        "tracking",
        "courier",
        "damaged in transit",
        "installation",
        "setup appointment",
    ],
    "SECURITY_PRIVACY": [
        "security",
        "privacy",
        "data",
        "permissions",
        "leak",
        "breach",
        "encryption",
        "compliance",
        "gdpr",
        "pii",
        "account security",
        "2fa",
        "mfa",
        "authentication",
        "authorization",
    ],
    "COMMUNICATION": [
        "communication",
        "updates",
        "status",
        "notifications",
        "email",
        "messaging",
        "transparency",
        "follow up",
        "no response",
        "announcement",
        "roadmap",
    ],
    "POLICIES_FAIRNESS": [
        "policy",
        "policies",
        "terms",
        "fair",
        "unfair",
        "ban",
        "moderation",
        "return policy",
        "hidden fees",
        "scam",
        "misleading",
    ],
    "OTHER": [
        "overall",
        "general",
        "experience",
        "comment",
        "feedback",
        "thoughts",
    ],
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "me",
    "my",
    "your",
    "our",
    "their",
    "them",
    "de",
    "la",
    "el",
    "los",
    "las",
    "y",
    "o",
    "pero",
    "para",
    "por",
    "con",
    "sin",
    "en",
    "un",
    "una",
    "unos",
    "unas",
    "es",
    "son",
    "fue",
    "eran",
    "ser",
    "yo",
    "tu",
    "tú",
    "mi",
    "mis",
    "su",
    "sus",
    "nos",
    "nosotros",
    "ustedes",
    "ellos",
    "ellas",
}

TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9']{2,}")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:,|;|\band\b|\bbut\b|\bhowever\b|\bthough\b|\bpero\b|\by\b)\s*",
    re.IGNORECASE,
)

POS_WORDS = {
    "amazing",
    "great",
    "good",
    "awesome",
    "excellent",
    "love",
    "loved",
    "perfect",
    "nice",
    "friendly",
    "helpful",
    "fast",
    "clean",
    "smooth",
    "reliable",
    "intuitive",
    "responsive",
    "excelente",
    "genial",
    "buen",
    "bueno",
    "perfecto",
    "rapido",
    "rápido",
}
NEG_WORDS = {
    "bad",
    "terrible",
    "awful",
    "hate",
    "hated",
    "slow",
    "rude",
    "dirty",
    "expensive",
    "overpriced",
    "broken",
    "defect",
    "crash",
    "crashes",
    "crashing",
    "bug",
    "bugs",
    "lag",
    "laggy",
    "unreliable",
    "scam",
    "misleading",
    "error",
    "errors",
    "fails",
    "failed",
    "failure",
    "freeze",
    "freezes",
    "malo",
    "horrible",
    "terrible",
    "lento",
    "caro",
    "sucio",
    "fallo",
    "falla",
}
NEGATION = {"not", "no", "never", "n't", "sin"}


@dataclass
class Granule:
    aspect: str
    excerpt: str
    evidence: list[str]
    sentiment: str
    sentiment_score: float
    similarity: float


def _tokenize(text: str) -> list[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    return [t for t in toks if t not in STOPWORDS]


def _split_units(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    sentences = [s.strip() for s in SENT_SPLIT_RE.split(raw) if s.strip()]
    units: list[str] = []
    for s in (sentences if sentences else [raw]):
        parts = [p.strip() for p in CLAUSE_SPLIT_RE.split(s) if p.strip()]
        units.extend(parts if parts else [s])
    return units


def _simple_sentiment(tokens: list[str]) -> tuple[str, float]:
    score = 0.0
    for i, t in enumerate(tokens):
        w = t.lower()
        neg = i > 0 and tokens[i - 1].lower() in NEGATION
        if w in POS_WORDS:
            score += -1.0 if neg else 1.0
        elif w in NEG_WORDS:
            score += 1.0 if neg else -1.0

    if score > 0.25:
        return "positive", float(score)
    if score < -0.25:
        return "negative", float(score)
    return "neutral", float(score)


def _normalize_aspects(custom: dict[str, list[str]] | None) -> dict[str, list[str]]:
    if not custom:
        return DEFAULT_ASPECTS

    out: dict[str, list[str]] = {}
    for k, seeds in custom.items():
        key = (k or "").strip().upper()
        if not key:
            continue
        if not isinstance(seeds, list) or not seeds:
            continue
        cleaned = [str(s).strip() for s in seeds if str(s).strip()]
        if cleaned:
            out[key] = cleaned

    if "OTHER" not in out:
        out["OTHER"] = DEFAULT_ASPECTS["OTHER"]

    return out


def _build_vectorizer() -> TfidfVectorizer:
    # Custom stopwords prevents "the", "is", etc. from showing up as evidence.
    # token_pattern aligns with TOKEN_RE.
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        stop_words=list(STOPWORDS),
        token_pattern=r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9']{2,}",
    )


def granulate_text(
    text: str,
    aspects: dict[str, list[str]] | None = None,
    top_k_evidence: int = 6,
    min_similarity: float = 0.04,
) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("text is required")

    units = _split_units(raw)
    if not units:
        raise ValueError("text is required")

    aspects_map = _normalize_aspects(aspects)

    aspect_names: list[str] = []
    prototypes: list[str] = []
    for name, seeds in aspects_map.items():
        aspect_names.append(name)
        prototypes.append(" ".join(seeds))

    corpus = prototypes + units
    vectorizer = _build_vectorizer()
    X = vectorizer.fit_transform(corpus)

    proto_vecs = X[: len(prototypes)]
    unit_vecs = X[len(prototypes) :]

    # Cosine similarity with explicit normalization makes scoring stable.
    proto_vecs = normalize(proto_vecs, norm="l2", axis=1, copy=False)
    unit_vecs = normalize(unit_vecs, norm="l2", axis=1, copy=False)

    feats = np.array(vectorizer.get_feature_names_out())

    granules: list[Granule] = []
    for i, unit in enumerate(units):
        uv = unit_vecs[i]

        sims = (proto_vecs @ uv.T).toarray().ravel()
        best_i = int(np.argmax(sims)) if sims.size else 0
        best_sim = float(sims[best_i]) if sims.size else 0.0

        aspect = aspect_names[best_i] if aspect_names else "OTHER"
        if best_sim < float(min_similarity):
            aspect = "OTHER"

        tokens = _tokenize(unit)
        sent_label, sent_score = _simple_sentiment(tokens)

        # Evidence from unit TF-IDF weights, filtered + de-duplicated.
        weights = uv.toarray().ravel()
        top_idx = np.argsort(weights)[-int(top_k_evidence) * 3 :][::-1]

        evidence: list[str] = []
        seen: set[str] = set()
        for j in top_idx:
            if weights[j] <= 0:
                break
            term = feats[j].strip()
            if not term:
                continue
            if term in seen:
                continue
            # Avoid evidence like single stopword fragments (extra safety).
            if term in STOPWORDS:
                continue
            seen.add(term)
            evidence.append(term)
            if len(evidence) >= int(top_k_evidence):
                break

        if not evidence:
            evidence = tokens[: int(top_k_evidence)]

        granules.append(
            Granule(
                aspect=aspect,
                excerpt=unit,
                evidence=evidence,
                sentiment=sent_label,
                sentiment_score=sent_score,
                similarity=best_sim,
            )
        )

    return {
        "text": raw,
        "units": units,
        "granules": [
            {
                "aspect": g.aspect,
                "excerpt": g.excerpt,
                "evidence": g.evidence,
                "sentiment": g.sentiment,
                "sentiment_score": round(g.sentiment_score, 3),
                "similarity": round(g.similarity, 4),
            }
            for g in granules
        ],
        "taxonomy": list(aspects_map.keys()),
    }
