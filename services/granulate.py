from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# =========================
# Stopwords / Regex
# =========================

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
    "mi",
    "mis",
    "su",
    "sus",
    "nos",
    "nosotros",
    "ustedes",
    "ellos",
    "ellas",
    "que",
    "del",
    "al",
    "se",
    "lo",
    "como",
}

# Extra blacklist for "evidence" extraction (weak filler tokens)
EVIDENCE_BLACKLIST = {
    "also",
    "quite",
    "just",
    "only",
    "really",
    "very",
    "overall",
    "almost",
    "so",
    "that",
    "could",
    "barely",
    "might",
    "back",
    "come",
    "if",
    "get",
    "had",
    "table",
    "itself",
    "looks",
    "level",
    "times",
    "actually",
    "only if",
    "has",
    "some",
    "since",
    "generally",
    "feels",
    "model",
    "platform",
    "like",
    "we",
    "were",
    "menu",
    "night",
    "throughout",
    "would",
    "again",
    "first",
    "after",
    "nearly",
    "slightly",
    "though",
    "started",
    "system",
    "users",
    "user",
    "potential",
    "needs",
    "better",
    "more",
    "during",
    "active",
    "sessions",
    "sometimes",
    "off",
    "once",
    "which",
    "theres",
    "there's",
    "around",
    "through",
}
EVIDENCE_PHRASE_BLACKLIST = {
    "platform has",
    "since some",
    "some advanced",
    "generally fast",
}
EVIDENCE_GENERIC_SINGLE_VERBS = {"worked", "feels", "like", "makes"}
SHORT_MEANINGFUL_EVIDENCE = {"ui", "ux", "2fa"}

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'_-]{1,}")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:,|;|:|\b(?:and|but|however|though|because|although|while|pero|aunque|porque|sin embargo|mientras)\b)\s*",
    re.IGNORECASE,
)

GENERIC_SEED_TOKENS = {
    "app",
    "service",
    "system",
    "platform",
    "producto",
    "servicio",
    "company",
    "business",
    "place",
    "thing",
}

MOJIBAKE_MARKERS = ("\u00e2\u0080", "\u00c3", "\u00c2")
MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u0080\u0099": "'",
    "\u00e2\u0080\u0098": "'",
    "\u00e2\u0080\u009c": '"',
    "\u00e2\u0080\u009d": '"',
    "\u00e2\u0080\u0093": "-",
    "\u00e2\u0080\u0094": "-",
    "\u00e2\u0080\u00a6": "...",
    "\u00c2 ": " ",
    "\u00c2": "",
}

# =========================
# Normalization / Tokenization
# =========================


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _repair_mojibake(text: str) -> str:
    raw = text or ""
    if any(marker in raw for marker in MOJIBAKE_MARKERS):
        try:
            repaired = raw.encode("latin1").decode("utf-8")
            if repaired:
                raw = repaired
        except Exception:
            pass

    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        raw = raw.replace(bad, good)
    return raw


def _normalize_text(text: str) -> str:
    lowered = _repair_mojibake(text or "").lower().strip()
    lowered = _strip_accents(lowered)
    lowered = re.sub(r"[“”«»]", '"', lowered)
    lowered = re.sub(r"[’`´]", "'", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered

def _tokenize(text: str) -> list[str]:
    norm = _normalize_text(text)
    toks = TOKEN_RE.findall(norm)
    return [tok for tok in toks if tok not in STOPWORDS]


def _split_units(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []

    sentences = [s.strip(" \t\r\n-*") for s in SENT_SPLIT_RE.split(raw) if s.strip()]
    units: list[str] = []

    for sentence in (sentences if sentences else [raw]):
        parts = [p.strip() for p in CLAUSE_SPLIT_RE.split(sentence) if p.strip()]
        units.extend(parts if parts else [sentence])

    merged: list[str] = []
    for u in units:
        u = u.strip()
        if not u:
            continue

        toks = _tokenize(u)
        is_tiny = (len(u) <= 10) or (len(toks) <= 1)

        if is_tiny and merged:
            merged[-1] = f"{merged[-1]} {u}".strip()
        else:
            merged.append(u)

    merged = [u for u in merged if len(u) >= 4]
    return merged


# =========================
# Sentiment (simple)
# =========================

POS_LEXICON = {
    "amazing": 1.8,
    "great": 1.5,
    "excellent": 2.0,
    "awesome": 1.8,
    "good": 1.1,
    "perfect": 1.8,
    "reliable": 1.4,
    "intuitive": 1.2,
    "responsive": 1.1,
    "stable": 1.1,
    "helpful": 1.2,
    "friendly": 1.0,
    "fast": 1.4,
    "excelente": 2.0,
    "genial": 1.7,
    "bueno": 1.1,
    "perfecto": 1.8,
    "rapido": 1.1,
    "estable": 1.1,
    "confiable": 1.4,
    "util": 1.0,
    "delicious": 1.7,
    "tasty": 1.5,
    "rico": 1.4,
    "delicioso": 1.8,
    "beautiful": 1.2,
    "creative": 1.1,
    "impressive": 1.2,
    "fresh": 1.3,
    "flavorful": 1.3,
    "romantic": 1.0,
    "stylish": 1.0,
    "cozy": 1.0,
    "clean": 1.2,
    "smooth": 1.2,
    "straightforward": 1.2,
    "thorough": 1.2,
    "robust": 1.3,
    "secure": 1.3,
    "clear": 1.1,
    "guided": 1.1,
    "well-structured": 1.2,
    "easy": 1.0,
    "reassuring": 1.1,
    "consistent": 1.1,
    "functional": 1.0,
    "sleek": 1.1,
    "polished": 1.1,
    "fine": 0.6,
}

NEG_LEXICON = {
    "bad": -1.2,
    "terrible": -2.0,
    "awful": -2.0,
    "slow": -1.1,
    "rude": -1.2,
    "expensive": -1.0,
    "overpriced": -1.4,
    "broken": -1.6,
    "defect": -1.4,
    "crash": -2.2,
    "crashes": -2.2,
    "bug": -1.2,
    "bugs": -1.2,
    "lag": -1.2,
    "laggy": -1.2,
    "unreliable": -1.4,
    "misleading": -1.3,
    "error": -1.3,
    "errors": -1.3,
    "fails": -1.4,
    "failure": -1.5,
    "freeze": -1.5,
    "freezes": -1.5,
    "malo": -1.2,
    "horrible": -2.0,
    "lento": -1.1,
    "caro": -1.0,
    "roto": -1.6,
    "falla": -1.4,
    "fallo": -1.4,
    "enganoso": -1.5,
    "noisy": -1.0,
    "ruidoso": -1.0,
    "dirty": -1.2,
    "sucio": -1.2,
    "bland": -1.2,
    "insipido": -1.2,
    "fail": -1.4,
    "overcooked": -1.4,
    "undercooked": -1.4,
    "under-seasoned": -1.2,
    "dry": -1.0,
    "invisible": -1.0,
    "inconsistent": -1.0,
    "pricey": -1.0,
    "logout": -1.2,
    "logs": -0.6,
    "overcrowded": -1.1,
    "chaotic": -1.1,
    "bureaucratic": -1.0,
    "inefficient": -1.2,
    "contradictory": -1.0,
    "unclear": -1.0,
}
NEG_LEXICON.update(
    {
        "waited": -1.0,
        "wait": -1.0,
        "waiting": -1.0,
        "overwhelmed": -1.0,
        "ignored": -1.2,
        "delay": -0.9,
        "delayed": -0.9,
        "minutes": -0.2,
        "minute": -0.2,
        "hour": -0.4,
        "hours": -0.4,
        "tardaron": -1.0,
        "tardo": -1.0,
        "espera": -1.0,
        "esperar": -1.0,
        "minutos": -0.2,
        "hora": -0.4,
        "horas": -0.4,
    }
)

POS_PHRASES = {
    "worth every penny": 2.2,
    "works like a charm": 1.8,
    "highly recommend": 2.0,
    "vale la pena": 1.8,
    "muy recomendable": 2.0,
    "funciona perfecto": 1.9,
    "generally fast": 1.1,
    "started off beautifully": 1.2,
    "tasted fresh and flavorful": 1.4,
    "well structured": 1.2,
    "worked fine": 1.0,
    "easy to follow": 1.2,
    "well-structured": 1.2,
    "highly intuitive": 1.3,
}

NEG_PHRASES = {
    "not worth it": -2.0,
    "waste of money": -2.2,
    "does not work": -1.8,
    "doesn't work": -1.8,
    "no response": -1.6,
    "double charge": -1.8,
    "data loss": -2.1,
    "too expensive": -1.5,
    "take forever": -1.4,
    "took forever": -1.4,
    "waited nearly": -1.2,
    "long wait": -1.2,
    "we were practically invisible": -1.5,
    "without any error message": -1.9,
    "no error message": -1.7,
    "fails without any error": -1.8,
    "fails without warning": -1.8,
    "without warning": -1.5,
    "fails silently": -1.8,
    "silent failure": -1.9,
    "randomly logs users out": -2.0,
    "logs users out": -1.8,
    "resets randomly": -1.7,
    "took several days": -1.4,
    "over 48 hours": -1.4,
    "response time feels too long": -1.5,
    "too long for enterprise": -1.6,
    "replies eventually": -1.1,
    "no vale la pena": -2.0,
    "no funciona": -1.8,
    "sin respuesta": -1.6,
    "cargo duplicado": -1.8,
    "perdida de datos": -2.1,
}
NEGATION_SAFE_NEG_PHRASES = {
    "without any error message",
    "no error message",
    "fails without any error",
    "fails without warning",
    "without warning",
    "fails silently",
    "silent failure",
}

NEGATION = {"not", "no", "never", "without", "hardly", "rarely", "sin", "nunca", "jamas"}
INTENSIFIERS = {
    "very": 1.35,
    "really": 1.2,
    "extremely": 1.6,
    "super": 1.4,
    "too": 1.25,
    "highly": 1.3,
    "muy": 1.35,
    "sumamente": 1.55,
    "bastante": 1.2,
    "demasiado": 1.25,
}
DOWNPLAYERS = {"slightly": 0.6, "somewhat": 0.75, "barely": 0.55, "poco": 0.65, "algo": 0.75, "apenas": 0.55}
SENTIMENT_NORM_SCALE = 2.0
SENTIMENT_POS_THRESHOLD = math.tanh(0.65 / SENTIMENT_NORM_SCALE)
SENTIMENT_NEG_THRESHOLD = -SENTIMENT_POS_THRESHOLD


def _is_negated(tokens: list[str], idx: int, window: int = 3) -> bool:
    start = max(0, idx - window)
    return any(tokens[j] in NEGATION for j in range(start, idx))


def _modifier(tokens: list[str], idx: int) -> float:
    start = max(0, idx - 2)
    factor = 1.0
    for j in range(start, idx):
        prev = tokens[j]
        if prev in INTENSIFIERS:
            factor *= INTENSIFIERS[prev]
        elif prev in DOWNPLAYERS:
            factor *= DOWNPLAYERS[prev]
    return factor


def _sentiment_from_phrases(normalized_unit: str) -> float:
    score = 0.0
    for phrase, value in POS_PHRASES.items():
        if phrase in normalized_unit:
            score += value
    for phrase, value in NEG_PHRASES.items():
        if phrase in normalized_unit:
            score += value
    return score


def _simple_sentiment(unit: str, tokens: list[str]) -> tuple[str, float, float]:
    normalized_unit = _normalize_text(unit)
    score = _sentiment_from_phrases(normalized_unit)
    protect_without_negation = any(phrase in normalized_unit for phrase in NEGATION_SAFE_NEG_PHRASES)
    has_phrase_match = any(phrase in normalized_unit for phrase in POS_PHRASES) or any(
        phrase in normalized_unit for phrase in NEG_PHRASES
    )
    token_weight = 0.6 if has_phrase_match else 1.0

    for idx, token in enumerate(tokens):
        mod = _modifier(tokens, idx)
        if token in POS_LEXICON:
            value = POS_LEXICON[token] * mod * token_weight
            score += -value if _is_negated(tokens, idx) else value
        elif token in NEG_LEXICON:
            value = NEG_LEXICON[token] * mod * token_weight
            negated = _is_negated(tokens, idx)
            if negated and protect_without_negation:
                window = tokens[max(0, idx - 3) : idx]
                if "without" in window:
                    negated = False
            score += -value if negated else value

    raw_score = float(score)
    norm_score = float(math.tanh(raw_score / SENTIMENT_NORM_SCALE))

    if norm_score > SENTIMENT_POS_THRESHOLD:
        return "positive", norm_score, raw_score
    if norm_score < SENTIMENT_NEG_THRESHOLD:
        return "negative", norm_score, raw_score
    return "neutral", norm_score, raw_score


# =========================
# Taxonomies
# =========================

SAAS_ASPECTS: dict[str, list[str]] = {
    "PRODUCT_QUALITY": [
        "quality",
        "build quality",
        "durability",
        "defect",
        "defective",
        "broken",
        "well made",
        "poor quality",
        "calidad",
        "defecto",
        "defectuoso",
        "roto",
        "mala calidad",
    ],
    "FEATURES_FUNCTIONALITY": [
        "feature",
        "features",
        "functionality",
        "capability",
        "missing feature",
        "does not work",
        "no funciona",
        "falta funcion",
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
        "setup",
        "onboarding",
        "wizard",
        "guided",
        "installation",
        "flow",
        "walkthrough",
        "easy to follow",
        "design",
        "learning curve",
        "facil de usar",
        "dificil de usar",
        "intuitivo",
        "confuso",
        "navegacion",
        "interfaz",
        "diseno",
    ],
    "PERFORMANCE_SPEED": [
        "fast",
        "slow",
        "take forever",
        "lag",
        "latency",
        "performance",
        "freeze",
        "load time",
        "loading",
        "rapido",
        "lento",
        "latencia",
        "rendimiento",
        "se congela",
        "tiempo de carga",
    ],
    "RELIABILITY_STABILITY": [
        "crash",
        "crashes",
        "bug",
        "bugs",
        "error",
        "fails",
        "logout",
        "logs out",
        "disconnect",
        "disconnects",
        "resets",
        "session",
        "wifi",
        "connection",
        "timeout",
        "timeouts",
        "offline",
        "upload",
        "downtime",
        "outage",
        "falla",
        "fallo",
        "intermitente",
    ],
    "CUSTOMER_SUPPORT_SERVICE": [
        "support",
        "customer service",
        "ticket",
        "refund",
        "warranty",
        "respond",
        "response",
        "reply",
        "replied",
        "replies",
        "response time",
        "slow to respond",
        "slow response",
        "no response",
        "over 48 hours",
        "took several days",
        "replies eventually",
        "soporte",
        "atencion al cliente",
        "sin respuesta",
        "reembolso",
        "garantia",
    ],
    "PRICE_VALUE": [
        "price",
        "pricing",
        "subscription",
        "expensive",
        "overpriced",
        "cheap",
        "value",
        "worth it",
        "not worth it",
        "precio",
        "caro",
        "barato",
        "vale la pena",
        "no vale la pena",
    ],
    "SECURITY_PRIVACY": [
        "security",
        "privacy",
        "breach",
        "leak",
        "2fa",
        "two-factor",
        "biometric",
        "face id",
        "touch id",
        "gdpr",
        "pii",
        "seguridad",
        "privacidad",
        "fuga de datos",
    ],
    "OTHER": ["overall", "general", "experience", "feedback", "en general"],
}

RESTAURANT_ASPECTS: dict[str, list[str]] = {
    "FOOD_QUALITY": [
        "food quality",
        "taste",
        "flavor",
        "flavorful",
        "delicious",
        "bland",
        "fresh",
        "presentation",
        "drinks",
        "cocktails",
        "under-seasoned",
        "overcooked",
        "undercooked",
        "dry",
        "perfectly cooked",
        "steak",
        "calidad de la comida",
        "sabor",
        "rico",
        "delicioso",
        "insipido",
        "fresco",
        # Desserts/sweets (added to avoid misclassifying dessert praise as OTHER)
        "desserts",
        "dessert",
        "postres",
        "postre",
        "sweet",
        "chocolate",
        "cake",
        "pastel",
        "ice cream",
        "helado",
        # Stronger phrases
        "desserts were amazing",
        "postres increibles",
    ],
    "SERVICE": [
        "service",
        "waiter",
        "server",
        "staff",
        "host",
        "attentive",
        "friendly",
        "rude",
        "ignored",
        "invisible",
        "unattended",
        "no one came",
        "no attention",
        "neglected",
        "wrong order",
        "servicio",
        "mesero",
        "mesera",
        "personal",
        "amable",
        "grosero",
        "nos ignoraron",
        "sin atencion",
        "nadie vino",
        "se equivocaron",
    ],
    "WAIT_TIME": [
        "wait time",
        "wait",
        "waited",
        "waiting",
        "long wait",
        "took forever",
        "minutes",
        "minute",
        "hour",
        "bill",
        "check",
        "an hour",
        "line",
        "queue",
        "tiempo de espera",
        "espera",
        "esperamos",
        "tardaron",
        "minutos",
        "hora",
        "cuenta",
        "una hora",
        "fila",
        "cola",
    ],
    "AMBIENCE": [
        "ambience",
        "atmosphere",
        "music",
        "lighting",
        "decor",
        "beautiful",
        "modern",
        "cozy",
        "noise",
        "noisy",
        "too loud",
        "ambiente",
        "atmosfera",
        "decoracion",
        "bonito",
        "moderno",
        "ruido",
        "ruidoso",
        "muy fuerte",
    ],
    "CLEANLINESS": [
        "clean",
        "cleanliness",
        "dirty",
        "bathroom",
        "restroom",
        "hygiene",
        "limpio",
        "limpieza",
        "sucio",
        "baÃ±o",
        "higiene",
    ],
    "PRICE_VALUE": [
        "price",
        "expensive",
        "overpriced",
        "value",
        "worth it",
        "not worth it",
        "portion size",
        "small portions",
        "precio",
        "caro",
        "sobreprecio",
        "vale la pena",
        "no vale la pena",
        "porciones",
        "porciones pequeÃ±as",
    ],
    "MENU_VARIETY": [
        "menu",
        "options",
        "choices",
        "limited menu",
        "specials",
        "menÃº",
        "variedad",
        "pocas opciones",
        "muchas opciones",
        "especiales",
    ],
    "DRINKS_BAR": [
        "drinks",
        "cocktail",
        "beer",
        "wine",
        "bar",
        "coffee",
        "bebidas",
        "coctel",
        "cerveza",
        "vino",
        "barra",
        "cafe",
    ],
    # New aspect: "intent to return" is common in restaurant reviews and should not be forced into FOOD_QUALITY
    "RETURN_INTENT": [
        "come back",
        "return",
        "would come again",
        "come again",
        "back again",
        "volveria",
        "regresaria",
        "volver",
        "regresar",
        "volvere",
        "regresare",
    ],
    "OTHER": ["overall", "general", "experience", "thoughts", "en general"],
}

GENERAL_ASPECTS: dict[str, list[str]] = {
    "QUALITY": ["quality", "good quality", "poor quality", "calidad", "mala calidad", "durable", "fragile"],
    "SPEED_TIME": ["fast", "slow", "wait", "waiting", "latency", "tiempo", "tardado", "lento", "rapido"],
    "WAIT_TIME": [
        "wait",
        "waited",
        "waiting",
        "delay",
        "delayed",
        "line",
        "queue",
        "minutes",
        "minute",
        "hours",
        "hour",
        "tardaron",
        "espera",
        "minutos",
        "hora",
        "fila",
        "cola",
    ],
    "SERVICE_SUPPORT": ["service", "support", "customer service", "help", "soporte", "servicio", "atencion"],
    "PRICE_VALUE": ["price", "pricing", "expensive", "cheap", "value", "precio", "caro", "barato", "vale la pena"],
    "CLARITY_COMMUNICATION": [
        "unclear",
        "clear",
        "instructions",
        "contradictory",
        "confusing",
        "communication",
        "explained",
        "explanation",
        "information",
    ],
    "PROCESS_EFFICIENCY": [
        "process",
        "inefficient",
        "bureaucratic",
        "paperwork",
        "repetitive",
        "full-day ordeal",
        "unnecessarily",
        "chaotic",
        "overcrowded",
    ],
    "FACILITIES_ENVIRONMENT": [
        "building",
        "facilities",
        "outdated",
        "functional",
        "waiting room",
        "overcrowded",
        "environment",
        "clean",
        "dirty",
    ],
    "OTHER": ["overall", "general", "experience", "feedback", "en general"],
}

TaxonomyName = str

TAXONOMIES: dict[TaxonomyName, dict[str, list[str]]] = {
    "restaurant": RESTAURANT_ASPECTS,
    "saas": SAAS_ASPECTS,
    "general": GENERAL_ASPECTS,
}

DEFAULT_FALLBACK_TAXONOMY: TaxonomyName = "general"

# Aspect token anchors used as lightweight per-unit confidence boosts.
ASPECT_ANCHORS: dict[str, dict[str, set[str]]] = {
    "restaurant": {
        "WAIT_TIME": {
            "wait",
            "waited",
            "waiting",
            "minutes",
            "minute",
            "hour",
            "bill",
            "check",
            "queue",
            "line",
            "tardaron",
            "esperamos",
            "espera",
            "minutos",
            "hora",
            "cuenta",
        },
        "SERVICE": {
            "service",
            "staff",
            "waiter",
            "server",
            "ignored",
            "invisible",
            "attention",
            "neglected",
            "unattended",
            "atencion",
        },
        "FOOD_QUALITY": {
            "steak",
            "risotto",
            "dessert",
            "desserts",
            "cocktail",
            "cocktails",
            "flavor",
            "flavorful",
            "fresh",
            "overcooked",
            "undercooked",
            "under-seasoned",
            "dry",
        },
        "AMBIENCE": {
            "music",
            "volume",
            "noise",
            "noisy",
            "ambience",
            "atmosphere",
            "romantic",
            "cozy",
        },
        "PRICE_VALUE": {
            "price",
            "pricing",
            "expensive",
            "overpriced",
            "pricey",
            "higher",
            "side",
            "value",
        },
    },
    "saas": {
        "USABILITY_UX": {
            "ui",
            "ux",
            "interface",
            "navigation",
            "setup",
            "onboarding",
            "wizard",
            "guided",
            "installation",
            "flow",
            "walkthrough",
            "easy",
            "intuitive",
            "smooth",
        },
        "RELIABILITY_STABILITY": {
            "crash",
            "crashes",
            "bug",
            "bugs",
            "error",
            "errors",
            "fail",
            "fails",
            "logout",
            "logs",
            "disconnect",
            "disconnects",
            "resets",
            "session",
            "wifi",
            "connection",
            "timeout",
            "timeouts",
            "upload",
            "offline",
        },
        "SECURITY_PRIVACY": {
            "security",
            "privacy",
            "2fa",
            "two-factor",
            "biometric",
            "face",
            "touch",
            "authentication",
            "auth",
            "secure",
        },
        "PERFORMANCE_SPEED": {
            "fast",
            "slow",
            "latency",
            "lag",
            "loading",
            "load",
            "minutes",
            "hours",
        },
        "CUSTOMER_SUPPORT_SERVICE": {
            "support",
            "ticket",
            "response",
            "respond",
            "reply",
            "replied",
            "replies",
            "hours",
            "days",
        },
    },
    "general": {
        "SPEED_TIME": {"fast", "slow", "latency", "tiempo", "rapido", "lento"},
        "WAIT_TIME": {"wait", "waiting", "waited", "delay", "line", "queue", "minutes", "hours", "espera"},
        "SERVICE_SUPPORT": {"service", "support", "help", "servicio", "soporte", "atencion"},
        "CLARITY_COMMUNICATION": {"unclear", "clear", "instructions", "contradictory", "communication", "information"},
        "PROCESS_EFFICIENCY": {"process", "inefficient", "bureaucratic", "paperwork", "repetitive", "chaotic", "overcrowded"},
        "FACILITIES_ENVIRONMENT": {"building", "facilities", "outdated", "functional", "waiting", "clean", "dirty"},
        "PRICE_VALUE": {"price", "pricing", "expensive", "cheap", "value", "precio", "caro", "barato"},
        "QUALITY": {"quality", "durable", "fragile", "calidad"},
    },
}

TAXONOMY_MIN_MARGIN = 0.05
TAXONOMY_MIN_SCORE = 0.10
TAXONOMY_ANCHOR_MIN_HITS = 2
TAXONOMY_DOMAIN_ANCHORS: dict[str, set[str]] = {
    "saas": {
        "app",
        "crash",
        "crashes",
        "logout",
        "subscription",
        "upload",
        "offline",
        "latency",
        "2fa",
        "authentication",
        "wifi",
        "timeout",
    },
    "restaurant": {
        "steak",
        "risotto",
        "waiter",
        "menu",
        "cocktails",
        "dessert",
        "table",
        "ambience",
        "service",
    },
}

SUPPORT_RESPONSE_TERMS = {"respond", "response", "reply", "replied", "replies"}
WAIT_TRIGGERS = {
    "wait",
    "waited",
    "waiting",
    "delay",
    "delayed",
    "minutes",
    "minute",
    "hour",
    "hours",
    "bill",
    "check",
    "queue",
    "line",
    "tardaron",
    "espera",
    "esperamos",
    "minutos",
    "hora",
    "cuenta",
}
FOOD_TRIGGERS = {"tasted", "flavor", "delicious", "fresh", "steak", "risotto"}
PRICE_TOKENS = {
    "price",
    "pricing",
    "cost",
    "costs",
    "fee",
    "fees",
    "charged",
    "charges",
    "expensive",
    "overpriced",
    "cheap",
    "subscription",
}
SUPPORT_TOKENS = {
    "support",
    "service",
    "ticket",
    "help",
    "response",
    "respond",
    "reply",
    "replied",
    "replies",
    "soporte",
    "atencion",
}
FAILURE_TOKENS = {
    "fail",
    "fails",
    "failed",
    "failure",
    "crash",
    "crashes",
    "error",
    "errors",
    "bug",
    "bugs",
    "timeout",
    "timeouts",
    "logout",
    "disconnect",
    "disconnects",
    "reset",
    "resets",
    "offline",
    "upload",
}

# =========================
# Generic taxonomy detector (N taxonomies)
# =========================


def _flatten_taxonomy(tax: dict[str, list[str]]) -> list[str]:
    seeds: list[str] = []
    for aspect, items in (tax or {}).items():
        if not isinstance(items, list):
            continue
        seeds.append(aspect)
        for s in items:
            s = str(s or "").strip()
            if s:
                seeds.append(s)
    return seeds


def _build_taxonomy_profiles(taxonomies: dict[TaxonomyName, dict[str, list[str]]]) -> tuple[list[str], list[TaxonomyName]]:
    docs: list[str] = []
    names: list[TaxonomyName] = []

    for name, tax in taxonomies.items():
        seeds = _flatten_taxonomy(tax)
        if not seeds:
            continue

        uniq: list[str] = []
        seen: set[str] = set()

        for s in seeds:
            n = _normalize_text(s)
            if not n or n in seen:
                continue
            seen.add(n)
            uniq.append(s)

        docs.append(" ".join(uniq))
        names.append(name)

    if not docs:
        raise ValueError("No valid taxonomies to build profiles")

    return docs, names


def _build_tax_detector_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=_normalize_text,
        lowercase=True,
        stop_words=list(STOPWORDS),
        ngram_range=(1, 3),
        analyzer="word",
        sublinear_tf=True,
        token_pattern=r"[a-z0-9][a-z0-9'_-]{1,}",
    )


def _build_tax_detector_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=_normalize_text,
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )


@dataclass
class TaxonomyDetector:
    taxonomies: dict[TaxonomyName, dict[str, list[str]]]
    word_vec: TfidfVectorizer
    char_vec: TfidfVectorizer
    names: list[TaxonomyName]
    word_profiles: Any
    char_profiles: Any
    alpha: float = 0.7

    @classmethod
    def build(cls, taxonomies: dict[TaxonomyName, dict[str, list[str]]], alpha: float = 0.7) -> "TaxonomyDetector":
        docs, names = _build_taxonomy_profiles(taxonomies)

        wv = _build_tax_detector_word_vectorizer()
        cv = _build_tax_detector_char_vectorizer()

        W = normalize(wv.fit_transform(docs), norm="l2", axis=1, copy=False)
        C = normalize(cv.fit_transform(docs), norm="l2", axis=1, copy=False)

        return cls(
            taxonomies=taxonomies,
            word_vec=wv,
            char_vec=cv,
            names=names,
            word_profiles=W,
            char_profiles=C,
            alpha=float(alpha),
        )

    def infer(self, text: str, min_margin: float = 0.06) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {"name": None, "confidence": 0.0, "margin": 0.0, "scores": []}

        w = normalize(self.word_vec.transform([raw]), norm="l2", axis=1, copy=False)
        c = normalize(self.char_vec.transform([raw]), norm="l2", axis=1, copy=False)

        word_scores = (self.word_profiles @ w.T).toarray().ravel()
        char_scores = (self.char_profiles @ c.T).toarray().ravel()
        scores = (self.alpha * word_scores) + ((1.0 - self.alpha) * char_scores)

        order = np.argsort(scores)[::-1]
        best_i = int(order[0])
        best = float(scores[best_i])
        second = float(scores[int(order[1])]) if scores.size > 1 else 0.0
        margin = best - second

        name = self.names[best_i] if margin >= float(min_margin) else None

        top: list[dict[str, Any]] = []
        for i in order[: min(5, scores.size)]:
            top.append({"taxonomy": self.names[int(i)], "score": float(scores[int(i)])})

        return {
            "name": name,
            "confidence": float(best),
            "margin": float(margin),
            "scores": top,
        }


_TAX_DETECTOR = TaxonomyDetector.build(TAXONOMIES, alpha=0.72)

# =========================
# Granulate (aspect assignment)
# =========================


@dataclass
class Granule:
    aspect: str
    excerpt: str
    evidence: list[str]
    sentiment: str
    sentiment_score: float
    sentiment_raw: float
    similarity: float
    lexical_overlap: float
    confidence: float
    confidence_margin: float = 0.0
    debug_top_aspects: list[dict[str, float]] | None = None


def _normalize_aspects(custom: dict[str, list[str]] | None) -> dict[str, list[str]]:
    if not custom:
        return GENERAL_ASPECTS

    out: dict[str, list[str]] = {}
    for key, seeds in custom.items():
        clean_key = (key or "").strip().upper()
        if not clean_key or not isinstance(seeds, list) or not seeds:
            continue
        cleaned = [str(seed).strip() for seed in seeds if str(seed).strip()]
        if cleaned:
            out[clean_key] = cleaned

    if "OTHER" not in out:
        out["OTHER"] = ["overall", "general", "experience", "feedback", "en general"]

    return out


def _build_vectorizer_for_granulate() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=_normalize_text,
        lowercase=True,
        stop_words=list(STOPWORDS),
        ngram_range=(1, 3),
        token_pattern=r"[a-z0-9][a-z0-9'_-]{1,}",
        sublinear_tf=True,
        min_df=1,
    )


def _make_seed_index(aspects_map: dict[str, list[str]]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for name, seeds in aspects_map.items():
        seed_tokens: set[str] = set()
        for seed in seeds:
            for token in _tokenize(seed):
                if token in GENERIC_SEED_TOKENS:
                    continue
                seed_tokens.add(token)

            norm_seed = _normalize_text(seed)
            if " " in norm_seed:
                seed_tokens.add(norm_seed)

        index[name] = seed_tokens
    return index


def _lexical_overlap(unit: str, tokens: list[str], seed_index: dict[str, set[str]]) -> dict[str, float]:
    norm_unit = _normalize_text(unit)
    unit_token_set = set(tokens)

    overlap: dict[str, float] = {}
    for aspect, seeds in seed_index.items():
        if not seeds:
            overlap[aspect] = 0.0
            continue

        token_hits = len(unit_token_set.intersection(seeds))
        phrase_hits = sum(1 for seed in seeds if " " in seed and seed in norm_unit)

        denom = max(1, len(tokens))
        overlap[aspect] = min(1.0, (token_hits + (1.6 * phrase_hits)) / denom)

    return overlap


def _build_seed_docs(aspects_map: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    docs: list[str] = []
    doc_aspects: list[str] = []

    for aspect, seeds in aspects_map.items():
        if aspect == "OTHER":
            continue

        seen: set[str] = set()
        uniq: list[str] = []
        for s in seeds:
            n = _normalize_text(s)
            if not n or n in seen:
                continue
            seen.add(n)
            uniq.append(s)

        for seed in uniq:
            docs.append(seed)
            doc_aspects.append(aspect)

        if len(uniq) >= 3:
            docs.append(" ".join(uniq[:12]))
            doc_aspects.append(aspect)

    return docs, doc_aspects


def _taxonomy_override_from_anchors(text: str) -> str | None:
    norm_text = _normalize_text(text)
    token_set = set(_tokenize(norm_text))
    if not token_set:
        return None

    scores: dict[str, int] = {}
    for tax, anchors in TAXONOMY_DOMAIN_ANCHORS.items():
        token_hits = len(token_set.intersection(anchors))
        phrase_hits = sum(1 for a in anchors if " " in a and a in norm_text)
        scores[tax] = int(token_hits + (2 * phrase_hits))

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return None
    best_tax, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if best_score >= TAXONOMY_ANCHOR_MIN_HITS and best_score > second_score:
        return best_tax
    return None


def _fallback_aspect_from_tokens(token_set: set[str], taxonomy: str, aspect_names: list[str]) -> tuple[str | None, float]:
    if token_set.intersection(SUPPORT_TOKENS):
        if "CUSTOMER_SUPPORT_SERVICE" in aspect_names:
            return "CUSTOMER_SUPPORT_SERVICE", 0.18
        if "SERVICE_SUPPORT" in aspect_names:
            return "SERVICE_SUPPORT", 0.18
        if "SERVICE" in aspect_names:
            return "SERVICE", 0.18

    if token_set.intersection(PRICE_TOKENS):
        if "PRICE_VALUE" in aspect_names:
            return "PRICE_VALUE", 0.18

    if token_set.intersection(WAIT_TRIGGERS):
        if "WAIT_TIME" in aspect_names:
            return "WAIT_TIME", 0.18
        if taxonomy == "saas" and "PERFORMANCE_SPEED" in aspect_names:
            return "PERFORMANCE_SPEED", 0.16
        if "SPEED_TIME" in aspect_names:
            return "SPEED_TIME", 0.16

    if token_set.intersection(FAILURE_TOKENS):
        if taxonomy == "saas" and "RELIABILITY_STABILITY" in aspect_names:
            return "RELIABILITY_STABILITY", 0.2
        if "QUALITY" in aspect_names:
            return "QUALITY", 0.18
        if "PRODUCT_QUALITY" in aspect_names:
            return "PRODUCT_QUALITY", 0.18

    return None, 0.0


def _is_evidence_term_allowed(term: str) -> bool:
    norm_term = _normalize_text(term)
    if not norm_term:
        return False
    if norm_term in STOPWORDS or norm_term in EVIDENCE_BLACKLIST:
        return False
    if any(blk in norm_term for blk in EVIDENCE_PHRASE_BLACKLIST):
        return False

    term_tokens = _tokenize(norm_term)
    if not term_tokens:
        return False

    if len(term_tokens) == 1:
        tok = term_tokens[0]
        if tok in EVIDENCE_GENERIC_SINGLE_VERBS:
            return False
        if tok in SHORT_MEANINGFUL_EVIDENCE:
            return True
        return len(tok) >= 4 and tok not in STOPWORDS and tok not in EVIDENCE_BLACKLIST

    # For n-grams, keep terms that still carry at least one meaningful token.
    meaningful_tokens = [tok for tok in term_tokens if tok not in STOPWORDS and tok not in EVIDENCE_BLACKLIST and len(tok) >= 4]
    return len(meaningful_tokens) >= 1


def _evidence_from_unit(vectorizer: TfidfVectorizer, unit_vec, tokens: list[str], k: int) -> list[str]:
    k = max(1, int(k))
    features = np.array(vectorizer.get_feature_names_out())
    weights = unit_vec.toarray().ravel()

    order = np.argsort(weights)[::-1]

    evidence: list[str] = []
    seen: set[str] = set()

    for j in order:
        if weights[j] <= 0:
            break
        term = str(features[j]).strip()
        if not term:
            continue
        norm_term = _normalize_text(term)
        if not _is_evidence_term_allowed(term) or norm_term in seen:
            continue
        if len(term) <= 2:
            continue
        seen.add(norm_term)
        evidence.append(term)
        if len(evidence) >= k:
            break

    if not evidence:
        return tokens[:k]
    return evidence


def _build_aspect_summary(
    granules: list[Granule], aspect_names: list[str], top_evidence_k: int = 3, include_other: bool = False
) -> dict[str, Any]:
    top_evidence_k = max(1, int(top_evidence_k))
    out: dict[str, Any] = {}

    for asp in aspect_names:
        if not include_other and asp == "OTHER":
            continue
        g_list = [g for g in granules if g.aspect == asp]
        if not g_list:
            out[asp] = {
                "count": 0,
                "avg_sentiment": 0.0,
                "avg_sentiment_score": 0.0,
                "avg_sentiment_raw": 0.0,
                "top_evidence": [],
            }
            continue

        avg_sent = float(sum(g.sentiment_score for g in g_list) / max(1, len(g_list)))
        avg_sent_raw = float(sum(g.sentiment_raw for g in g_list) / max(1, len(g_list)))

        freq: dict[str, int] = {}
        for g in g_list:
            for e in g.evidence:
                e_norm = _normalize_text(e)
                if not _is_evidence_term_allowed(e):
                    continue
                freq[e] = freq.get(e, 0) + 1

        top = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0])))
        out[asp] = {
            "count": int(len(g_list)),
            "avg_sentiment": round(avg_sent, 3),
            "avg_sentiment_score": round(avg_sent, 3),
            "avg_sentiment_raw": round(avg_sent_raw, 3),
            "top_evidence": [t for (t, _) in top[:top_evidence_k]],
        }

    return out


def granulate_text(
    text: str,
    aspects: dict[str, list[str]] | None = None,
    top_k_evidence: int = 6,
    min_similarity: float = 0.12,
) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("text is required")

    units = _split_units(raw)
    if not units:
        raise ValueError("text is required")

    detected_taxonomy: str | None = None
    detection_margin: float | None = None
    taxonomy_candidates: list[dict[str, Any]] = []

    if aspects is not None:
        aspects_map = _normalize_aspects(aspects)
        detected_taxonomy = "custom"
    else:
        det = _TAX_DETECTOR.infer(raw, min_margin=TAXONOMY_MIN_MARGIN)
        taxonomy_candidates = det["scores"]
        detection_margin = float(det["margin"]) if det["margin"] is not None else None

        top_taxonomy = taxonomy_candidates[0]["taxonomy"] if taxonomy_candidates else DEFAULT_FALLBACK_TAXONOMY
        top_score = float(taxonomy_candidates[0]["score"]) if taxonomy_candidates else 0.0
        margin_value = float(det["margin"]) if det["margin"] is not None else 0.0

        chosen = DEFAULT_FALLBACK_TAXONOMY
        if (
            top_taxonomy != DEFAULT_FALLBACK_TAXONOMY
            and top_score >= TAXONOMY_MIN_SCORE
            and margin_value >= TAXONOMY_MIN_MARGIN
        ):
            chosen = str(top_taxonomy)
        else:
            anchor_taxonomy = _taxonomy_override_from_anchors(raw)
            if anchor_taxonomy in TAXONOMIES:
                chosen = anchor_taxonomy

        detected_taxonomy = str(chosen)
        aspects_map = TAXONOMIES.get(detected_taxonomy, TAXONOMIES[DEFAULT_FALLBACK_TAXONOMY])

    aspect_names = list(aspects_map.keys())
    seed_index = _make_seed_index(aspects_map)
    aspect_anchor_map = ASPECT_ANCHORS.get((detected_taxonomy or "").lower(), {})

    seed_docs, seed_doc_aspects = _build_seed_docs(aspects_map)
    vectorizer = _build_vectorizer_for_granulate()

    matrix = vectorizer.fit_transform(seed_docs + units)

    seed_vecs = normalize(matrix[: len(seed_docs)], norm="l2", axis=1, copy=False)
    unit_vecs = normalize(matrix[len(seed_docs) :], norm="l2", axis=1, copy=False)

    min_similarity = float(min_similarity)

    granules: list[Granule] = []

    for i, unit in enumerate(units):
        tokens = _tokenize(unit)
        token_set = set(tokens)
        has_support_context = any(tok in token_set for tok in {"support", "service", "ticket", "customer", "soporte", "atencion"})
        has_support_delay_context = any(tok in token_set for tok in {"hours", "hour", "days", "day", "response", "respond", "reply", "replied", "replies"})
        support_response_hit = any(tok in SUPPORT_RESPONSE_TERMS for tok in token_set)
        unit_vec = unit_vecs[i]

        cosine_seed_scores = (seed_vecs @ unit_vec.T).toarray().ravel()
        best_cos_by_aspect: dict[str, float] = {}

        for j, asp in enumerate(seed_doc_aspects):
            s = float(cosine_seed_scores[j])
            if s > best_cos_by_aspect.get(asp, 0.0):
                best_cos_by_aspect[asp] = s

        lexical_scores = _lexical_overlap(unit, tokens, seed_index)

        best_aspect = "OTHER"
        best_conf = 0.0
        best_cos = 0.0
        best_lex = 0.0
        confidence_margin = 0.0
        debug_top_aspects: list[dict[str, float]] = []

        lexical_weight = 0.30
        aspect_conf_map: dict[str, float] = {}
        aspect_cos_map: dict[str, float] = {}
        aspect_lex_map: dict[str, float] = {}
        aspect_anchor_hits_map: dict[str, int] = {}

        for asp in aspect_names:
            if asp == "OTHER":
                continue
            cos = float(best_cos_by_aspect.get(asp, 0.0))
            lex = float(lexical_scores.get(asp, 0.0))
            conf = (1.0 - lexical_weight) * cos + lexical_weight * lex
            anchor_hits = len(token_set.intersection(aspect_anchor_map.get(asp, set())))
            conf += 0.08 * anchor_hits
            if asp == "CUSTOMER_SUPPORT_SERVICE":
                if support_response_hit:
                    conf += 0.08
                if has_support_context and has_support_delay_context:
                    conf += 0.15

            aspect_conf_map[asp] = conf
            aspect_cos_map[asp] = cos
            aspect_lex_map[asp] = lex
            aspect_anchor_hits_map[asp] = int(anchor_hits)

        if aspect_conf_map:
            ranked = sorted(aspect_conf_map.items(), key=lambda kv: kv[1], reverse=True)
            debug_top_aspects = [{"aspect": asp, "confidence": round(float(conf), 4)} for asp, conf in ranked[:3]]
            best_aspect = ranked[0][0]
            best_conf = float(ranked[0][1])
            best_cos = float(aspect_cos_map.get(best_aspect, 0.0))
            best_lex = float(aspect_lex_map.get(best_aspect, 0.0))

        aspect_forced = False
        if any(t in SUPPORT_RESPONSE_TERMS for t in token_set) and best_aspect == "PERFORMANCE_SPEED":
            best_aspect = "CUSTOMER_SUPPORT_SERVICE"
            best_cos = float(aspect_cos_map.get(best_aspect, 0.0))
            best_lex = float(aspect_lex_map.get(best_aspect, 0.0))
            best_conf = float(aspect_conf_map.get(best_aspect, (1.0 - lexical_weight) * best_cos + lexical_weight * best_lex))
            aspect_forced = True

        has_wait_triggers = any(t in WAIT_TRIGGERS for t in token_set)
        if "WAIT_TIME" in aspect_names and has_wait_triggers:
            # Keep WAIT_TIME dominant even when food context appears in the same clause.
            best_aspect = "WAIT_TIME"
            best_cos = float(aspect_cos_map.get(best_aspect, 0.0))
            best_lex = float(aspect_lex_map.get(best_aspect, 0.0))
            best_conf = float(aspect_conf_map.get(best_aspect, (1.0 - lexical_weight) * best_cos + lexical_weight * best_lex))
            best_conf = max(best_conf, 0.75)
            aspect_forced = True

        if aspect_conf_map and best_aspect != "OTHER":
            others = [v for asp, v in aspect_conf_map.items() if asp != best_aspect]
            second_conf = max(others) if others else 0.0
            confidence_margin = float(best_conf - second_conf)
            if confidence_margin < 0.04:
                best_conf *= 0.8

        if (not aspect_forced) and best_cos < min_similarity and best_conf < max(min_similarity, 0.16):
            max_anchor_hits = max(aspect_anchor_hits_map.values()) if aspect_anchor_hits_map else 0
            if not (max_anchor_hits >= 2 and best_conf >= 0.14):
                best_aspect = "OTHER"
                confidence_margin = 0.0

        if best_aspect == "OTHER":
            fallback_aspect, fallback_conf = _fallback_aspect_from_tokens(token_set, str(detected_taxonomy or ""), aspect_names)
            if fallback_aspect is not None:
                best_aspect = fallback_aspect
                best_conf = max(best_conf, fallback_conf)
                best_cos = float(aspect_cos_map.get(best_aspect, best_cos))
                best_lex = float(aspect_lex_map.get(best_aspect, best_lex))

        sentiment_label, sentiment_score, sentiment_raw = _simple_sentiment(unit, tokens)
        evidence = _evidence_from_unit(vectorizer, unit_vec, tokens, int(top_k_evidence))

        if best_aspect != "OTHER":
            norm_unit = _normalize_text(unit)
            seed_candidates = sorted(seed_index.get(best_aspect, set()), key=len, reverse=True)
            for seed in seed_candidates:
                if " " not in seed:
                    continue
                if seed in norm_unit and seed not in evidence:
                    evidence.append(seed)
                if len(evidence) >= int(top_k_evidence):
                    break

        evidence = evidence[: int(top_k_evidence)]

        granules.append(
            Granule(
                aspect=best_aspect,
                excerpt=unit,
                evidence=evidence,
                sentiment=sentiment_label,
                sentiment_score=float(sentiment_score),
                sentiment_raw=float(sentiment_raw),
                similarity=float(best_cos),
                lexical_overlap=float(best_lex),
                confidence=float(best_conf),
                confidence_margin=float(confidence_margin),
                debug_top_aspects=debug_top_aspects,
            )
        )

    aspect_summary = _build_aspect_summary(granules, aspect_names, top_evidence_k=3, include_other=False)
    include_other_highlights = False
    highlight_candidates = granules if include_other_highlights else [g for g in granules if g.aspect != "OTHER"]
    if not highlight_candidates:
        highlight_candidates = granules
    highlights = sorted(highlight_candidates, key=lambda g: (abs(g.sentiment_score) * (0.6 + g.confidence)), reverse=True)[:3]

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
                "sentiment_raw": round(g.sentiment_raw, 3),
                "similarity": round(g.similarity, 4),
                "lexical_overlap": round(g.lexical_overlap, 4),
                "confidence": round(g.confidence, 4),
                "confidence_margin": round(g.confidence_margin, 4),
                "debug_top_aspects": g.debug_top_aspects or [],
            }
            for g in granules
        ],
        "taxonomy": aspect_names,
        "detected_taxonomy": detected_taxonomy,
        "taxonomy_candidates": taxonomy_candidates,
        "detection_margin": None if detection_margin is None else round(float(detection_margin), 4),
        # UI-friendly aggregates
        "aspect_summary": aspect_summary,
        "highlights": [
            {
                "aspect": g.aspect,
                "excerpt": g.excerpt,
                "evidence": g.evidence,
                "sentiment": g.sentiment,
                "sentiment_score": round(g.sentiment_score, 3),
                "sentiment_raw": round(g.sentiment_raw, 3),
                "similarity": round(g.similarity, 4),
                "confidence": round(g.confidence, 4),
                "lexical_overlap": round(g.lexical_overlap, 4),
                "confidence_margin": round(g.confidence_margin, 4),
                "highlight_score": round(abs(g.sentiment_score) * (0.6 + g.confidence), 4),
                "impact_score": round(abs(g.sentiment_score) * (0.6 + g.confidence), 4),
                "debug_top_aspects": g.debug_top_aspects or [],
            }
            for g in highlights
        ],
    }


# =========================
# Clustering (project side)
# =========================

@dataclass
class ProjectModel:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    kmeans: MiniBatchKMeans
    pca2: PCA
    feature_names: np.ndarray
    cluster_top_terms: dict[int, list[str]]
    cluster_labels: dict[int, str]


def _build_vectorizer_for_clustering() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=_normalize_text,
        lowercase=True,
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        token_pattern=r"[a-z0-9][a-z0-9'_-]{1,}",
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
    )


def _pick_k_by_silhouette(X: np.ndarray, k_min: int, k_max: int, random_state: int) -> tuple[int, float]:
    n = X.shape[0]
    if n < 3:
        return 1, 0.0

    k_min = max(2, int(k_min))
    k_max = max(k_min, int(k_max))
    k_max = min(k_max, n - 1)

    best_k = k_min
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=10, batch_size=1024)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X, labels, metric="euclidean"))
        if score > best_score:
            best_score = score
            best_k = k

    if best_score < 0:
        return k_min, 0.0
    return best_k, best_score


def _top_terms_for_cluster(tfidf_matrix, labels: np.ndarray, feature_names: np.ndarray, top_n: int = 7) -> dict[int, list[str]]:
    top_n = max(3, int(top_n))
    out: dict[int, list[str]] = {}

    for cid in sorted(set(labels.tolist())):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            out[cid] = []
            continue

        mean_vec = tfidf_matrix[idx].mean(axis=0)
        mean_arr = np.asarray(mean_vec).ravel()
        order = np.argsort(mean_arr)[::-1]

        terms: list[str] = []
        for j in order:
            if mean_arr[j] <= 0:
                break
            t = str(feature_names[j]).strip()
            if not t or t in STOPWORDS or len(t) <= 2:
                continue
            if t in terms:
                continue
            terms.append(t)
            if len(terms) >= top_n:
                break

        out[cid] = terms

    return out


def _cluster_labels_from_terms(cluster_terms: dict[int, list[str]]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for cid, terms in cluster_terms.items():
        labels[cid] = f"Cluster {cid}" if not terms else " / ".join(terms[:3])
    return labels


def train_project(
    texts: Sequence[str],
    k_min: int = 2,
    k_max: int = 12,
    svd_components: int = 120,
    random_state: int = 42,
) -> ProjectModel:
    cleaned = [str(t or "").strip() for t in texts if str(t or "").strip()]
    if not cleaned:
        raise ValueError("texts is required")
    if len(cleaned) < 2:
        raise ValueError("need at least 2 texts to cluster")

    vectorizer = _build_vectorizer_for_clustering()
    tfidf = vectorizer.fit_transform(cleaned)
    feature_names = np.array(vectorizer.get_feature_names_out())

    n_features = tfidf.shape[1]
    n_comp = int(min(max(2, svd_components), max(2, n_features - 1)))
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)

    X_red = svd.fit_transform(tfidf).astype(np.float32)
    X_red = normalize(X_red, norm="l2", axis=1, copy=False)

    if len(cleaned) < 3:
        k = 1
    else:
        k, _ = _pick_k_by_silhouette(X_red, k_min=k_min, k_max=k_max, random_state=random_state)

    if k == 1:
        kmeans = MiniBatchKMeans(n_clusters=1, random_state=random_state, n_init=1)
        kmeans.fit(X_red)
        labels = np.zeros((X_red.shape[0],), dtype=int)
    else:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=20, batch_size=1024)
        labels = kmeans.fit_predict(X_red)

    pca2 = PCA(n_components=2, random_state=random_state)
    pca2.fit(X_red)

    cluster_terms = _top_terms_for_cluster(tfidf, labels, feature_names, top_n=8)
    cluster_labels = _cluster_labels_from_terms(cluster_terms)

    return ProjectModel(
        vectorizer=vectorizer,
        svd=svd,
        kmeans=kmeans,
        pca2=pca2,
        feature_names=feature_names,
        cluster_top_terms=cluster_terms,
        cluster_labels=cluster_labels,
    )


def project_results(model: ProjectModel, texts: Sequence[str], preview_len: int = 140) -> dict[str, Any]:
    cleaned = [str(t or "").strip() for t in texts]
    tfidf = model.vectorizer.transform(cleaned)
    X_red = model.svd.transform(tfidf).astype(np.float32)
    X_red = normalize(X_red, norm="l2", axis=1, copy=False)

    labels = model.kmeans.predict(X_red)
    xy = model.pca2.transform(X_red)

    points: list[dict[str, Any]] = []
    for i, (x, y) in enumerate(xy):
        preview = cleaned[i].replace("\n", " ").strip()
        points.append(
            {
                "id": i,
                "cluster_id": int(labels[i]),
                "x": float(x),
                "y": float(y),
                "text_preview": preview[:preview_len],
            }
        )

    cluster_ids = sorted(set(labels.tolist()))
    clusters: list[dict[str, Any]] = []
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        terms = model.cluster_top_terms.get(int(cid), [])
        label = model.cluster_labels.get(int(cid), f"Cluster {cid}")
        clusters.append(
            {
                "cluster_id": int(cid),
                "cluster_label": label,
                "size": int(idx.size),
                "top_terms": terms[:10],
            }
        )

    stats = {
        "n_texts": int(len(cleaned)),
        "n_clusters": int(len(cluster_ids)),
    }

    return {"stats": stats, "points": points, "clusters": clusters}


def classify_text(model: ProjectModel, text: str, preview_len: int = 140) -> dict[str, Any]:
    t = (text or "").strip()
    if not t:
        raise ValueError("text is required")

    tfidf = model.vectorizer.transform([t])
    X_red = model.svd.transform(tfidf).astype(np.float32)
    X_red = normalize(X_red, norm="l2", axis=1, copy=False)

    dists = model.kmeans.transform(X_red).ravel()
    best = int(np.argmin(dists))
    sorted_d = np.sort(dists)
    margin = float(sorted_d[1] - sorted_d[0]) if sorted_d.size >= 2 else 0.0

    x, y = model.pca2.transform(X_red)[0]
    label = model.cluster_labels.get(best, f"Cluster {best}")

    preview = t.replace("\n", " ").strip()[:preview_len]
    return {
        "cluster_id": best,
        "cluster_label": label,
        "text_preview": preview,
        "x": float(x),
        "y": float(y),
        "confidence_margin": margin,
    }


def _resolve_eval_file() -> Path | None:
    candidates = [
        Path("/mnt/data/granulate_all_tests_20.json"),
        Path(__file__).resolve().parent.parent / "granulate_all_tests_20.json",
        Path.cwd() / "granulate_all_tests_20.json",
        Path("/mnt/data/granulate_all_tests.json"),
        Path(__file__).resolve().parent.parent / "granulate_all_tests.json",
        Path.cwd() / "granulate_all_tests.json",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def run_granulate_evaluation(test_file: str | Path | None = None) -> None:
    path = Path(test_file) if test_file is not None else _resolve_eval_file()
    if path is None or (not path.exists()):
        print("No evaluation file found (expected granulate_all_tests_20.json).")
        return

    with path.open("r", encoding="utf-8-sig") as f:
        cases = json.load(f)

    if not isinstance(cases, list) or not cases:
        print(f"No test cases available in {path}.")
        return

    taxonomy_counter: Counter[str] = Counter()
    total_granules = 0
    total_other = 0

    print(f"Loaded {len(cases)} cases from {path}")

    for idx, case in enumerate(cases, start=1):
        case_id = int(case.get("test_number", idx))
        text = str(case.get("input_text", "")).strip()
        if not text:
            continue

        out = granulate_text(text)
        granules = out.get("granules", [])
        other_count = sum(1 for g in granules if g.get("aspect") == "OTHER")

        total_granules += len(granules)
        total_other += other_count
        chosen = str(out.get("detected_taxonomy", "unknown"))
        taxonomy_counter[chosen] += 1

        other_pct = 100.0 * other_count / max(1, len(granules))
        top_candidate = (out.get("taxonomy_candidates") or [{}])[0]
        top_name = str(top_candidate.get("taxonomy", "n/a"))
        top_score = float(top_candidate.get("score", 0.0))
        print(
            f"Case {case_id}: OTHER={other_count}/{max(1, len(granules))} ({other_pct:.1f}%) "
            f"top={top_name}:{top_score:.4f} chosen={chosen}"
        )

    overall_other_pct = (100.0 * total_other / max(1, total_granules))
    print("")
    print(f"Total granules: {total_granules}")
    print(f"OTHER overall: {total_other}/{total_granules} ({overall_other_pct:.1f}%)")
    print("Detected taxonomy counts:", dict(taxonomy_counter))


if __name__ == "__main__":
    run_granulate_evaluation()


