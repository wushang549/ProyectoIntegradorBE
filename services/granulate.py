from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
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
}

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

# =========================
# Normalization / Tokenization
# =========================


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = _strip_accents(lowered)
    lowered = re.sub(r"[“”]", '"', lowered)
    lowered = re.sub(r"[’]", "'", lowered)
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
    "crash": -1.6,
    "crashes": -1.6,
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
}
NEG_LEXICON.update(
    {
        "wait": -1.0,
        "waiting": -1.0,
        "overwhelmed": -1.0,
        "ignored": -1.0,
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
    "no vale la pena": -2.0,
    "no funciona": -1.8,
    "sin respuesta": -1.6,
    "cargo duplicado": -1.8,
    "perdida de datos": -2.1,
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


def _simple_sentiment(unit: str, tokens: list[str]) -> tuple[str, float]:
    score = _sentiment_from_phrases(_normalize_text(unit))

    for idx, token in enumerate(tokens):
        mod = _modifier(tokens, idx)
        if token in POS_LEXICON:
            value = POS_LEXICON[token] * mod
            score += -value if _is_negated(tokens, idx) else value
        elif token in NEG_LEXICON:
            value = NEG_LEXICON[token] * mod
            score += -value if _is_negated(tokens, idx) else value

    if score > 0.65:
        return "positive", float(score)
    if score < -0.65:
        return "negative", float(score)
    return "neutral", float(score)


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
        "soporte",
        "atencion al cliente",
        "sin respuesta",
        "reembolso",
        "garantia",
    ],
    "PRICE_VALUE": [
        "price",
        "pricing",
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
        "delicious",
        "bland",
        "fresh",
        "overcooked",
        "undercooked",
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
        "wrong order",
        "servicio",
        "mesero",
        "mesera",
        "personal",
        "amable",
        "grosero",
        "nos ignoraron",
        "se equivocaron",
    ],
    "WAIT_TIME": [
        "wait time",
        "waiting",
        "long wait",
        "took forever",
        "minutes",
        "an hour",
        "line",
        "queue",
        "tiempo de espera",
        "espera",
        "tardaron",
        "minutos",
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
        "baño",
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
        "porciones pequeñas",
    ],
    "MENU_VARIETY": [
        "menu",
        "options",
        "choices",
        "limited menu",
        "specials",
        "menú",
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
    "SERVICE_SUPPORT": ["service", "support", "customer service", "help", "soporte", "servicio", "atencion"],
    "PRICE_VALUE": ["price", "pricing", "expensive", "cheap", "value", "precio", "caro", "barato", "vale la pena"],
    "OTHER": ["overall", "general", "experience", "feedback", "en general"],
}

TaxonomyName = str

TAXONOMIES: dict[TaxonomyName, dict[str, list[str]]] = {
    "restaurant": RESTAURANT_ASPECTS,
    "saas": SAAS_ASPECTS,
    "general": GENERAL_ASPECTS,
}

DEFAULT_FALLBACK_TAXONOMY: TaxonomyName = "general"

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
    similarity: float
    lexical_overlap: float
    confidence: float


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
        if not term or term in STOPWORDS or term in EVIDENCE_BLACKLIST or term in seen:
            continue
        if len(term) <= 2:
            continue
        seen.add(term)
        evidence.append(term)
        if len(evidence) >= k:
            break

    if not evidence:
        return tokens[:k]
    return evidence


def _build_aspect_summary(granules: list[Granule], aspect_names: list[str], top_evidence_k: int = 6) -> dict[str, Any]:
    top_evidence_k = max(3, int(top_evidence_k))
    out: dict[str, Any] = {}

    for asp in aspect_names:
        g_list = [g for g in granules if g.aspect == asp]
        if not g_list:
            out[asp] = {"count": 0, "avg_sentiment": 0.0, "top_evidence": []}
            continue

        avg_sent = float(sum(g.sentiment_score for g in g_list) / max(1, len(g_list)))

        freq: dict[str, int] = {}
        for g in g_list:
            for e in g.evidence:
                e_norm = _normalize_text(e)
                if not e_norm or e_norm in STOPWORDS or e_norm in EVIDENCE_BLACKLIST:
                    continue
                freq[e] = freq.get(e, 0) + 1

        top = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0])))
        out[asp] = {
            "count": int(len(g_list)),
            "avg_sentiment": round(avg_sent, 3),
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
        det = _TAX_DETECTOR.infer(raw, min_margin=0.06)
        taxonomy_candidates = det["scores"]
        detection_margin = float(det["margin"]) if det["margin"] is not None else None

        chosen = det["name"]
        if chosen is None:
            chosen = taxonomy_candidates[0]["taxonomy"] if taxonomy_candidates else DEFAULT_FALLBACK_TAXONOMY

        detected_taxonomy = str(chosen)
        aspects_map = TAXONOMIES.get(detected_taxonomy, TAXONOMIES[DEFAULT_FALLBACK_TAXONOMY])

    aspect_names = list(aspects_map.keys())
    seed_index = _make_seed_index(aspects_map)

    seed_docs, seed_doc_aspects = _build_seed_docs(aspects_map)
    vectorizer = _build_vectorizer_for_granulate()

    matrix = vectorizer.fit_transform(seed_docs + units)

    seed_vecs = normalize(matrix[: len(seed_docs)], norm="l2", axis=1, copy=False)
    unit_vecs = normalize(matrix[len(seed_docs) :], norm="l2", axis=1, copy=False)

    min_similarity = float(min_similarity)

    granules: list[Granule] = []

    for i, unit in enumerate(units):
        tokens = _tokenize(unit)
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

        lexical_weight = 0.30

        for asp in aspect_names:
            if asp == "OTHER":
                continue
            cos = float(best_cos_by_aspect.get(asp, 0.0))
            lex = float(lexical_scores.get(asp, 0.0))
            conf = (1.0 - lexical_weight) * cos + lexical_weight * lex

            if conf > best_conf:
                best_conf = conf
                best_aspect = asp
                best_cos = cos
                best_lex = lex

        if best_cos < min_similarity and best_conf < max(min_similarity, 0.16):
            best_aspect = "OTHER"

        sentiment_label, sentiment_score = _simple_sentiment(unit, tokens)
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
                similarity=float(best_cos),
                lexical_overlap=float(best_lex),
                confidence=float(best_conf),
            )
        )

    aspect_summary = _build_aspect_summary(granules, aspect_names, top_evidence_k=min(8, max(3, int(top_k_evidence))))
    highlights = sorted(granules, key=lambda g: g.confidence, reverse=True)[:3]

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
                "lexical_overlap": round(g.lexical_overlap, 4),
                "confidence": round(g.confidence, 4),
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
                "sentiment": g.sentiment,
                "sentiment_score": round(g.sentiment_score, 3),
                "confidence": round(g.confidence, 4),
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
