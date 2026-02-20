from __future__ import annotations

import hashlib
import json
import re
import socket
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np

from services.embeddings import clean_text_for_embeddings

OLLAMA_CIRCUIT_BREAK_KEY = "ollama_disabled"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"
PROMPT_VERSION = 2
WORD_RE = re.compile(r"[A-Za-z0-9]+")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HEXISH_RE = re.compile(r"\b[0-9a-fA-F]{8,}\b")
LONG_NUM_RE = re.compile(r"\b\d{5,}\b")
ORDER_SKU_RE = re.compile(r"\b(?:order|sku|product|ticket|case)\s*[:#-]?\s*[A-Za-z0-9-]{3,}\b", re.IGNORECASE)
ALNUM_CODE_RE = re.compile(r"\b[A-Za-z]{1,5}[-_ ]?\d{2,}\b")
SPACE_RE = re.compile(r"\s+")

DEFAULT_LABEL_TAXONOMY = [
    "Order Delay",
    "Billing Problem",
    "Shipping Issue",
    "Return Request",
    "Login Failure",
    "App Crash",
    "Slow Performance",
    "Support Escalation",
    "Product Defect",
    "Purchase Inquiry",
]
BROAD_NODE_BUCKETS = ["General Issues", "Product Issues", "Service Issues", "Other"]
GENERIC_LABELS = {"Topic", "Other", "General Issues", "Product Issues", "Service Issues"}
GENERIC_VERBS = {
    "does",
    "do",
    "can",
    "need",
    "want",
    "require",
    "possible",
    "working",
    "works",
    "purchased",
    "purchase",
    "bought",
    "buy",
    "like",
    "looking",
    "interested",
    "help",
    "ask",
    "tell",
    "say",
    "calling",
    "inquire",
    "question",
    "questions",
}
CALL_CENTER_STOP_TERMS = {
    "calling",
    "call",
    "inquire",
    "inquiry",
    "question",
    "questions",
    "help",
    "please",
    "order",
    "number",
    "product",
    "thanks",
    "thank",
    "hello",
    "hi",
    "good",
    "morning",
    "afternoon",
    "evening",
    "name",
    "customer",
    "guys",
}
STOP_PHRASES = {
    "order number",
    "product number",
    "my name is",
    "this is",
    "good morning",
    "good afternoon",
    "good evening",
    "thank you",
    "thanks for",
    "calling to",
    "just calling",
    "hello this",
}
LABEL_BLACKLIST = {
    "calling",
    "does",
    "hello",
    "hi",
    "possible",
    "working",
    "purchased",
    "purchase",
    "business",
    "issue",
    "problem",
    "topic",
    "other",
    "question",
    "questions",
}
SPECIFIC_SINGLE_LABEL_ALLOWLIST = {
    "refund",
    "returns",
    "billing",
    "shipping",
    "delivery",
    "outage",
    "warranty",
    "login",
    "crashes",
    "performance",
    "subscription",
    "router",
    "laptop",
    "tablet",
    "speaker",
    "monitor",
    "keyboard",
    "microwave",
    "oven",
    "dishwasher",
    "refrigerator",
    "television",
    "camera",
    "software",
    "server",
    "device",
    "printer",
}
FALLBACK_ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "i",
    "you",
    "we",
    "they",
    "this",
    "those",
    "these",
    "my",
    "your",
    "our",
    "their",
}


def _load_cache(path: Path) -> dict[str, str]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            return {}
    return {}


def _save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


@lru_cache(maxsize=1)
def _english_stop_words() -> set[str]:
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return set(str(w).lower() for w in ENGLISH_STOP_WORDS)
    except Exception:
        return set(FALLBACK_ENGLISH_STOPWORDS)


@lru_cache(maxsize=1)
def get_label_stop_words() -> set[str]:
    return _english_stop_words() | CALL_CENTER_STOP_TERMS | GENERIC_VERBS


def sanitize_text_for_llm(text: str) -> str:
    source = str(text or "")
    clean = clean_text_for_embeddings(source)
    clean = EMAIL_RE.sub(" ", clean)
    clean = URL_RE.sub(" ", clean)
    clean = ORDER_SKU_RE.sub(" ", clean)
    clean = ALNUM_CODE_RE.sub(" ", clean)
    clean = HEXISH_RE.sub(" ", clean)
    clean = LONG_NUM_RE.sub(" ", clean)
    clean = SPACE_RE.sub(" ", clean).strip()
    if clean:
        return clean

    if any(ch.isalpha() for ch in source):
        alpha_fallback = re.sub(r"[^A-Za-z\s]", " ", source)
        alpha_fallback = SPACE_RE.sub(" ", alpha_fallback).strip()
        if alpha_fallback:
            return alpha_fallback
        letters_only = "".join(ch for ch in source if ch.isalpha())
        if letters_only:
            return letters_only[:64]
    return clean


def _normalize_candidate(label: str) -> str:
    cleaned = str(label or "")
    cleaned = cleaned.replace("`", " ").replace('"', " ").replace("'", " ")
    cleaned = re.sub(r"[^A-Za-z0-9\s-]", " ", cleaned)
    cleaned = SPACE_RE.sub(" ", cleaned).strip()
    if not cleaned:
        return ""
    words = WORD_RE.findall(cleaned)
    if not words:
        return ""
    return " ".join(words[:3]).title()


def _text_tokens(text: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_RE.findall(str(text or "")) if tok and any(ch.isalpha() for ch in tok)}


def _has_signal(text: str) -> bool:
    s = sanitize_text_for_llm(text)
    if len(s) < 3:
        return False
    return any(ch.isalpha() for ch in s)


def _is_generic_or_boilerplate_label(label: str, force_broad: bool) -> bool:
    low = label.strip().lower()
    if not low:
        return True
    if low in LABEL_BLACKLIST:
        return not (force_broad and low == "other")
    if low in {"topic", "general", "misc"}:
        return True
    if low in GENERIC_VERBS:
        return True
    if low in get_label_stop_words():
        return True
    return False


def _is_specific_single_word(label: str) -> bool:
    low = label.strip().lower()
    return low in SPECIFIC_SINGLE_LABEL_ALLOWLIST


def is_informative_term(term: str) -> bool:
    candidate = str(term or "").strip().lower()
    candidate = SPACE_RE.sub(" ", candidate)
    if not candidate:
        return False
    if candidate in STOP_PHRASES:
        return False

    tokens = [tok for tok in TOKEN_RE.findall(candidate) if tok]
    if not tokens:
        return False

    stop_words = get_label_stop_words()
    if all(tok in stop_words for tok in tokens):
        return False
    if len(tokens) == 1:
        tok = tokens[0]
        if tok in stop_words or tok in LABEL_BLACKLIST or tok in GENERIC_VERBS:
            return False
        if tok not in SPECIFIC_SINGLE_LABEL_ALLOWLIST:
            return False
    else:
        informative_tokens = [tok for tok in tokens if tok not in stop_words and tok not in GENERIC_VERBS]
        if len(informative_tokens) < 2:
            return False
        if tokens[0] in stop_words or tokens[-1] in stop_words:
            return False
    if any(ch.isdigit() for ch in candidate):
        return False
    if any(phrase in candidate for phrase in STOP_PHRASES):
        return False
    return True


def _sample_texts_for_keywords(texts: list[str], max_docs: int = 140) -> list[str]:
    docs = [sanitize_text_for_llm(t) for t in texts if _has_signal(t)]
    if len(docs) <= max_docs:
        return docs
    step = max(1, len(docs) // max_docs)
    sampled = [docs[i] for i in range(0, len(docs), step)]
    return sampled[:max_docs]


def _extract_top_keywords(texts: list[str], top_n: int = 12) -> list[str]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = _sample_texts_for_keywords(texts)
        if not docs:
            return []
        min_df = 2 if len(docs) >= 10 else 1
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words=sorted(get_label_stop_words()),
                ngram_range=(1, 2),
                max_features=8000,
                min_df=min_df,
            )
            mat = vectorizer.fit_transform(docs)
        except ValueError:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words=sorted(get_label_stop_words()),
                ngram_range=(1, 2),
                max_features=8000,
                min_df=1,
            )
            mat = vectorizer.fit_transform(docs)

        scores = mat.mean(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        if scores.size == 0:
            return []

        order = np.argsort(scores)[::-1]
        out: list[str] = []
        seen: set[str] = set()
        for idx in order:
            base_score = float(scores[int(idx)])
            if base_score <= 0:
                break
            term = str(terms[int(idx)]).strip().lower()
            if not is_informative_term(term):
                continue
            score = base_score * (1.25 if " " in term else 1.0)
            if score <= 0:
                continue
            signature = term.replace(" ", "")
            if signature in seen:
                continue
            seen.add(signature)
            out.append(term)
            if len(out) >= top_n:
                break
        return out
    except Exception:
        return []


def _jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def _select_representative_items(texts: list[str], embeddings: np.ndarray | None, max_items: int = 5) -> list[str]:
    cleaned = [sanitize_text_for_llm(t) for t in texts]
    indexed = [(i, txt) for i, txt in enumerate(cleaned) if _has_signal(txt)]
    if not indexed:
        return []

    selected_texts: list[str] = []
    selected_tokens: list[set[str]] = []

    def _push_if_diverse(text: str) -> bool:
        tokens = _text_tokens(text)
        if not tokens:
            return False
        if any(_jaccard(tokens, prev) >= 0.82 for prev in selected_tokens):
            return False
        selected_tokens.append(tokens)
        selected_texts.append(text[:280])
        return True

    if embeddings is not None and isinstance(embeddings, np.ndarray) and embeddings.shape[0] == len(texts):
        try:
            idx = np.array([i for i, _ in indexed], dtype=int)
            vecs = embeddings[idx]
            centroid = vecs.mean(axis=0)
            dists = np.linalg.norm(vecs - centroid, axis=1)
            order = np.argsort(dists)
            for pos in order:
                txt = indexed[int(pos)][1]
                _push_if_diverse(txt)
                if len(selected_texts) >= max_items:
                    break
            if selected_texts:
                return selected_texts
        except Exception:
            pass

    indexed.sort(key=lambda row: len(row[1]), reverse=True)
    for _, txt in indexed:
        _push_if_diverse(txt)
        if len(selected_texts) >= max_items:
            break
    return selected_texts


def build_evidence_pack(
    texts: list[str],
    embeddings: np.ndarray | None = None,
    sibling_texts: list[str] | None = None,
    keyword_limit: int = 12,
    representative_limit: int = 5,
) -> dict[str, list[str]]:
    keyword_limit = int(max(10, min(20, keyword_limit)))
    representative_limit = int(max(5, min(10, representative_limit)))
    top_keywords = _extract_top_keywords(texts, top_n=keyword_limit)
    representative_items = _select_representative_items(texts, embeddings=embeddings, max_items=representative_limit)

    sibling_negative_keywords: list[str] = []
    if sibling_texts:
        sibling_terms = _extract_top_keywords(sibling_texts, top_n=10)
        own = set(top_keywords)
        for term in sibling_terms:
            if term in own or term in sibling_negative_keywords:
                continue
            sibling_negative_keywords.append(term)
            if len(sibling_negative_keywords) >= 5:
                break

    return {
        "top_keywords": top_keywords,
        "representative_items": representative_items,
        "sibling_negative_keywords": sibling_negative_keywords,
    }


def _keyword_to_candidate(keyword: str) -> str:
    term = sanitize_text_for_llm(keyword)
    term = SPACE_RE.sub(" ", term).strip()
    return _normalize_candidate(term)


def _is_candidate_label_ok(label: str, force_broad: bool = False) -> bool:
    if not label:
        return False
    if _is_generic_or_boilerplate_label(label, force_broad=force_broad):
        return False
    words = label.split()
    if len(words) == 1 and not _is_specific_single_word(label):
        return False
    return True


def fallback_tfidf_label(snippets: list[str]) -> str:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = [sanitize_text_for_llm(s) for s in snippets if _has_signal(s)]
        if not docs:
            return "Customer Request"

        min_df = 2 if len(docs) >= 8 else 1
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words=sorted(get_label_stop_words()),
                ngram_range=(1, 2),
                max_features=5000,
                min_df=min_df,
            )
            mat = vectorizer.fit_transform(docs)
        except ValueError:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words=sorted(get_label_stop_words()),
                ngram_range=(1, 2),
                max_features=5000,
                min_df=1,
            )
            mat = vectorizer.fit_transform(docs)

        scores = mat.mean(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        if scores.size == 0:
            return "Customer Request"

        order = np.argsort(scores)[::-1]
        for idx in order:
            if float(scores[int(idx)]) <= 0:
                break
            term = str(terms[int(idx)]).strip()
            if not is_informative_term(term):
                continue
            candidate = _normalize_candidate(term)
            if not _is_candidate_label_ok(candidate):
                continue
            return candidate
        return "Customer Request"
    except Exception:
        return "Customer Request"


def build_candidate_labels(
    top_keywords: list[str],
    snippets: list[str],
    taxonomy: list[str] | None = None,
    extra_candidates: list[str] | None = None,
    min_candidates: int = 8,
    max_candidates: int = 16,
) -> list[str]:
    taxonomy_values = taxonomy or DEFAULT_LABEL_TAXONOMY
    out: list[str] = []
    seen: set[str] = set()

    def _add(raw: str, allow_generic: bool = False) -> None:
        cand = _normalize_candidate(raw)
        if not cand:
            return
        if not allow_generic and not _is_candidate_label_ok(cand):
            return
        key = cand.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(cand)

    # Prefer informative bi-grams first.
    for kw in top_keywords:
        if " " not in kw:
            continue
        _add(_keyword_to_candidate(kw))
        if len(out) >= max_candidates:
            return out[:max_candidates]
    for kw in top_keywords:
        _add(_keyword_to_candidate(kw))
        if len(out) >= max_candidates:
            return out[:max_candidates]

    _add(fallback_tfidf_label(snippets))

    for cand in (extra_candidates or []):
        _add(cand, allow_generic=True)
        if len(out) >= max_candidates:
            return out[:max_candidates]

    for cand in taxonomy_values:
        _add(cand, allow_generic=True)
        if len(out) >= max_candidates:
            return out[:max_candidates]

    if len(out) < min_candidates:
        for cand in taxonomy_values:
            _add(cand, allow_generic=True)
            if len(out) >= min_candidates:
                break
    return out[:max_candidates]


def keywords_are_diverse(top_keywords: list[str]) -> bool:
    cleaned = [k.strip().lower() for k in top_keywords if k and k.strip()]
    if len(cleaned) < 8:
        return False
    first_tokens = {chunk.split()[0] for chunk in cleaned if chunk}
    return len(first_tokens) >= 6


def _build_candidate_prompt(
    candidates: list[str],
    top_keywords: list[str],
    representative_items: list[str],
    sibling_negative_keywords: list[str],
    prefer_broad: bool,
    strict: bool = False,
) -> str:
    strategy_hint = "Prefer a broad umbrella theme." if prefer_broad else "Prefer the most specific stable theme."
    strict_hint = (
        "STRICT: Avoid greeting/opening verbs and generic words. Use a concrete 2-3 word noun phrase."
        if strict
        else "Prefer 2-3 words. Use 1 word only if highly specific."
    )
    reps = representative_items[:5]
    return (
        "You are labeling a text cluster.\n"
        "Select one label from candidates only.\n"
        "Return valid JSON only with this exact schema:\n"
        '{"label":"<candidate>","confidence":0.0,"why":["reason1","reason2"]}\n'
        "Rules:\n"
        "- label must be EXACTLY one candidate.\n"
        "- label must be Title Case with 1-3 words.\n"
        "- avoid greetings, openers, filler verbs, and generic labels.\n"
        f"- {strict_hint}\n"
        f"- {strategy_hint}\n\n"
        f"Candidates: {json.dumps(candidates, ensure_ascii=False)}\n"
        f"Top Keywords: {json.dumps(top_keywords[:16], ensure_ascii=False)}\n"
        f"Sibling Negative Keywords: {json.dumps(sibling_negative_keywords[:5], ensure_ascii=False)}\n"
        "Representative Items:\n"
        + "\n".join(f"- {item[:240]}" for item in reps)
        + "\n\nJSON:"
    )


def _ollama_generate(prompt: str, timeout_sec: int = 20) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": str(prompt or "").strip(),
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 96},
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(OLLAMA_URL, method="POST", data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    return str(parsed.get("response", "") or "")


def _parse_candidate_response(raw_text: str) -> tuple[str, float | None]:
    body = str(raw_text or "").strip()
    if not body:
        return "", None
    if not body.startswith("{"):
        left = body.find("{")
        right = body.rfind("}")
        if left >= 0 and right > left:
            body = body[left : right + 1]
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        return "", None

    label = _normalize_candidate(str(parsed.get("label", "") or ""))
    confidence_raw = parsed.get("confidence")
    confidence: float | None = None
    if isinstance(confidence_raw, (float, int)):
        confidence = float(confidence_raw)
    elif isinstance(confidence_raw, str):
        try:
            confidence = float(confidence_raw.strip())
        except ValueError:
            confidence = None
    return label, confidence


def _contains_suspicious_code(label: str, top_keywords: list[str]) -> bool:
    suspect_pattern = re.compile(r"\b[a-z]{1,4}\s*\d{2,}\b", re.IGNORECASE)
    lowered_keywords = " ".join(top_keywords).lower()
    compact_keywords = lowered_keywords.replace(" ", "")
    for match in suspect_pattern.finditer(label.lower()):
        token = match.group(0).strip()
        compact_token = token.replace(" ", "")
        if token not in lowered_keywords and compact_token not in compact_keywords:
            return True
    return False


def _has_keyword_overlap(label: str, top_keywords: list[str]) -> bool:
    label_tokens = _text_tokens(label)
    keyword_tokens = _text_tokens(" ".join(top_keywords))
    if not label_tokens or not keyword_tokens:
        return False
    return bool(label_tokens & keyword_tokens)


def _is_valid_llm_label(
    label: str,
    candidates: list[str],
    top_keywords: list[str],
    allow_generic_no_overlap: bool,
) -> str:
    normalized = _normalize_candidate(label)
    if not normalized:
        return ""

    candidate_map = {_normalize_candidate(c).lower(): _normalize_candidate(c) for c in candidates if _normalize_candidate(c)}
    canon = candidate_map.get(normalized.lower())
    if not canon:
        return ""

    alpha_count = sum(ch.isalpha() for ch in canon)
    if alpha_count < 3:
        return ""

    alnum_count = sum(ch.isalnum() for ch in canon)
    if alnum_count <= 0:
        return ""
    alpha_ratio = alpha_count / max(1, alnum_count)
    if alpha_ratio < 0.55:
        return ""

    word_count = len(canon.split())
    if word_count > 3:
        return ""
    if word_count == 1 and not _is_specific_single_word(canon):
        return ""

    if _is_generic_or_boilerplate_label(canon, force_broad=allow_generic_no_overlap):
        return ""

    if _contains_suspicious_code(canon, top_keywords):
        return ""

    if not _has_keyword_overlap(canon, top_keywords):
        if not (allow_generic_no_overlap and canon in GENERIC_LABELS):
            return ""
    return canon


def _budget_remaining(budget: dict[str, int] | None) -> int | None:
    if budget is None:
        return None
    raw_remaining = budget.get("remaining", 0)
    try:
        return int(raw_remaining)
    except (TypeError, ValueError):
        return 0


def _consume_budget_attempt(budget: dict[str, int] | None) -> None:
    remaining = _budget_remaining(budget)
    if budget is None or remaining is None:
        return
    budget["remaining"] = max(0, remaining - 1)


def _disable_llm_for_budget(budget: dict[str, int] | None) -> None:
    if budget is None:
        return
    budget["remaining"] = 0
    budget[OLLAMA_CIRCUIT_BREAK_KEY] = 1


def generate_label(
    snippets: list[str],
    cache_path: Path,
    prompt_tag: str = "default",
    top_keywords: list[str] | None = None,
    sibling_negative_keywords: list[str] | None = None,
    candidates: list[str] | None = None,
    taxonomy: list[str] | None = None,
    force_broad: bool = False,
    budget: dict[str, int] | None = None,
) -> str:
    usable = [sanitize_text_for_llm(s)[:300] for s in snippets if _has_signal(s)]
    if not usable:
        return "Customer Request"

    kw = [sanitize_text_for_llm(k).lower() for k in (top_keywords or []) if _has_signal(k)]
    kw = [k for k in kw if is_informative_term(k)]
    if not kw:
        kw = _extract_top_keywords(usable, top_n=14)
    kw = kw[:16]

    negatives = [sanitize_text_for_llm(k).lower() for k in (sibling_negative_keywords or []) if _has_signal(k)]
    negatives = [k for k in negatives if is_informative_term(k)][:5]
    final_candidates = build_candidate_labels(
        top_keywords=kw,
        snippets=usable,
        taxonomy=taxonomy,
        extra_candidates=candidates,
        min_candidates=8,
        max_candidates=18,
    )
    if not final_candidates:
        final_candidates = ["Customer Request", "Billing Problem", "Shipping Issue"]

    cache = _load_cache(cache_path)
    evidence_hash = hashlib.sha256("||".join(usable).encode("utf-8")).hexdigest()
    key_payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt_version": PROMPT_VERSION,
        "prompt_tag": prompt_tag,
        "evidence_hash": evidence_hash,
        "snippets": usable[:8],
        "top_keywords": kw,
        "negative_keywords": negatives,
        "candidates": final_candidates,
        "force_broad": bool(force_broad),
    }
    cache_key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    label = ""
    remaining = _budget_remaining(budget)
    budget_disables_ollama = bool((budget or {}).get(OLLAMA_CIRCUIT_BREAK_KEY, 0))
    should_attempt_ollama = not budget_disables_ollama and (remaining is None or remaining > 0)
    if should_attempt_ollama:
        _consume_budget_attempt(budget)
        try:
            prompt = _build_candidate_prompt(
                candidates=final_candidates,
                top_keywords=kw,
                representative_items=usable[:5],
                sibling_negative_keywords=negatives,
                prefer_broad=bool(force_broad),
                strict=False,
            )
            raw_response = _ollama_generate(prompt)
            parsed_label, confidence = _parse_candidate_response(raw_response)
            if confidence is not None and confidence < 0.35:
                parsed_label = ""
            label = _is_valid_llm_label(
                parsed_label,
                candidates=final_candidates,
                top_keywords=kw,
                allow_generic_no_overlap=bool(force_broad),
            )
            if not label:
                strict_prompt = _build_candidate_prompt(
                    candidates=final_candidates,
                    top_keywords=kw,
                    representative_items=usable[:5],
                    sibling_negative_keywords=negatives,
                    prefer_broad=bool(force_broad),
                    strict=True,
                )
                strict_raw = _ollama_generate(strict_prompt)
                strict_label, strict_conf = _parse_candidate_response(strict_raw)
                if strict_conf is not None and strict_conf < 0.35:
                    strict_label = ""
                label = _is_valid_llm_label(
                    strict_label,
                    candidates=final_candidates,
                    top_keywords=kw,
                    allow_generic_no_overlap=bool(force_broad),
                )
        except (
            TimeoutError,
            socket.timeout,
            error.HTTPError,
            error.URLError,
            json.JSONDecodeError,
            ValueError,
            IndexError,
            KeyError,
            TypeError,
        ):
            _disable_llm_for_budget(budget)
            label = ""

    if not label:
        fallback_inputs = usable + kw[:4]
        label = fallback_tfidf_label(fallback_inputs)

    label = _normalize_candidate(label) or "Customer Request"
    if _is_generic_or_boilerplate_label(label, force_broad=bool(force_broad)):
        label = fallback_tfidf_label(usable + kw[:4])
        label = _normalize_candidate(label) or "Customer Request"
    cache[cache_key] = label
    _save_cache(cache_path, cache)
    return label
