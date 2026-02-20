from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import inspect
from pathlib import Path
import re

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HEXISH_RE = re.compile(r"\b[0-9a-fA-F]{8,}\b")
LONG_NUM_RE = re.compile(r"\b\d{5,}\b")
ORDER_NUMBER_RE = re.compile(
    r"\b(?:order|pedido)\s*(?:number|num(?:ber)?|no|#)?\s*[:#-]?\s*[A-Za-z0-9-]{3,}\b",
    re.IGNORECASE,
)
PRODUCT_NUMBER_RE = re.compile(
    r"\b(?:product|producto)\s*(?:number|num(?:ber)?|no|#)?\s*[:#-]?\s*[A-Za-z0-9-]{3,}\b",
    re.IGNORECASE,
)
PRODUCT_CODE_RE = re.compile(r"\b[A-Z]{2,}(?:[-_][A-Z0-9]{2,})+\b")
ALNUM_CODE_RE = re.compile(r"\b[A-Za-z]{1,4}[-_ ]?\d{3,}\b")
GREETING_PREFIX_RE = re.compile(
    r"^\s*(?:hello|hi|hey|good morning|good afternoon|good evening)[,!\.\s-]*",
    re.IGNORECASE,
)
CALL_OPENER_RE = re.compile(
    r"^\s*(?:i am|i'm|im)\s+(?:calling|reaching out|writing)\b(?:\s+to)?\s*",
    re.IGNORECASE,
)
INTRO_PREFIX_RE = re.compile(
    r"^\s*(?:this is|my name is)\s+[a-z]+(?:\s+[a-z]+){0,2}[,!\.\s-]*",
    re.IGNORECASE,
)
SELF_NAME_RE = re.compile(
    r"\b(?:this is|my name is)\s+[a-z]+(?:\s+[a-z]+){0,2}\b",
    re.IGNORECASE,
)
GENERIC_OPENERS = (
    "i just wanted to",
    "i wanted to",
    "i am calling to",
    "i'm calling to",
    "i am writing to",
    "i'm writing to",
    "thanks for your help",
    "thank you for your help",
)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{1,}")


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    model_name: str
    backend: str


def clean_text_for_embeddings(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        return ""

    clean = source
    clean = EMAIL_RE.sub(" ", clean)
    clean = URL_RE.sub(" ", clean)
    clean = ORDER_NUMBER_RE.sub(" ", clean)
    clean = PRODUCT_NUMBER_RE.sub(" ", clean)
    clean = PRODUCT_CODE_RE.sub(" ", clean)
    clean = ALNUM_CODE_RE.sub(" ", clean)
    clean = HEXISH_RE.sub(" ", clean)
    clean = LONG_NUM_RE.sub(" ", clean)

    # Trim opening boilerplate aggressively only near the start.
    prefix = clean[:220]
    remainder = clean[220:]
    prefix = GREETING_PREFIX_RE.sub("", prefix)
    prefix = CALL_OPENER_RE.sub("", prefix)
    prefix = INTRO_PREFIX_RE.sub("", prefix)
    for phrase in GENERIC_OPENERS:
        prefix = re.sub(rf"^\s*{re.escape(phrase)}\b[:,\s-]*", "", prefix, flags=re.IGNORECASE)
    clean = f"{prefix} {remainder}".strip()
    clean = SELF_NAME_RE.sub(" ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    if not clean and any(ch.isalpha() for ch in source):
        alpha_only = re.sub(r"[^A-Za-z\s]", " ", source)
        alpha_only = re.sub(r"\s+", " ", alpha_only).strip()
        return alpha_only
    return clean


def clean_texts_for_embeddings(texts: list[str]) -> list[str]:
    cleaned: list[str] = []
    for text in texts:
        normalized = clean_text_for_embeddings(text)
        if len(TOKEN_RE.findall(normalized)) < 2:
            # Keep semantic content by falling back to lightly normalized source.
            fallback = re.sub(r"\s+", " ", str(text or "")).strip()
            cleaned.append(fallback)
        else:
            cleaned.append(normalized)
    return cleaned


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import normalize

    if vectors.size == 0:
        return vectors.astype(np.float32, copy=False)
    return normalize(vectors, norm="l2", axis=1, copy=False).astype(np.float32, copy=False)


def _tfidf_fallback(texts: list[str], normalize: bool = True) -> EmbeddingResult:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        token_pattern=r"[a-zA-Z0-9][a-zA-Z0-9'_-]{1,}",
    )
    try:
        tfidf = vectorizer.fit_transform(texts)
    except Exception:
        vectors = np.ones((len(texts), 1), dtype=np.float32)
        if normalize:
            vectors = _l2_normalize(vectors)
        return EmbeddingResult(vectors=vectors, model_name="tfidf_dense", backend="fallback")

    n_samples, n_features = tfidf.shape
    max_components = min(n_samples - 1, n_features - 1)
    if max_components < 2:
        vectors = tfidf.toarray().astype(np.float32, copy=False)
        if vectors.shape[1] == 0:
            vectors = np.ones((n_samples, 1), dtype=np.float32)
        if normalize:
            vectors = _l2_normalize(vectors)
        return EmbeddingResult(vectors=vectors, model_name="tfidf_dense", backend="fallback")

    n_components = int(min(256, max_components))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    vectors = svd.fit_transform(tfidf).astype(np.float32)
    if normalize:
        vectors = _l2_normalize(vectors)
    return EmbeddingResult(vectors=vectors, model_name="tfidf_svd", backend="fallback")


@lru_cache(maxsize=2)
def _get_st_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def generate_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 128,
    normalize: bool = True,
    device: str | None = None,
) -> EmbeddingResult:
    if not texts:
        raise ValueError("texts is required for embeddings")

    try:
        model = _get_st_model(model_name)
        encode_kwargs = {
            "batch_size": max(1, int(batch_size)),
            "show_progress_bar": False,
            "normalize_embeddings": bool(normalize),
            "convert_to_numpy": True,
        }
        if device:
            try:
                if "device" in inspect.signature(model.encode).parameters:
                    encode_kwargs["device"] = device
            except Exception:
                pass
        vectors = model.encode(
            texts,
            **encode_kwargs,
        ).astype(np.float32)
        return EmbeddingResult(vectors=vectors, model_name=model_name, backend="sentence_transformers")
    except Exception:
        return _tfidf_fallback(texts, normalize=normalize)


def save_embeddings(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)
