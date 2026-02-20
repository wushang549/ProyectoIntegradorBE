from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import inspect
from pathlib import Path

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    model_name: str
    backend: str


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
