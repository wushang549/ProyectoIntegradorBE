"""Embedding stage using sentence-transformers with robust lexical fallback."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(slots=True)
class EmbeddingResult:
    """Embedding output and metadata."""

    vectors: np.ndarray
    method: str


_MODEL_LOCK = threading.Lock()
_MODEL = None


def compute_embeddings(texts: list[str]) -> EmbeddingResult:
    """Compute deterministic embeddings for input texts."""

    if not texts:
        return EmbeddingResult(vectors=np.zeros((0, 0), dtype=np.float32), method="empty")

    try:
        vectors = _encode_with_sentence_transformers(texts)
        return EmbeddingResult(vectors=vectors, method="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        vectors = _encode_with_lexical_fallback(texts)
        return EmbeddingResult(vectors=vectors, method="tfidf-hybrid-svd-fallback")


def _encode_with_sentence_transformers(texts: list[str]) -> np.ndarray:
    """Encode texts with all-MiniLM-L6-v2."""

    global _MODEL  # pylint: disable=global-statement

    with _MODEL_LOCK:
        if _MODEL is None:
            from sentence_transformers import SentenceTransformer  # Local import for fallback safety.

            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    vectors = _MODEL.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.astype(np.float32)


def _encode_with_lexical_fallback(texts: list[str]) -> np.ndarray:
    """Fallback strategy using word+char TF-IDF with deterministic SVD compression."""

    matrices: list[sparse.csr_matrix] = []

    word_vectorizer = TfidfVectorizer(
        max_features=4096,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=4096,
        min_df=2,
        sublinear_tf=True,
    )

    for vectorizer in (word_vectorizer, char_vectorizer):
        try:
            matrix = vectorizer.fit_transform(texts)
        except Exception:
            continue
        if matrix.shape[1] > 0:
            matrices.append(matrix.tocsr())

    if not matrices:
        return np.zeros((len(texts), 1), dtype=np.float32)

    merged = matrices[0] if len(matrices) == 1 else sparse.hstack(matrices, format="csr")
    n_samples, n_features = merged.shape
    if n_samples <= 1 or n_features <= 1:
        dense = merged.toarray().astype(np.float32)
        return _l2_normalize(dense)

    n_components = min(256, n_samples - 1, n_features - 1)
    if n_components >= 8:
        try:
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(merged).astype(np.float32)
            return _l2_normalize(reduced)
        except Exception:
            pass

    dense = merged.toarray().astype(np.float32)
    return _l2_normalize(dense)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)
