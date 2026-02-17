from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    model_name: str
    backend: str


def _tfidf_fallback(texts: list[str]) -> EmbeddingResult:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        token_pattern=r"[a-zA-Z0-9][a-zA-Z0-9'_-]{1,}",
    )
    tfidf = vectorizer.fit_transform(texts)
    n_components = int(min(256, max(8, min(tfidf.shape[0] - 1, tfidf.shape[1] - 1))))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    vectors = svd.fit_transform(tfidf).astype(np.float32)
    vectors = normalize(vectors, norm="l2", axis=1, copy=False)
    return EmbeddingResult(vectors=vectors, model_name="tfidf_svd", backend="fallback")


def generate_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 128,
) -> EmbeddingResult:
    if not texts:
        raise ValueError("texts is required for embeddings")

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return EmbeddingResult(vectors=vectors, model_name=model_name, backend="sentence_transformers")
    except Exception:
        return _tfidf_fallback(texts)


def save_embeddings(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)
