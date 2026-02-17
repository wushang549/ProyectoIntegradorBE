from __future__ import annotations

import numpy as np


def project_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    if embeddings.shape[0] <= 1:
        return np.zeros((embeddings.shape[0], 2), dtype=np.float32)

    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=max(2, min(int(n_neighbors), max(2, embeddings.shape[0] - 1))),
            min_dist=float(min_dist),
            metric="cosine",
            random_state=random_state,
            transform_seed=random_state,
        )
        out = reducer.fit_transform(embeddings)
        return out.astype(np.float32)
    except Exception:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=random_state)
        out = pca.fit_transform(embeddings)
        return out.astype(np.float32)
