# app/services/ml.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TrainOutput:
    points: list[dict[str, Any]]
    clusters: list[dict[str, Any]]
    labels: dict[int, str]


def train_and_save(
    texts: list[str],
    project_dir: Path,
    n_clusters: int = 8,
    max_features: int = 25000,
) -> TrainOutput:
    if len(texts) < 5:
        raise ValueError("Need at least 5 texts to cluster")

    # Keep clusters reasonable for small datasets
    n_clusters = max(2, min(int(n_clusters), max(2, len(texts) // 2)))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=int(max_features),
        ngram_range=(1, 2),
        min_df=2 if len(texts) >= 50 else 1,
    )

    X = vectorizer.fit_transform(texts)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=1024,
        n_init="auto",
    )
    cluster_ids = kmeans.fit_predict(X)

    # 2D projection for plotting
    svd = TruncatedSVD(n_components=2, random_state=42)
    X2 = svd.fit_transform(X)

    terms = np.array(vectorizer.get_feature_names_out())
    cluster_labels: dict[int, str] = {}
    clusters: list[dict[str, Any]] = []

    for cid in range(n_clusters):
        idx = np.where(cluster_ids == cid)[0]
        size = int(idx.size)

        centroid = kmeans.cluster_centers_[cid]
        top_idx = np.argsort(centroid)[-10:][::-1]
        top_terms = terms[top_idx].tolist() if terms.size else []

        label = top_terms[0] if top_terms else f"Cluster {cid}"
        cluster_labels[int(cid)] = label

        clusters.append(
            {
                "cluster_id": int(cid),
                "label": label,
                "size": size,
                "top_terms": top_terms,
            }
        )

    points: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        preview = text if len(text) <= 140 else text[:137] + "..."
        points.append(
            {
                "x": float(round(X2[i, 0], 6)),
                "y": float(round(X2[i, 1], 6)),
                "cluster_id": int(cluster_ids[i]),
                "text_preview": preview,
            }
        )

    joblib.dump(vectorizer, project_dir / "vectorizer.joblib")
    joblib.dump(kmeans, project_dir / "kmeans.joblib")
    joblib.dump(svd, project_dir / "svd2d.joblib")
    joblib.dump(cluster_labels, project_dir / "cluster_labels.joblib")

    return TrainOutput(points=points, clusters=clusters, labels=cluster_labels)


def _load_artifacts(project_dir: Path):
    vectorizer = joblib.load(project_dir / "vectorizer.joblib")
    kmeans = joblib.load(project_dir / "kmeans.joblib")
    svd = joblib.load(project_dir / "svd2d.joblib")
    labels = joblib.load(project_dir / "cluster_labels.joblib")
    return vectorizer, kmeans, svd, labels


def predict_one(text: str, project_dir: Path) -> dict[str, Any]:
    vectorizer, kmeans, svd, labels = _load_artifacts(project_dir)

    X = vectorizer.transform([text])
    cid = int(kmeans.predict(X)[0])

    dists = kmeans.transform(X)[0]
    dsorted = np.sort(dists)
    margin = float(dsorted[1] - dsorted[0]) if dsorted.size >= 2 else 0.0

    xy = svd.transform(X)[0]
    preview = text if len(text) <= 140 else text[:137] + "..."

    return {
        "cluster_id": cid,
        "cluster_label": labels.get(cid, f"Cluster {cid}"),
        "text_preview": preview,
        "x": float(round(float(xy[0]), 6)),
        "y": float(round(float(xy[1]), 6)),
        "confidence_margin": float(round(margin, 6)),
    }
