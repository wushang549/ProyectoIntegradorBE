from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from services.embeddings import generate_embeddings, save_embeddings
from services.granulate import granulate_text
from services.hierarchy import build_hierarchy
from services.labeling import fallback_tfidf_label, generate_label
from services.storage import read_json, write_json, write_status
from services.umap_service import project_umap
from services.validation import sanitize_preview


def _sample_indices(n_items: int, max_rows: int) -> list[int]:
    if n_items <= max_rows:
        return list(range(n_items))
    rng = random.Random(42)
    return sorted(rng.sample(range(n_items), k=max_rows))


def _cluster_top_terms(texts: list[str], labels: list[int], top_n: int = 8) -> dict[int, list[str]]:
    try:
        vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=10000)
        mat = vec.fit_transform(texts)
        terms = vec.get_feature_names_out()
    except Exception:
        return {int(cid): [] for cid in sorted(set(labels))}
    out: dict[int, list[str]] = {}
    labels_arr = np.array(labels, dtype=int)
    for cid in sorted(set(labels)):
        idx = np.where(labels_arr == cid)[0]
        if idx.size == 0:
            out[cid] = []
            continue
        scores = mat[idx].mean(axis=0).A1
        best = np.argsort(scores)[::-1]
        picked: list[str] = []
        for j in best:
            if scores[j] <= 0:
                break
            t = str(terms[j]).strip()
            if not t or t in picked:
                continue
            picked.append(t)
            if len(picked) >= top_n:
                break
        out[cid] = picked
    return out


def _cluster_representatives(
    embeddings: np.ndarray,
    labels: list[int],
    item_ids: list[str],
    previews: list[str],
    per_cluster: int = 5,
) -> dict[int, list[dict[str, str]]]:
    labels_arr = np.array(labels, dtype=int)
    reps: dict[int, list[dict[str, str]]] = {}
    for cid in sorted(set(labels)):
        idx = np.where(labels_arr == cid)[0]
        cluster_vecs = embeddings[idx]
        centroid = cluster_vecs.mean(axis=0)
        dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
        order = np.argsort(dists)[: min(per_cluster, idx.size)]
        reps[cid] = [{"id": item_ids[int(idx[k])], "preview": previews[int(idx[k])]} for k in order]
    return reps


def _label_clusters(
    analysis_dir: Path,
    cluster_reps: dict[int, list[dict[str, str]]],
    top_terms: dict[int, list[str]],
) -> dict[int, str]:
    cache_path = analysis_dir / "llm_cache.json"
    labels: dict[int, str] = {}
    for cid, reps in cluster_reps.items():
        snippets = [r["preview"] for r in reps if r.get("preview")]
        if len(snippets) < 3:
            snippets.extend(top_terms.get(cid, [])[:3])
        try:
            labels[cid] = generate_label(snippets, cache_path=cache_path, prompt_tag=f"cluster_{cid}")
        except Exception:
            labels[cid] = fallback_tfidf_label(snippets) or "Topic"
    return labels


def _label_internal_nodes(
    analysis_dir: Path,
    nodes: dict[str, dict[str, Any]],
    texts: list[str],
    enable_ollama_for_internal: bool,
) -> None:
    cache_path = analysis_dir / "llm_cache.json"
    internal_nodes = [n for n in nodes.values() if n["children_ids"]]
    internal_nodes.sort(key=lambda n: int(n["size"]), reverse=True)
    ollama_budget = 80 if enable_ollama_for_internal else 0

    for node in internal_nodes:
        member_idx: list[int] = node.get("member_indices", [])
        snippets = [sanitize_preview(texts[i], limit=280) for i in member_idx[:8]]
        if not snippets:
            node["label"] = "Topic"
            continue
        if ollama_budget > 0 and len(member_idx) >= 8:
            try:
                node["label"] = generate_label(snippets, cache_path=cache_path, prompt_tag=f"node_{node['node_id']}")
            except Exception:
                node["label"] = fallback_tfidf_label(snippets) or "Topic"
            ollama_budget -= 1
        else:
            node["label"] = fallback_tfidf_label(snippets)


def _aggregate_granulate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for item in results:
        summary = item.get("aspect_summary", {})
        if isinstance(summary, dict):
            for aspect, info in summary.items():
                count = int((info or {}).get("count", 0)) if isinstance(info, dict) else 0
                if aspect:
                    counter[str(aspect)] += count
        elif isinstance(summary, list):
            for row in summary:
                if not isinstance(row, dict):
                    continue
                aspect = str(row.get("aspect", "")).strip()
                count = int(row.get("count", 0))
                if aspect:
                    counter[aspect] += count
    return [{"aspect": aspect, "count": count} for aspect, count in counter.most_common()]


def run_analysis_pipeline(
    analysis_id: str,
    analysis_dir: Path,
    mode: Literal["text", "csv"],
    texts: list[str],
    options: dict[str, Any],
) -> None:
    timings: dict[str, float] = {}
    started = time.perf_counter()

    try:
        write_status(analysis_dir, status="processing", stage="embeddings", pct=10)
        t0 = time.perf_counter()
        emb = generate_embeddings(texts=texts)
        save_embeddings(analysis_dir / "embeddings.npy", emb.vectors)
        timings["embeddings_sec"] = round(time.perf_counter() - t0, 4)

        item_ids = [f"item_{i}" for i in range(len(texts))]
        previews = [sanitize_preview(t, 140) for t in texts]
        write_json(
            analysis_dir / "items.json",
            [{"id": item_ids[i], "preview": previews[i], "text": texts[i]} for i in range(len(texts))],
        )

        write_status(analysis_dir, status="processing", stage="hierarchy", pct=30)
        t0 = time.perf_counter()
        h = build_hierarchy(
            embeddings=emb.vectors,
            item_ids=item_ids,
            k_clusters=int(options.get("k_clusters", 8)),
        )
        timings["hierarchy_sec"] = round(time.perf_counter() - t0, 4)

        write_status(analysis_dir, status="processing", stage="clusters", pct=50)
        t0 = time.perf_counter()
        top_terms = _cluster_top_terms(texts, h.cluster_ids)
        reps = _cluster_representatives(emb.vectors, h.cluster_ids, item_ids, previews)
        cluster_labels = _label_clusters(analysis_dir, reps, top_terms)
        clusters_payload: list[dict[str, Any]] = []
        counts = Counter(h.cluster_ids)
        for cid in sorted(counts.keys()):
            clusters_payload.append(
                {
                    "cluster_id": int(cid),
                    "label": cluster_labels.get(cid, f"Cluster {cid}"),
                    "size": int(counts[cid]),
                    "top_terms": top_terms.get(cid, []),
                    "representatives": reps.get(cid, []),
                }
            )
        write_json(
            analysis_dir / "clusters.json",
            {
                "clusters": clusters_payload,
                "item_cluster_map": [{"id": item_ids[i], "cluster_id": int(h.cluster_ids[i])} for i in range(len(item_ids))],
            },
        )
        timings["clusters_sec"] = round(time.perf_counter() - t0, 4)

        write_status(analysis_dir, status="processing", stage="umap", pct=65)
        t0 = time.perf_counter()
        xy = project_umap(
            emb.vectors,
            n_neighbors=int(options.get("umap_n_neighbors", 15)),
            min_dist=float(options.get("umap_min_dist", 0.1)),
        )
        map_points = [
            {
                "id": item_ids[i],
                "x": float(xy[i, 0]),
                "y": float(xy[i, 1]),
                "cluster_id": int(h.cluster_ids[i]),
                "preview": previews[i],
            }
            for i in range(len(item_ids))
        ]
        write_json(analysis_dir / "umap_2d.json", {"points": map_points, "clusters": clusters_payload})
        timings["umap_sec"] = round(time.perf_counter() - t0, 4)

        write_status(analysis_dir, status="processing", stage="labeling", pct=75)
        t0 = time.perf_counter()
        _label_internal_nodes(
            analysis_dir=analysis_dir,
            nodes=h.nodes,
            texts=texts,
            enable_ollama_for_internal=bool(options.get("label_internal_nodes", True)),
        )
        for node in h.nodes.values():
            node.pop("member_indices", None)
        write_json(analysis_dir / "hierarchy.json", {"root_id": h.root_id, "nodes": h.nodes, "leaves": h.leaves})
        timings["labeling_sec"] = round(time.perf_counter() - t0, 4)

        write_status(analysis_dir, status="processing", stage="granulate", pct=90)
        t0 = time.perf_counter()
        do_granulate = bool(options.get("granulate", True))
        granulate_max_rows = int(options.get("granulate_max_rows", 200))
        return_items = bool(options.get("granulate_return_items", False))
        gran_results: list[dict[str, Any]] = []
        selected_indices: list[int] = []
        if do_granulate:
            selected_indices = _sample_indices(len(texts), granulate_max_rows)
            for idx in selected_indices:
                out = granulate_text(texts[idx])
                gran_results.append(out)
            aggregate = _aggregate_granulate(gran_results)
        else:
            aggregate = []
        write_json(
            analysis_dir / "granulate_aggregate.json",
            {
                "mode": mode,
                "aggregate_aspect_summary": aggregate,
                "items_included": len(selected_indices),
                "items_total": len(texts),
                "item_ids_included": [item_ids[i] for i in selected_indices],
            },
        )
        if return_items and gran_results:
            gran_items_payload = []
            for local_idx, result in enumerate(gran_results):
                i = selected_indices[local_idx]
                gran_items_payload.append(
                    {
                        "id": item_ids[i],
                        "preview": previews[i],
                        "result": result,
                    }
                )
            write_json(analysis_dir / "granulate_items.json", gran_items_payload)
        timings["granulate_sec"] = round(time.perf_counter() - t0, 4)

        write_status(analysis_dir, status="processing", stage="overview", pct=98)
        overview = {
            "counts": {
                "items": len(texts),
                "clusters": len(clusters_payload),
                "aspects": len(read_json(analysis_dir / "granulate_aggregate.json").get("aggregate_aspect_summary", [])),
            },
            "top_clusters": sorted(clusters_payload, key=lambda c: int(c["size"]), reverse=True)[:5],
            "top_aspects": read_json(analysis_dir / "granulate_aggregate.json").get("aggregate_aspect_summary", [])[:8],
            "timing": timings,
        }
        write_json(analysis_dir / "overview.json", overview)

        timings["total_sec"] = round(time.perf_counter() - started, 4)
        write_json(analysis_dir / "timing.json", timings)
        write_status(analysis_dir, status="completed", stage="completed", pct=100)
    except Exception as exc:
        write_status(analysis_dir, status="failed", stage="failed", pct=100, error=str(exc))


def load_artifact_or_404(analysis_dir: Path, filename: str) -> dict[str, Any]:
    artifact = analysis_dir / filename
    if not artifact.exists():
        raise FileNotFoundError(f"{filename} not found for analysis {analysis_dir.name}")
    return read_json(artifact)


def load_granulate_payload(analysis_dir: Path, include_items: bool) -> dict[str, Any]:
    aggregate = load_artifact_or_404(analysis_dir, "granulate_aggregate.json")
    payload = {
        "mode": aggregate.get("mode", "csv"),
        "aggregate_aspect_summary": aggregate.get("aggregate_aspect_summary", []),
        "items_included": int(aggregate.get("items_included", 0)),
        "items_total": int(aggregate.get("items_total", 0)),
        "items": [],
    }
    if include_items:
        items_file = analysis_dir / "granulate_items.json"
        if items_file.exists():
            payload["items"] = read_json(items_file)
    return payload
