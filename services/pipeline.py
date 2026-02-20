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
from services.labeling import BROAD_NODE_BUCKETS, build_evidence_pack, fallback_tfidf_label, generate_label, keywords_are_diverse
from services.storage import read_json, write_json, write_status
from services.umap_service import project_umap
from services.validation import sanitize_preview

INTERNAL_BROAD_SIZE_THRESHOLD = 30
INTERNAL_OLLAMA_MAX_CALLS = 80
DEFAULT_PREVIEW_CHAR_LIMIT = 320


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _parse_pipeline_options(options: dict[str, Any] | None) -> dict[str, Any]:
    raw = options or {}
    return {
        "k_clusters": _clamp_int(raw.get("k_clusters", 8), default=8, minimum=2, maximum=64),
        "umap_n_neighbors": _clamp_int(raw.get("umap_n_neighbors", 15), default=15, minimum=2, maximum=100),
        "umap_min_dist": _clamp_float(raw.get("umap_min_dist", 0.1), default=0.1, minimum=0.0, maximum=0.99),
        "granulate_max_rows": _clamp_int(raw.get("granulate_max_rows", 200), default=200, minimum=1, maximum=2000),
        "label_internal_nodes": _parse_bool(raw.get("label_internal_nodes", True), default=True),
        "granulate": _parse_bool(raw.get("granulate", True), default=True),
        "granulate_return_items": _parse_bool(raw.get("granulate_return_items", False), default=False),
        "llm_label_budget": _clamp_int(raw.get("llm_label_budget", 120), default=120, minimum=0, maximum=10000),
    }


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
    budget: dict[str, int] | None = None,
) -> dict[int, str]:
    def _sibling_negative_keywords(cluster_id: int, max_terms: int = 5) -> list[str]:
        own = {t.strip().lower() for t in top_terms.get(cluster_id, []) if t and t.strip()}
        out: list[str] = []
        for other_cid in sorted(top_terms.keys()):
            if other_cid == cluster_id:
                continue
            for term in top_terms.get(other_cid, []):
                candidate = str(term or "").strip().lower()
                if not candidate or candidate in own or candidate in out:
                    continue
                out.append(candidate)
                if len(out) >= max_terms:
                    return out
        return out

    cache_path = analysis_dir / "llm_cache.json"
    labels: dict[int, str] = {}
    for cid, reps in cluster_reps.items():
        snippets = [sanitize_preview(r["preview"], limit=280) for r in reps if r.get("preview")]
        if len(snippets) < 3:
            snippets.extend(top_terms.get(cid, [])[:3])
        try:
            labels[cid] = generate_label(
                snippets,
                cache_path=cache_path,
                prompt_tag=f"cluster_{cid}",
                top_keywords=top_terms.get(cid, []),
                sibling_negative_keywords=_sibling_negative_keywords(cid),
                budget=budget,
            )
        except Exception:
            labels[cid] = fallback_tfidf_label(snippets) or "Topic"
    return labels


def _label_internal_nodes(
    analysis_dir: Path,
    nodes: dict[str, dict[str, Any]],
    texts: list[str],
    embeddings: np.ndarray,
    enable_ollama_for_internal: bool,
    budget: dict[str, int] | None = None,
    max_llm_calls: int = INTERNAL_OLLAMA_MAX_CALLS,
) -> None:
    def _sibling_texts(node: dict[str, Any], max_items: int = 60) -> list[str]:
        parent_id = str(node.get("parent_id") or "").strip()
        if not parent_id or parent_id not in nodes:
            return []
        siblings: list[str] = []
        for sibling_id in nodes[parent_id].get("children_ids", []):
            if sibling_id == node.get("node_id"):
                continue
            sibling = nodes.get(sibling_id, {})
            member_indices: list[int] = sibling.get("member_indices", [])
            for idx in member_indices:
                if 0 <= int(idx) < len(texts):
                    siblings.append(texts[int(idx)])
                    if len(siblings) >= max_items:
                        return siblings
        return siblings

    def _child_label_candidates(node: dict[str, Any]) -> list[str]:
        labels: list[str] = []
        for child_id in node.get("children_ids", []):
            child = nodes.get(child_id, {})
            label = str(child.get("label", "")).strip()
            if not label or label.lower() in {"item", "topic"}:
                continue
            if label not in labels:
                labels.append(label)
        labels.extend(BROAD_NODE_BUCKETS)
        return labels

    cache_path = analysis_dir / "llm_cache.json"
    internal_nodes = [n for n in nodes.values() if n["children_ids"]]
    # Label children first so parent nodes can reuse child labels as broad candidates.
    internal_nodes.sort(key=lambda n: int(n["size"]))
    llm_calls = 0

    for node in internal_nodes:
        member_idx = [int(i) for i in node.get("member_indices", []) if isinstance(i, int) or str(i).isdigit()]
        member_idx = [i for i in member_idx if 0 <= i < len(texts)]
        member_texts = [texts[i] for i in member_idx]
        member_embeddings: np.ndarray | None = None
        if len(member_idx) > 0 and isinstance(embeddings, np.ndarray):
            try:
                member_embeddings = embeddings[np.array(member_idx, dtype=int)]
            except Exception:
                member_embeddings = None

        evidence = build_evidence_pack(
            texts=member_texts,
            embeddings=member_embeddings,
            sibling_texts=_sibling_texts(node),
            keyword_limit=14,
            representative_limit=6,
        )
        snippets = evidence.get("representative_items", [])
        if len(snippets) < 3:
            snippets.extend([sanitize_preview(texts[i], limit=280) for i in member_idx[: max(0, 3 - len(snippets))]])
        if not snippets:
            node["label"] = "Topic"
            continue

        is_broad_node = int(node.get("size", len(member_idx))) >= INTERNAL_BROAD_SIZE_THRESHOLD or keywords_are_diverse(
            evidence.get("top_keywords", [])
        )
        candidates = _child_label_candidates(node) if is_broad_node else None

        can_use_llm = enable_ollama_for_internal and llm_calls < max(0, int(max_llm_calls))
        if can_use_llm and len(member_idx) >= 6:
            try:
                node["label"] = generate_label(
                    snippets,
                    cache_path=cache_path,
                    prompt_tag=f"node_{node['node_id']}",
                    top_keywords=evidence.get("top_keywords", []),
                    sibling_negative_keywords=evidence.get("sibling_negative_keywords", []),
                    candidates=candidates,
                    force_broad=is_broad_node,
                    budget=budget,
                )
            except Exception:
                node["label"] = fallback_tfidf_label(snippets) or "Topic"
            llm_calls += 1
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
    parsed_options = _parse_pipeline_options(options)

    try:
        if not texts or not any(str(t or "").strip() for t in texts):
            raise ValueError("No non-empty texts were provided for analysis")

        write_status(analysis_dir, status="processing", stage="embeddings", pct=10)
        t0 = time.perf_counter()
        emb = generate_embeddings(texts=texts)
        save_embeddings(analysis_dir / "embeddings.npy", emb.vectors)
        timings["embeddings_sec"] = round(time.perf_counter() - t0, 4)

        item_ids = [f"item_{i}" for i in range(len(texts))]
        previews = [sanitize_preview(t, DEFAULT_PREVIEW_CHAR_LIMIT) for t in texts]
        write_json(
            analysis_dir / "items.json",
            [{"id": item_ids[i], "preview": previews[i], "text": texts[i]} for i in range(len(texts))],
        )

        write_status(analysis_dir, status="processing", stage="hierarchy", pct=30)
        t0 = time.perf_counter()
        h = build_hierarchy(
            embeddings=emb.vectors,
            item_ids=item_ids,
            k_clusters=parsed_options["k_clusters"],
        )
        timings["hierarchy_sec"] = round(time.perf_counter() - t0, 4)

        write_status(analysis_dir, status="processing", stage="clusters", pct=50)
        t0 = time.perf_counter()
        top_terms = _cluster_top_terms(texts, h.cluster_ids)
        reps = _cluster_representatives(emb.vectors, h.cluster_ids, item_ids, previews)
        llm_budget = {"remaining": parsed_options["llm_label_budget"]}
        cluster_labels = _label_clusters(analysis_dir, reps, top_terms, budget=llm_budget)
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
            n_neighbors=parsed_options["umap_n_neighbors"],
            min_dist=parsed_options["umap_min_dist"],
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
            embeddings=emb.vectors,
            enable_ollama_for_internal=parsed_options["label_internal_nodes"],
            budget=llm_budget,
            max_llm_calls=min(INTERNAL_OLLAMA_MAX_CALLS, parsed_options["llm_label_budget"]),
        )
        for node in h.nodes.values():
            node.pop("member_indices", None)
        write_json(analysis_dir / "hierarchy.json", {"root_id": h.root_id, "nodes": h.nodes, "leaves": h.leaves})
        timings["labeling_sec"] = round(time.perf_counter() - t0, 4)

        write_status(analysis_dir, status="processing", stage="granulate", pct=90)
        t0 = time.perf_counter()
        do_granulate = parsed_options["granulate"]
        granulate_max_rows = parsed_options["granulate_max_rows"]
        return_items = parsed_options["granulate_return_items"]
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
        top_aspects = aggregate[:8]
        overview = {
            "counts": {
                "items": len(texts),
                "clusters": len(clusters_payload),
                "aspects": len(aggregate),
            },
            "top_clusters": sorted(clusters_payload, key=lambda c: int(c["size"]), reverse=True)[:5],
            "top_aspects": top_aspects,
            "timing": timings,
        }
        write_json(analysis_dir / "overview.json", overview)

        timings["total_sec"] = round(time.perf_counter() - started, 4)
        write_json(analysis_dir / "timing.json", timings)
        write_status(analysis_dir, status="completed", stage="completed", pct=100)
    except Exception as exc:
        message = f"{exc.__class__.__name__}: {exc}"
        try:
            write_status(analysis_dir, status="failed", stage="failed", pct=100, error=message)
        except Exception:
            pass


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
