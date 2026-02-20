from __future__ import annotations

import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from services.embeddings import clean_texts_for_embeddings, generate_embeddings, save_embeddings
from services.granulate import granulate_text
from services.hierarchy import build_hierarchy
from services.labeling import (
    BROAD_NODE_BUCKETS,
    build_evidence_pack,
    fallback_tfidf_label,
    generate_label,
    get_label_stop_words,
    is_informative_term,
    keywords_are_diverse,
)
from services.storage import read_json, write_json, write_status
from services.umap_service import project_umap
from services.validation import sanitize_preview

INTERNAL_BROAD_SIZE_THRESHOLD = 30
INTERNAL_OLLAMA_MAX_CALLS = 80
DEFAULT_PREVIEW_CHAR_LIMIT = 320
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


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
        "umap_scale_points": _parse_bool(raw.get("umap_scale_points", True), default=True),
        "umap_scale_clamp": _clamp_float(raw.get("umap_scale_clamp", 3.0), default=3.0, minimum=1.5, maximum=6.0),
        "granulate_max_rows": _clamp_int(raw.get("granulate_max_rows", 200), default=200, minimum=1, maximum=2000),
        "granulate_per_cluster": _parse_bool(raw.get("granulate_per_cluster", False), default=False),
        "granulate_per_cluster_max_rows": _clamp_int(
            raw.get("granulate_per_cluster_max_rows", 80),
            default=80,
            minimum=1,
            maximum=300,
        ),
        "label_internal_nodes": _parse_bool(raw.get("label_internal_nodes", True), default=True),
        "granulate": _parse_bool(raw.get("granulate", True), default=True),
        "granulate_return_items": _parse_bool(raw.get("granulate_return_items", False), default=False),
        "llm_label_budget": _clamp_int(raw.get("llm_label_budget", 120), default=120, minimum=0, maximum=10000),
        "preview_char_limit": _clamp_int(
            raw.get("preview_char_limit", DEFAULT_PREVIEW_CHAR_LIMIT),
            default=DEFAULT_PREVIEW_CHAR_LIMIT,
            minimum=120,
            maximum=1200,
        ),
        "insights_theme_count": _clamp_int(raw.get("insights_theme_count", 5), default=5, minimum=3, maximum=8),
    }


def _sample_indices(n_items: int, max_rows: int) -> list[int]:
    if n_items <= max_rows:
        return list(range(n_items))
    rng = random.Random(42)
    return sorted(rng.sample(range(n_items), k=max_rows))


def _sample_from_indices(indices: list[int], max_rows: int, seed: int) -> list[int]:
    if len(indices) <= max_rows:
        return list(indices)
    rng = random.Random(seed)
    return sorted(rng.sample(indices, k=max_rows))


def _clean_metadata(meta: Any) -> dict[str, str]:
    if not isinstance(meta, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in meta.items():
        k = str(key or "").strip()
        v = str(value or "").strip()
        if not k or not v:
            continue
        out[k] = v
    return out


def _term_tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in WORD_RE.findall(str(text or "").lower()):
        if len(tok) < 3:
            continue
        out.append(tok)
    return out


def _title_from_tokens(tokens: list[str]) -> str:
    cleaned = [str(tok).strip().lower() for tok in tokens if str(tok).strip()]
    if not cleaned:
        return "Customer Request"
    return " ".join(tok.capitalize() for tok in cleaned[:3])


def _rule_based_label_from_terms(top_terms: list[str]) -> str | None:
    lexicon: list[tuple[str, set[str]]] = [
        ("Delivery Delay", {"delivery", "shipping", "arrive", "arrival", "delay", "late", "courier", "deli"}),
        ("Purchase Inquiry", {"order", "purchase", "buy", "quote", "price", "deals", "estimated"}),
        (
            "Technical Support",
            {"install", "software", "router", "server", "support", "assist", "error", "connection", "wifi", "wireless"},
        ),
        ("Battery And Power", {"battery", "charge", "power", "usage", "runtime"}),
        ("Product Inquiry", {"specifications", "information", "website", "features", "device"}),
        ("Product Defect", {"damaged", "broken", "defect", "repair", "technician", "warranty", "quality"}),
        ("Bulk Business Order", {"bulk", "chairs", "desk", "office", "business", "company"}),
        ("Positive Feedback", {"satisfied", "pleased", "amazing", "great", "excellent", "love", "blown"}),
    ]

    term_tokens: list[str] = []
    for term in top_terms:
        term_tokens.extend(_term_tokens(term))
    token_set = set(term_tokens)
    if not token_set:
        return None

    best_label: str | None = None
    best_score = 0
    for label, keywords in lexicon:
        score = len(token_set & keywords)
        if score > best_score:
            best_score = score
            best_label = label
    if best_score >= 1:
        return best_label
    return None


def _derive_label_from_terms(top_terms: list[str], fallback: str = "Customer Request") -> str:
    rule_label = _rule_based_label_from_terms(top_terms)
    if rule_label:
        return rule_label

    for raw_term in top_terms:
        term = str(raw_term or "").strip().lower()
        if not term:
            continue
        tokens = _term_tokens(term)
        if not tokens:
            continue
        if len(tokens) > 3:
            tokens = tokens[:3]
        phrase = " ".join(tokens)
        if is_informative_term(phrase):
            return _title_from_tokens(tokens)
    fallback_tokens = _term_tokens(fallback)
    if fallback_tokens:
        candidate = _title_from_tokens(fallback_tokens)
        if is_informative_term(" ".join(fallback_tokens)):
            return candidate
    return "Customer Request"


def _refine_cluster_label(label: str, top_terms: list[str]) -> str:
    raw = " ".join(str(label or "").split()).strip()
    if not raw:
        return _derive_label_from_terms(top_terms)

    tokens = [tok.lower() for tok in WORD_RE.findall(raw)]
    if not tokens:
        return _derive_label_from_terms(top_terms)

    connector_words = {"about", "there", "right", "with", "and", "this", "that", "those", "these"}
    weak_words = {"information", "website", "absolutely", "blown", "away", "deals", "right", "possible", "working"}
    if any(tok in connector_words for tok in tokens) or any(tok in weak_words for tok in tokens):
        return _derive_label_from_terms(top_terms, fallback=raw)

    phrase = " ".join(tokens[:3])
    if not is_informative_term(phrase):
        return _derive_label_from_terms(top_terms, fallback=raw)

    if len(tokens) == 1 and not is_informative_term(tokens[0]):
        return _derive_label_from_terms(top_terms, fallback=raw)
    return _title_from_tokens(tokens)


def _fallback_cluster_terms(texts: list[str], top_n: int = 8) -> list[str]:
    scores: Counter[str] = Counter()
    for text in texts:
        tokens = _term_tokens(text)
        for token in tokens:
            if is_informative_term(token):
                scores[token] += 1
        for left, right in zip(tokens, tokens[1:]):
            phrase = f"{left} {right}"
            if is_informative_term(phrase):
                scores[phrase] += 3
        for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
            tri = f"{a} {b} {c}"
            if is_informative_term(tri):
                scores[tri] += 4
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], -int(" " in kv[0]), kv[0]))
    return [term for term, _ in ranked[:top_n]]


def _cluster_top_terms(texts: list[str], labels: list[int], top_n: int = 8) -> dict[int, list[str]]:
    labels_arr = np.array(labels, dtype=int)
    out: dict[int, list[str]] = {}

    stop_words = sorted(get_label_stop_words())
    try:
        min_df = 2 if len(texts) >= 10 else 1
        try:
            vec = TfidfVectorizer(
                lowercase=True,
                stop_words=stop_words,
                ngram_range=(1, 3),
                max_features=12000,
                min_df=min_df,
            )
            mat = vec.fit_transform(texts)
        except ValueError:
            vec = TfidfVectorizer(
                lowercase=True,
                stop_words=stop_words,
                ngram_range=(1, 3),
                max_features=12000,
                min_df=1,
            )
            mat = vec.fit_transform(texts)
        terms = vec.get_feature_names_out()
    except Exception:
        for cid in sorted(set(labels)):
            idx = np.where(labels_arr == cid)[0]
            out[cid] = _fallback_cluster_terms([texts[int(i)] for i in idx], top_n=top_n)
        return out

    for cid in sorted(set(labels)):
        idx = np.where(labels_arr == cid)[0]
        if idx.size == 0:
            out[cid] = []
            continue
        scores = mat[idx].mean(axis=0).A1
        order = np.argsort(scores)[::-1]
        candidates: list[str] = []
        seen: set[str] = set()
        for j in order:
            score = float(scores[int(j)])
            if score <= 0:
                break
            term = str(terms[int(j)]).strip().lower()
            if not is_informative_term(term):
                continue
            signature = term.replace(" ", "")
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(term)
            if len(candidates) >= top_n * 8:
                break
        bigrams_first = [t for t in candidates if " " in t]
        unigrams = [t for t in candidates if " " not in t]
        picked = bigrams_first[:top_n]
        if len(picked) < top_n:
            picked.extend(unigrams[: max(0, top_n - len(picked))])
        if not picked:
            picked = _fallback_cluster_terms([texts[int(i)] for i in idx], top_n=top_n)
        out[cid] = picked
    return out


def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def _cluster_representatives(
    embeddings: np.ndarray,
    labels: list[int],
    item_ids: list[str],
    previews: list[str],
    cleaned_texts: list[str],
    metadata_rows: list[dict[str, str]],
    per_cluster: int = 5,
) -> dict[int, list[dict[str, Any]]]:
    labels_arr = np.array(labels, dtype=int)
    reps: dict[int, list[dict[str, Any]]] = {}
    for cid in sorted(set(labels)):
        idx = np.where(labels_arr == cid)[0]
        cluster_vecs = embeddings[idx]
        centroid = cluster_vecs.mean(axis=0)
        dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
        order = np.argsort(dists)

        selected: list[dict[str, Any]] = []
        selected_token_sets: list[set[str]] = []
        for pos in order:
            i = int(idx[int(pos)])
            tokens = set(_term_tokens(cleaned_texts[i]))
            if tokens and any(_jaccard_similarity(tokens, prev) >= 0.82 for prev in selected_token_sets):
                continue

            rep: dict[str, Any] = {
                "id": item_ids[i],
                "preview": previews[i],
                "cleaned_snippet": cleaned_texts[i][:280],
            }
            meta = metadata_rows[i] if i < len(metadata_rows) else {}
            if meta:
                rep["metadata"] = meta
            selected.append(rep)
            if tokens:
                selected_token_sets.append(tokens)
            if len(selected) >= min(per_cluster, idx.size):
                break

        if len(selected) < min(per_cluster, idx.size):
            used = {str(rep["id"]) for rep in selected}
            for pos in order:
                i = int(idx[int(pos)])
                if item_ids[i] in used:
                    continue
                rep = {
                    "id": item_ids[i],
                    "preview": previews[i],
                    "cleaned_snippet": cleaned_texts[i][:280],
                }
                meta = metadata_rows[i] if i < len(metadata_rows) else {}
                if meta:
                    rep["metadata"] = meta
                selected.append(rep)
                if len(selected) >= min(per_cluster, idx.size):
                    break
        reps[cid] = selected
    return reps


def _label_clusters(
    analysis_dir: Path,
    cluster_reps: dict[int, list[dict[str, Any]]],
    top_terms: dict[int, list[str]],
    budget: dict[str, int] | None = None,
) -> dict[int, str]:
    def _sibling_negative_keywords(cluster_id: int, max_terms: int = 6) -> list[str]:
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
        snippets = [str(r.get("cleaned_snippet", "")).strip() for r in reps if str(r.get("cleaned_snippet", "")).strip()]
        if len(snippets) < 3:
            snippets.extend(top_terms.get(cid, [])[:3])
        try:
            label = generate_label(
                snippets,
                cache_path=cache_path,
                prompt_tag=f"cluster_{cid}",
                top_keywords=top_terms.get(cid, []),
                sibling_negative_keywords=_sibling_negative_keywords(cid),
                budget=budget,
            )
            labels[cid] = label
        except Exception:
            labels[cid] = fallback_tfidf_label(snippets) or "Customer Request"
    return labels


def _label_internal_nodes(
    analysis_dir: Path,
    nodes: dict[str, dict[str, Any]],
    cleaned_texts: list[str],
    embeddings: np.ndarray,
    enable_ollama_for_internal: bool,
    budget: dict[str, int] | None = None,
    max_llm_calls: int = INTERNAL_OLLAMA_MAX_CALLS,
) -> None:
    def _sibling_texts(node: dict[str, Any], max_items: int = 80) -> list[str]:
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
                i = int(idx)
                if 0 <= i < len(cleaned_texts):
                    siblings.append(cleaned_texts[i])
                    if len(siblings) >= max_items:
                        return siblings
        return siblings

    def _child_label_candidates(node: dict[str, Any]) -> list[str]:
        labels: list[str] = []
        for child_id in node.get("children_ids", []):
            child = nodes.get(child_id, {})
            label = str(child.get("label", "")).strip()
            if not label or label.lower() in {"item", "topic", "other"}:
                continue
            if label not in labels:
                labels.append(label)
        labels.extend(BROAD_NODE_BUCKETS)
        return labels

    cache_path = analysis_dir / "llm_cache.json"
    internal_nodes = [n for n in nodes.values() if n.get("children_ids")]
    internal_nodes.sort(key=lambda n: int(n.get("size", 0)))
    llm_calls = 0

    for node in internal_nodes:
        member_idx = [int(i) for i in node.get("member_indices", []) if isinstance(i, int) or str(i).isdigit()]
        member_idx = [i for i in member_idx if 0 <= i < len(cleaned_texts)]
        member_texts = [cleaned_texts[i] for i in member_idx]
        member_embeddings: np.ndarray | None = None
        if member_idx and isinstance(embeddings, np.ndarray):
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
            snippets.extend([cleaned_texts[i] for i in member_idx[: max(0, 3 - len(snippets))]])
        snippets = [s for s in snippets if s]
        if not snippets:
            node["label"] = "Customer Request"
            continue

        is_broad_node = int(node.get("size", len(member_idx))) >= INTERNAL_BROAD_SIZE_THRESHOLD or keywords_are_diverse(
            evidence.get("top_keywords", [])
        )
        candidates = _child_label_candidates(node) if is_broad_node else None

        can_use_llm = enable_ollama_for_internal and llm_calls < max(0, int(max_llm_calls))
        if can_use_llm and (len(member_idx) >= 6 or is_broad_node):
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
                node["label"] = fallback_tfidf_label(snippets) or "Customer Request"
            llm_calls += 1
        else:
            node["label"] = fallback_tfidf_label(snippets)

        if str(node.get("label", "")).strip().lower() in {"topic", "other", "issue", "problem"}:
            node["label"] = fallback_tfidf_label(member_texts[:8]) or "Customer Request"


def _aggregate_granulate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for item in results:
        summary = item.get("aspect_summary", {})
        if isinstance(summary, dict):
            for aspect, info in summary.items():
                count = int((info or {}).get("count", 0)) if isinstance(info, dict) else 0
                if aspect and count > 0:
                    counter[str(aspect)] += count
        elif isinstance(summary, list):
            for row in summary:
                if not isinstance(row, dict):
                    continue
                aspect = str(row.get("aspect", "")).strip()
                count = int(row.get("count", 0))
                if aspect and count > 0:
                    counter[aspect] += count
    return [{"aspect": aspect, "count": count} for aspect, count in counter.most_common()]


def _aggregate_granulate_per_cluster(
    texts: list[str],
    cluster_ids: list[int],
    cluster_label_map: dict[int, str],
    max_rows: int,
) -> list[dict[str, Any]]:
    by_cluster: dict[int, list[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        by_cluster.setdefault(int(cid), []).append(int(idx))

    out: list[dict[str, Any]] = []
    for cid in sorted(by_cluster.keys()):
        all_indices = by_cluster[cid]
        sampled = _sample_from_indices(all_indices, max_rows=max_rows, seed=42 + cid)
        results: list[dict[str, Any]] = []
        for idx in sampled:
            try:
                results.append(granulate_text(texts[idx]))
            except Exception:
                continue
        out.append(
            {
                "cluster_id": cid,
                "cluster_label": cluster_label_map.get(cid, f"Cluster {cid}"),
                "items_included": len(sampled),
                "items_total": len(all_indices),
                "aggregate_aspect_summary": _aggregate_granulate(results),
            }
        )
    return out


def _scale_umap_points(xy: np.ndarray, clamp: float = 3.0) -> np.ndarray:
    if xy.shape[0] <= 1:
        return xy.astype(np.float32, copy=False)
    mean = xy.mean(axis=0, keepdims=True)
    std = xy.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    scaled = (xy - mean) / std
    scaled = np.clip(scaled, -float(clamp), float(clamp))
    return scaled.astype(np.float32, copy=False)


def _summary_terms_for_indices(member_idx: list[int], cleaned_texts: list[str], top_n: int = 3) -> list[str]:
    if not member_idx:
        return []
    sampled_idx = member_idx
    if len(sampled_idx) > 120:
        step = max(1, len(sampled_idx) // 120)
        sampled_idx = sampled_idx[::step][:120]

    counter: Counter[str] = Counter()
    for idx in sampled_idx:
        tokens = _term_tokens(cleaned_texts[idx])
        for tok in tokens:
            if is_informative_term(tok):
                counter[tok] += 1
        for left, right in zip(tokens, tokens[1:]):
            phrase = f"{left} {right}"
            if is_informative_term(phrase):
                counter[phrase] += 2
    if not counter:
        return []

    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], -int(" " in kv[0]), kv[0]))
    out: list[str] = []
    seen_tokens: set[str] = set()
    for term, _score in ranked:
        tokens = set(term.split())
        if tokens & seen_tokens:
            continue
        out.append(term)
        seen_tokens |= tokens
        if len(out) >= top_n:
            break
    return out


def _enrich_hierarchy_nodes(
    nodes: dict[str, dict[str, Any]],
    cleaned_texts: list[str],
    cluster_ids: list[int],
    cluster_label_map: dict[int, str],
) -> None:
    for node in nodes.values():
        member_idx = [int(i) for i in node.get("member_indices", []) if isinstance(i, int) or str(i).isdigit()]
        member_idx = [i for i in member_idx if 0 <= i < len(cleaned_texts)]
        if member_idx:
            node["descendant_leaf_count"] = int(len(member_idx))
        node["similarity"] = float(node.get("cohesion", 1.0))

        if not node.get("children_ids"):
            continue

        cluster_counter: Counter[int] = Counter(int(cluster_ids[i]) for i in member_idx if 0 <= i < len(cluster_ids))
        if cluster_counter:
            dominant_cluster_id, dominant_count = sorted(cluster_counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            node["dominant_cluster_id"] = int(dominant_cluster_id)
            node["dominant_cluster_share"] = float(round(dominant_count / max(1, len(member_idx)), 4))
        else:
            node["dominant_cluster_id"] = None
            node["dominant_cluster_share"] = 0.0

        terms = _summary_terms_for_indices(member_idx, cleaned_texts, top_n=3)
        if terms:
            node["summary"] = ", ".join(terms)
        elif node.get("dominant_cluster_id") is not None:
            cid = int(node["dominant_cluster_id"])
            node["summary"] = cluster_label_map.get(cid, "Mixed requests")
        else:
            node["summary"] = "Mixed requests"


def _build_quality_warnings(raw_texts: list[str], cleaned_texts: list[str]) -> list[str]:
    warnings: list[str] = []
    if not raw_texts:
        return warnings

    normalized_raw = [" ".join(t.lower().split()) for t in raw_texts if str(t).strip()]
    if normalized_raw:
        dup_ratio = 1.0 - (len(set(normalized_raw)) / len(normalized_raw))
        if dup_ratio >= 0.15:
            warnings.append("Many records look duplicated, which can skew theme quality.")

    short_ratio = sum(1 for t in cleaned_texts if len(str(t).strip()) < 20) / max(1, len(cleaned_texts))
    if short_ratio >= 0.3:
        warnings.append("Many texts are short after cleaning, so some clusters may be noisy.")

    raw_token_total = 0
    raw_code_like = 0
    clean_token_total = 0
    for raw, clean in zip(raw_texts, cleaned_texts):
        raw_tokens = re.findall(r"[A-Za-z0-9]+", str(raw or ""))
        clean_tokens = re.findall(r"[A-Za-z0-9]+", str(clean or ""))
        raw_token_total += len(raw_tokens)
        clean_token_total += len(clean_tokens)
        raw_code_like += sum(1 for tok in raw_tokens if any(ch.isdigit() for ch in tok))

    if raw_token_total > 0 and (clean_token_total / raw_token_total) < 0.45:
        warnings.append("Text appears dominated by boilerplate or IDs; cleaning removed a large share of tokens.")
    if raw_token_total > 0 and (raw_code_like / raw_token_total) > 0.22:
        warnings.append("Many tokens look like codes/IDs, which may reduce semantic clustering quality.")
    return warnings


def _metadata_finding(items_payload: list[dict[str, Any]]) -> str | None:
    distributions: dict[str, Counter[str]] = {}
    for item in items_payload:
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        for key, value in metadata.items():
            k = str(key or "").strip()
            v = str(value or "").strip()
            if not k or not v:
                continue
            distributions.setdefault(k, Counter())[v] += 1
    if not distributions:
        return None

    selected_key = sorted(distributions.keys(), key=lambda k: (-sum(distributions[k].values()), k.lower()))[0]
    counter = distributions[selected_key]
    top_value, top_count = counter.most_common(1)[0]
    total = sum(counter.values())
    pct = (top_count / max(1, total)) * 100.0
    return f"Most common {selected_key} is '{top_value}' ({top_count}/{total}, {pct:.0f}%)."


def _build_insights(
    clusters_payload: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    items_payload: list[dict[str, Any]],
    quality_warnings: list[str],
    theme_count: int,
) -> dict[str, Any]:
    sorted_clusters = sorted(clusters_payload, key=lambda c: int(c.get("size", 0)), reverse=True)
    total_items = len(items_payload)
    key_findings: list[str] = []

    key_findings.append(f"Processed {total_items} records and detected {len(sorted_clusters)} main themes.")

    if sorted_clusters:
        top = sorted_clusters[0]
        top_size = int(top.get("size", 0))
        pct = (top_size / max(1, total_items)) * 100.0
        key_findings.append(f"Largest theme is '{top.get('label', 'Theme')}' with {top_size} records ({pct:.0f}%).")
    if len(sorted_clusters) >= 2:
        second = sorted_clusters[1]
        key_findings.append(
            f"Second largest theme is '{second.get('label', 'Theme')}' with {int(second.get('size', 0))} records."
        )

    top_aspects = [str(a.get("aspect", "")).strip() for a in aggregate[:3] if str(a.get("aspect", "")).strip()]
    if top_aspects:
        key_findings.append(f"Top reported aspects are: {', '.join(top_aspects)}.")

    metadata_line = _metadata_finding(items_payload)
    if metadata_line:
        key_findings.append(metadata_line)

    if quality_warnings:
        key_findings.append(f"Quality note: {quality_warnings[0]}")

    while len(key_findings) < 3:
        key_findings.append("Themes are stable enough to explore by cluster and representative examples.")
    key_findings = key_findings[:6]

    theme_summary: list[dict[str, Any]] = []
    for cluster in sorted_clusters[:theme_count]:
        reps = cluster.get("representatives", [])
        examples = [str(rep.get("preview", "")).strip() for rep in reps if str(rep.get("preview", "")).strip()][:2]
        theme_summary.append(
            {
                "label": cluster.get("label", "Theme"),
                "size": int(cluster.get("size", 0)),
                "top_terms": cluster.get("top_terms", []),
                "examples": examples,
            }
        )

    return {
        "key_findings": key_findings,
        "theme_summary": theme_summary,
        "quality_warnings": quality_warnings,
    }


def run_analysis_pipeline(
    analysis_id: str,
    analysis_dir: Path,
    mode: Literal["text", "csv"],
    texts: list[str],
    options: dict[str, Any],
    input_items: list[dict[str, Any]] | None = None,
) -> None:
    timings: dict[str, float] = {}
    started = time.perf_counter()
    parsed_options = _parse_pipeline_options(options)

    try:
        raw_texts = [str(t or "").strip() for t in texts]
        if input_items and len(input_items) == len(raw_texts):
            for i, row in enumerate(input_items):
                txt = str(row.get("text", "")).strip()
                if txt:
                    raw_texts[i] = txt

        if not raw_texts or not any(raw_texts):
            raise ValueError("No non-empty texts were provided for analysis")

        metadata_rows: list[dict[str, str]] = [{} for _ in raw_texts]
        if input_items and len(input_items) == len(raw_texts):
            for i, row in enumerate(input_items):
                metadata_rows[i] = _clean_metadata(row.get("metadata", {}))

        cleaned_texts = clean_texts_for_embeddings(raw_texts)

        write_status(analysis_dir, status="processing", stage="embeddings", pct=10)
        t0 = time.perf_counter()
        emb = generate_embeddings(texts=cleaned_texts)
        save_embeddings(analysis_dir / "embeddings.npy", emb.vectors)
        timings["embeddings_sec"] = round(time.perf_counter() - t0, 4)

        item_ids = [f"item_{i}" for i in range(len(raw_texts))]
        preview_limit = parsed_options["preview_char_limit"]
        previews = [sanitize_preview(text, preview_limit) for text in raw_texts]
        items_payload: list[dict[str, Any]] = []
        for i in range(len(raw_texts)):
            item_payload: dict[str, Any] = {
                "id": item_ids[i],
                "preview": previews[i],
                "text": raw_texts[i],
            }
            if metadata_rows[i]:
                item_payload["metadata"] = metadata_rows[i]
            items_payload.append(item_payload)
        write_json(analysis_dir / "items.json", items_payload)

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
        top_terms = _cluster_top_terms(cleaned_texts, h.cluster_ids)
        reps = _cluster_representatives(
            emb.vectors,
            h.cluster_ids,
            item_ids,
            previews,
            cleaned_texts,
            metadata_rows,
            per_cluster=5,
        )
        llm_budget = {"remaining": parsed_options["llm_label_budget"]}
        raw_cluster_labels = _label_clusters(analysis_dir, reps, top_terms, budget=llm_budget)
        cluster_labels: dict[int, str] = {}
        for cid in sorted(set(h.cluster_ids)):
            cluster_labels[int(cid)] = _refine_cluster_label(raw_cluster_labels.get(int(cid), ""), top_terms.get(int(cid), []))
        cluster_label_map: dict[int, str] = {int(cid): label for cid, label in cluster_labels.items()}

        clusters_payload: list[dict[str, Any]] = []
        counts = Counter(h.cluster_ids)
        for cid in sorted(counts.keys()):
            representatives = []
            for rep in reps.get(cid, []):
                clean_rep = {"id": rep.get("id"), "preview": rep.get("preview")}
                if isinstance(rep.get("metadata"), dict) and rep["metadata"]:
                    clean_rep["metadata"] = rep["metadata"]
                representatives.append(clean_rep)
            clusters_payload.append(
                {
                    "cluster_id": int(cid),
                    "label": cluster_labels.get(cid, f"Cluster {cid}"),
                    "size": int(counts[cid]),
                    "top_terms": top_terms.get(cid, []),
                    "representatives": representatives,
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
        xy_raw = project_umap(
            emb.vectors,
            n_neighbors=parsed_options["umap_n_neighbors"],
            min_dist=parsed_options["umap_min_dist"],
        )
        xy_scaled = _scale_umap_points(xy_raw, clamp=parsed_options["umap_scale_clamp"])
        xy = xy_scaled if parsed_options["umap_scale_points"] else xy_raw
        map_points = []
        for i in range(len(item_ids)):
            point: dict[str, Any] = {
                "id": item_ids[i],
                "x": float(xy[i, 0]),
                "y": float(xy[i, 1]),
                "x_raw": float(xy_raw[i, 0]),
                "y_raw": float(xy_raw[i, 1]),
                "cluster_id": int(h.cluster_ids[i]),
                "cluster_label": cluster_label_map.get(int(h.cluster_ids[i]), f"Cluster {int(h.cluster_ids[i])}"),
                "preview": previews[i],
            }
            if metadata_rows[i]:
                point["metadata"] = metadata_rows[i]
            map_points.append(point)
        write_json(
            analysis_dir / "umap_2d.json",
            {
                "points": map_points,
                "clusters": clusters_payload,
                "advanced": {
                    "umap_scaled": bool(parsed_options["umap_scale_points"]),
                    "scale_clamp": float(parsed_options["umap_scale_clamp"]),
                },
            },
        )
        timings["umap_sec"] = round(time.perf_counter() - t0, 4)

        write_status(analysis_dir, status="processing", stage="labeling", pct=75)
        t0 = time.perf_counter()
        _label_internal_nodes(
            analysis_dir=analysis_dir,
            nodes=h.nodes,
            cleaned_texts=cleaned_texts,
            embeddings=emb.vectors,
            enable_ollama_for_internal=parsed_options["label_internal_nodes"],
            budget=llm_budget,
            max_llm_calls=min(INTERNAL_OLLAMA_MAX_CALLS, parsed_options["llm_label_budget"]),
        )
        _enrich_hierarchy_nodes(
            nodes=h.nodes,
            cleaned_texts=cleaned_texts,
            cluster_ids=h.cluster_ids,
            cluster_label_map=cluster_label_map,
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
        do_per_cluster = parsed_options["granulate_per_cluster"]
        per_cluster_max_rows = parsed_options["granulate_per_cluster_max_rows"]

        gran_results: list[tuple[int, dict[str, Any]]] = []
        selected_indices: list[int] = []
        aggregate: list[dict[str, Any]] = []
        per_cluster_aggregate: list[dict[str, Any]] = []

        if do_granulate:
            selected_indices = _sample_indices(len(raw_texts), granulate_max_rows)
            for idx in selected_indices:
                try:
                    out = granulate_text(raw_texts[idx])
                    gran_results.append((idx, out))
                except Exception:
                    continue
            aggregate = _aggregate_granulate([row[1] for row in gran_results])

            if do_per_cluster:
                per_cluster_aggregate = _aggregate_granulate_per_cluster(
                    texts=raw_texts,
                    cluster_ids=h.cluster_ids,
                    cluster_label_map=cluster_label_map,
                    max_rows=per_cluster_max_rows,
                )

        write_json(
            analysis_dir / "granulate_aggregate.json",
            {
                "mode": mode,
                "aggregate_aspect_summary": aggregate,
                "per_cluster_aggregate": per_cluster_aggregate,
                "items_included": len(selected_indices),
                "items_total": len(raw_texts),
                "item_ids_included": [item_ids[i] for i in selected_indices],
            },
        )
        if return_items and gran_results:
            gran_items_payload = []
            for idx, result in gran_results:
                gran_items_payload.append(
                    {
                        "id": item_ids[idx],
                        "preview": previews[idx],
                        "result": result,
                    }
                )
            write_json(analysis_dir / "granulate_items.json", gran_items_payload)
        timings["granulate_sec"] = round(time.perf_counter() - t0, 4)

        write_status(analysis_dir, status="processing", stage="overview", pct=98)
        top_aspects = aggregate[:8]
        overview = {
            "counts": {
                "items": len(raw_texts),
                "clusters": len(clusters_payload),
                "aspects": len(aggregate),
            },
            "top_clusters": sorted(clusters_payload, key=lambda c: int(c["size"]), reverse=True)[:5],
            "top_aspects": top_aspects,
            "timing": timings,
        }
        write_json(analysis_dir / "overview.json", overview)

        quality_warnings = _build_quality_warnings(raw_texts, cleaned_texts)
        insights = _build_insights(
            clusters_payload=clusters_payload,
            aggregate=aggregate,
            items_payload=items_payload,
            quality_warnings=quality_warnings,
            theme_count=parsed_options["insights_theme_count"],
        )
        write_json(analysis_dir / "insights.json", insights)

        timings["total_sec"] = round(time.perf_counter() - started, 4)
        write_json(analysis_dir / "timing.json", timings)
        write_status(analysis_dir, status="completed", stage="completed", pct=100)
    except Exception as exc:
        user_error = "Analysis failed while processing the dataset. Please verify your input and try again."
        debug_error = f"{exc.__class__.__name__}: {exc}"
        try:
            write_status(
                analysis_dir,
                status="failed",
                stage="failed",
                pct=100,
                error=user_error,
                debug_error=debug_error,
            )
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
        "per_cluster_aggregate": aggregate.get("per_cluster_aggregate", []),
        "items_included": int(aggregate.get("items_included", 0)),
        "items_total": int(aggregate.get("items_total", 0)),
        "items": [],
    }
    if include_items:
        items_file = analysis_dir / "granulate_items.json"
        if items_file.exists():
            payload["items"] = read_json(items_file)
    return payload
