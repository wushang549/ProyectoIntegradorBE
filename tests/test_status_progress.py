"""Regression tests for analysis status and progress stages."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from fastapi import HTTPException

from api import routes_analysis
from models.schemas import AnalysisOptions, GranulatedItem, IngestedRecord
from services.auth import AuthenticatedUser
from services import pipeline


def test_run_analysis_pipeline_reports_ai_summary_stage(monkeypatch) -> None:
    """The pipeline should report insights and AI summary as separate progress stages."""

    progress_calls: list[tuple[str, int, str, str]] = []

    monkeypatch.setattr(pipeline, "ensure_analysis_dir", lambda _analysis_id: None)
    monkeypatch.setattr(
        pipeline,
        "ingest",
        lambda _payload: [
            IngestedRecord(id="row_1", text="Great food"),
            IngestedRecord(id="row_2", text="Slow service"),
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "granulate_records",
        lambda _records, granulate=True: [
            GranulatedItem(
                id="item_1",
                text="Great food",
                source_id="row_1",
                source_text="Great food",
                chunk_index=0,
                metadata={},
            ),
            GranulatedItem(
                id="item_2",
                text="Slow service",
                source_id="row_2",
                source_text="Slow service",
                chunk_index=0,
                metadata={},
            ),
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "compute_embeddings",
        lambda _texts: SimpleNamespace(vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float), method="mock"),
    )
    monkeypatch.setattr(
        pipeline,
        "build_hierarchy",
        lambda _vectors: {"nodes": [], "linkage_matrix": [[0.0, 1.0, 0.1, 2.0]]},
    )
    monkeypatch.setattr(
        pipeline,
        "auto_cluster_with_buckets",
        lambda **_kwargs: {"cluster_labels": [0, 1], "k_clusters": 2},
    )
    monkeypatch.setattr(pipeline, "evaluate_cluster_partition", lambda **_kwargs: {"silhouette": 0.2})
    monkeypatch.setattr(
        pipeline,
        "build_cluster_summaries",
        lambda *_args, **_kwargs: [
            {"cluster_id": 0, "size": 1, "top_terms": ["food"], "representatives": ["Great food"]},
            {"cluster_id": 1, "size": 1, "top_terms": ["service"], "representatives": ["Slow service"]},
        ],
    )
    monkeypatch.setattr(pipeline, "_compute_umap_points", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        pipeline,
        "apply_labels",
        lambda **_kwargs: [
            {"cluster_id": 0, "size": 1, "top_terms": ["food"], "representatives": ["Great food"], "label": "Food"},
            {"cluster_id": 1, "size": 1, "top_terms": ["service"], "representatives": ["Slow service"], "label": "Service"},
        ],
    )
    monkeypatch.setattr(pipeline, "enrich_hierarchy_nodes", lambda **kwargs: kwargs["hierarchy_data"])
    monkeypatch.setattr(
        pipeline,
        "build_insight_heuristics",
        lambda **_kwargs: {
            "key_findings": ["Food appears in 50.0% of comments."],
            "theme_summary": "Top themes: Food, Service.",
            "quality_warnings": ["No major quality-risk pattern was detected in the current clusters."],
        },
    )
    monkeypatch.setattr(
        pipeline,
        "build_overall_summary",
        lambda **_kwargs: ("Overall summary text.", "llm"),
    )
    monkeypatch.setattr(pipeline, "default_openai_text_model", lambda: "gpt-5-nano")
    monkeypatch.setattr(pipeline, "_persist_to_supabase", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "write_json_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "write_embeddings", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "now_utc_iso", lambda: "2026-03-10T00:00:00+00:00")

    pipeline.run_analysis_pipeline(
        analysis_id="analysis-1",
        owner_id="user-1",
        payload=pipeline.PipelinePayload(input_type="text", text="Great food\nSlow service"),
        options=AnalysisOptions(),
        progress=lambda stage, pct, stage_label, message, _records, _items: progress_calls.append(
            (stage, pct, stage_label, message)
        ),
    )

    assert ("insights", 94, "Insights", "Extracting key findings and quality warnings.") in progress_calls
    assert ("ai_summary", 97, "AI Summary", "Generating overall AI summary with gpt-5-nano.") in progress_calls


def test_analysis_status_normalizes_ai_summary_and_preserves_raw_stage(monkeypatch) -> None:
    """Status responses should expose the frontend stage and the raw backend stage."""

    monkeypatch.setattr(
        routes_analysis,
        "_require_analysis_access",
        lambda _analysis_id, _owner_id: {"analysis_id": "analysis-1", "owner_id": "user-1"},
    )
    monkeypatch.setattr(
        routes_analysis.worker,
        "get_job_payload",
        lambda _analysis_id: {
            "analysis_id": "analysis-1",
            "status": "processing",
            "stage": "ai_summary",
            "pct": 97,
            "stage_label": "AI Summary",
            "message": "Generating overall AI summary with gpt-5-nano.",
            "elapsed_sec": 12.3,
            "updated_at": "2026-03-10T00:00:00+00:00",
            "error": None,
        },
    )

    payload = routes_analysis.analysis_status("analysis-1", current_user=AuthenticatedUser(user_id="user-1"))

    assert payload["progress"]["stage"] == "overview"
    assert payload["progress"]["raw_stage"] == "ai_summary"
    assert payload["progress"]["stage_label"] == "AI Summary"
    assert payload["progress"]["message"] == "Generating overall AI summary with gpt-5-nano."


def test_recent_analyses_normalizes_stage_and_keeps_raw_stage(monkeypatch) -> None:
    """Recent analyses should follow the same stage mapping as the status endpoint."""

    monkeypatch.setattr(
        routes_analysis,
        "list_recent",
        lambda limit=10, owner_id=None: [
            {
                "analysis_id": "analysis-1",
                "owner_id": owner_id,
                "created_at": "2026-03-10T00:00:00+00:00",
                "updated_at": "2026-03-10T00:00:30+00:00",
                "status": "processing",
                "stage": "ai_summary",
                "pct": 97,
                "total_records": 12,
                "total_items": 18,
            }
        ],
    )

    payload = routes_analysis.recent_analyses(limit=10, current_user=AuthenticatedUser(user_id="user-1"))

    assert payload["items"][0]["stage"] == "overview"
    assert payload["items"][0]["raw_stage"] == "ai_summary"
    assert payload["items"][0]["item_count"] == 18


def test_analysis_status_hides_other_users_analyses(monkeypatch) -> None:
    """Status should not expose analyses that belong to another user."""

    monkeypatch.setattr(
        routes_analysis,
        "get_index_entry",
        lambda _analysis_id: {"analysis_id": "analysis-1", "owner_id": "other-user"},
    )

    try:
        routes_analysis.analysis_status("analysis-1", current_user=AuthenticatedUser(user_id="user-1"))
    except HTTPException as exc:
        assert exc.status_code == 404
        assert exc.detail == "Analysis id not found."
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected HTTPException for unauthorized analysis access.")
