"""Tests for grounded analysis chat context and route wiring."""

from __future__ import annotations

from pathlib import Path

from api import routes_analysis
from services.analysis_chat import build_analysis_chat_context, generate_analysis_chat_answer
from services.auth import AuthenticatedUser


def test_build_analysis_chat_context_includes_core_sections_and_selection() -> None:
    """The analysis chat context should summarize artifacts and active selection."""

    context, sections = build_analysis_chat_context(
        current_question="What themes stand out?",
        items_payload=[],
        overview_payload={
            "counts": {"items": 12, "clusters": 3, "aspects": 2},
            "top_clusters": [{"cluster_id": 0, "label": "Service", "size": 5}],
            "top_aspects": [{"aspect": "speed", "count": 4}],
        },
        insights_payload={
            "overall_summary": "Service issues dominate the dataset.",
            "key_findings": ["Service delays are concentrated in one cluster."],
            "quality_warnings": ["Small sample size for one branch."],
            "theme_summary": [
                {"label": "Service", "size": 5, "top_terms": ["slow", "queue"], "examples": ["Long wait time"]}
            ],
        },
        clusters_payload={
            "clusters": [
                {
                    "cluster_id": 0,
                    "label": "Service",
                    "size": 5,
                    "top_terms": ["slow", "queue"],
                    "representatives": [{"preview": "Long wait time"}],
                }
            ]
        },
        map_payload={
            "points": [{"id": "row_1", "x": 0.1, "y": 0.2, "cluster_label": "Service", "preview": "Long wait time"}],
            "clusters": [{"cluster_id": 0, "label": "Service"}],
        },
        hierarchy_payload={
            "root_id": "root",
            "nodes": {
                "root": {"label": "All feedback", "descendant_leaf_count": 12, "size": 12, "parent_id": None},
                "node_1": {
                    "label": "Service",
                    "descendant_leaf_count": 5,
                    "size": 5,
                    "parent_id": "root",
                    "summary": "Queue and speed complaints.",
                },
            },
        },
        selection={"selected_cluster_id": 0, "selected_point_id": "row_1", "selected_node_id": "node_1"},
    )

    assert "OVERVIEW" in context
    assert "INSIGHTS" in context
    assert "CLUSTERS" in context
    assert "MAP" in context
    assert "HIERARCHY" in context
    assert "SELECTION" in context
    assert "Selected cluster: Service (5)" in context
    assert "Selected point: row_1" in context
    assert "Selected hierarchy node: Service (5)" in context
    assert sections == ["overview", "insights", "clusters", "map", "hierarchy", "selection"]


def test_generate_analysis_chat_answer_returns_grounding(monkeypatch) -> None:
    """The chat service should call OpenAI with grounded context and return metadata."""

    captured: dict[str, str] = {}

    def fake_request(prompt: str, **kwargs) -> str:
        captured["prompt"] = prompt
        captured["model_name"] = str(kwargs.get("model_name") or "")
        return "The largest theme is Service."

    monkeypatch.setattr("services.analysis_chat.request_openai_text", fake_request)

    result = generate_analysis_chat_answer(
        analysis_id="analysis-1",
        messages=[{"role": "user", "content": "What is the largest theme?"}],
        items_payload=[],
        overview_payload={"counts": {"items": 12, "clusters": 3, "aspects": 2}, "top_clusters": [], "top_aspects": []},
        insights_payload={"key_findings": [], "theme_summary": [], "quality_warnings": [], "overall_summary": ""},
        clusters_payload={"clusters": []},
        map_payload={"points": [], "clusters": []},
        hierarchy_payload={"root_id": "", "nodes": {}},
        selection=None,
        model_name="gpt-5-nano",
    )

    assert "Current user question:\n\nWhat is the largest theme?" in captured["prompt"]
    assert result["answer"] == "The largest theme is Service."
    assert result["grounding"]["analysis_id"] == "analysis-1"
    assert result["model"] == "gpt-5-nano"


def test_build_analysis_chat_context_includes_retrieved_negative_examples() -> None:
    """Negative row-level questions should include a few raw supporting comments."""

    context, sections = build_analysis_chat_context(
        current_question="What is the most negative comment?",
        items_payload=[
            {
                "id": "row_1",
                "text": "The food was greasy and bland, and service was slow.",
                "metadata": {"_analysis": {"aspect": "food", "polarity": "negative", "polarity_confidence": 0.9}},
            },
            {
                "id": "row_2",
                "text": "Great flavors and friendly staff.",
                "metadata": {"_analysis": {"aspect": "service", "polarity": "positive", "polarity_confidence": 0.8}},
            },
        ],
        overview_payload={"counts": {"items": 2, "clusters": 1, "aspects": 1}, "top_clusters": [], "top_aspects": []},
        insights_payload={"key_findings": [], "theme_summary": [], "quality_warnings": [], "overall_summary": ""},
        clusters_payload={
            "clusters": [],
            "item_cluster_map": [{"id": "row_1", "cluster_id": 0}, {"id": "row_2", "cluster_id": 0}],
        },
        map_payload={"points": [], "clusters": []},
        hierarchy_payload={"root_id": "", "nodes": {}},
        selection=None,
    )

    assert "RAW_EXAMPLES" in context
    assert "row_1 | cluster 0 | polarity=negative | aspect=food" in context
    assert "greasy and bland" in context
    assert "raw_examples" in sections


def test_analysis_chat_handler_returns_grounded_answer(monkeypatch) -> None:
    """The chat route should load artifacts and return one grounded answer."""

    monkeypatch.setattr(
        routes_analysis,
        "_require_analysis_access",
        lambda _analysis_id, _owner_id: {"analysis_id": "analysis-1", "owner_id": "user-1"},
    )
    monkeypatch.setattr(routes_analysis, "_require_ready", lambda _analysis_id, _artifact_name, _owner_id: False)
    monkeypatch.setattr(routes_analysis, "_build_chat_overview_payload", lambda _analysis_id: {"counts": {}})
    monkeypatch.setattr(
        routes_analysis,
        "_build_chat_insights_payload",
        lambda _analysis_id: {"key_findings": [], "theme_summary": [], "quality_warnings": [], "overall_summary": ""},
    )
    monkeypatch.setattr(
        routes_analysis,
        "_build_cluster_payload",
        lambda analysis_id, requested_k=None: {"clusters": [], "cluster_labels": [], "item_cluster_map": []},
    )
    monkeypatch.setattr(
        routes_analysis,
        "_build_map_payload_from_artifacts",
        lambda analysis_id, cluster_payload: {"points": [], "clusters": []},
    )
    monkeypatch.setattr(routes_analysis, "hierarchy_file", lambda _analysis_id: Path("fake_hierarchy.json"))
    monkeypatch.setattr(routes_analysis, "read_json_file", lambda _path: {"nodes": [], "linkage_matrix": []})
    monkeypatch.setattr(routes_analysis, "_ensure_hierarchy_enriched", lambda analysis_id, owner_id, hierarchy_data: hierarchy_data)
    monkeypatch.setattr(
        routes_analysis,
        "_build_hierarchy_response",
        lambda analysis_id, hierarchy_data: routes_analysis.HierarchyResponse(root_id="root", nodes={}, leaves=[]),
    )
    monkeypatch.setattr(routes_analysis, "_load_analysis_llm_model", lambda *args, **kwargs: "gpt-5-nano")
    monkeypatch.setattr(
        routes_analysis,
        "generate_analysis_chat_answer",
        lambda **kwargs: {
            "answer": "The main theme is Service.",
            "model": "gpt-5-nano",
            "grounding": {"analysis_id": kwargs["analysis_id"], "sections": ["overview"], "selection_applied": False},
        },
    )

    payload = routes_analysis.AnalysisChatRequest.model_validate(
        {"messages": [{"role": "user", "content": "What is the main theme?"}]}
    )
    result = routes_analysis.analysis_chat(
        "analysis-1",
        payload=payload,
        current_user=AuthenticatedUser(user_id="user-1"),
    )

    assert result["answer"] == "The main theme is Service."
    assert result["grounding"]["analysis_id"] == "analysis-1"
