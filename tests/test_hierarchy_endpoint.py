"""Tests for hierarchy endpoint response contract."""

from __future__ import annotations

from pathlib import Path

from api import routes_analysis


def test_hierarchy_handler_returns_frontend_contract(monkeypatch) -> None:
    """Hierarchy endpoint should return root_id + node map + leaves list."""

    fake_hierarchy = {
        "nodes": [
            {
                "node_id": "leaf_0",
                "parent_id": "node_2",
                "children_ids": [],
                "size": 1,
                "height": 0.0,
                "cohesion": 1.0,
                "similarity": 1.0,
                "descendant_leaf_count": 1,
            },
            {
                "node_id": "leaf_1",
                "parent_id": "node_2",
                "children_ids": [],
                "size": 1,
                "height": 0.0,
                "cohesion": 1.0,
                "similarity": 1.0,
                "descendant_leaf_count": 1,
            },
            {
                "node_id": "node_2",
                "parent_id": None,
                "children_ids": ["leaf_0", "leaf_1"],
                "size": 2,
                "height": 0.42,
                "cohesion": 0.58,
                "similarity": 0.58,
                "descendant_leaf_count": 2,
            },
        ],
        "linkage_matrix": [[0.0, 1.0, 0.42, 2.0]],
    }

    monkeypatch.setattr(routes_analysis, "_require_ready", lambda _analysis_id, _artifact_name: None)
    monkeypatch.setattr(routes_analysis, "hierarchy_file", lambda _analysis_id: Path("fake_hierarchy.json"))
    monkeypatch.setattr(
        routes_analysis,
        "_load_leaf_mappings",
        lambda _analysis_id: ({0: "row_0", 1: "row_1"}, {0: 3, 1: 7}),
    )
    monkeypatch.setattr(routes_analysis, "read_json_file", lambda _path: fake_hierarchy)

    result = routes_analysis.analysis_hierarchy("test-analysis")
    payload = result.model_dump() if hasattr(result, "model_dump") else result

    assert "root_id" in payload
    assert payload["root_id"]
    assert isinstance(payload["nodes"], dict)
    assert isinstance(payload["leaves"], list)
    assert payload["root_id"] in payload["nodes"]
