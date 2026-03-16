"""Regression tests for pipeline quality heuristics."""

from __future__ import annotations

from models.schemas import IngestedRecord
from services.granulation import granulate_records
from services.labeling import apply_labels
from services.summaries import _extract_distinctive_terms


def test_granulation_splits_multi_aspect_review_into_clean_chunks() -> None:
    """Granulation should separate long mixed reviews into aspect-sized chunks."""

    record = IngestedRecord(
        id="row_1",
        text=(
            "We tried the ramen and it was light but delicious. "
            "The staff was friendly and checked on us often. "
            "The restaurant had a cozy atmosphere with warm lighting. "
            "It was slightly expensive but still worth it."
        ),
    )

    items = granulate_records([record], granulate=True)

    assert [item.text for item in items] == [
        "The ramen was light but delicious",
        "The staff was friendly and checked on us often",
        "The restaurant had a cozy atmosphere with warm lighting",
        "It was slightly expensive but still worth it",
    ]
    assert [item.metadata["_analysis"]["aspect"] for item in items] == [
        "food",
        "service",
        "atmosphere",
        "value",
    ]


def test_distinctive_terms_drop_template_noise() -> None:
    """Top-term extraction should prefer semantic terms over template scaffolding."""

    terms = _extract_distinctive_terms(
        all_texts=[
            "We tried the pizza and it was super tender",
            "I ordered the pizza but it was missing seasoning",
            "The staff was friendly and checked on us often",
            "The waiter gave great recommendations",
        ],
        grouped_indices={1: [0, 1], 2: [2, 3]},
        top_n=6,
    )

    food_terms = set(terms[1])
    service_terms = set(terms[2])

    assert "tried" not in food_terms
    assert "ordered" not in food_terms
    assert {"pizza", "seasoning"}.intersection(food_terms)
    assert "checked" not in service_terms
    assert "gave" not in service_terms
    assert "recommendations" in service_terms


def test_apply_labels_prefers_contextual_labels_over_generic_outputs(monkeypatch, tmp_path) -> None:
    """Labeling should choose a specific cluster name when the evidence is clear."""

    monkeypatch.setattr(
        "services.labeling._call_openai_label",
        lambda _prompt, model_name=None: "Value",
    )

    labeled = apply_labels(
        cluster_summaries=[
            {
                "cluster_id": 1,
                "size": 7,
                "top_terms": ["prices", "higher expected", "small portions"],
                "representatives": [
                    "Prices were higher than expected",
                    "Portions were small considering the price",
                ],
                "dominant_aspect": "value",
                "dominant_polarity": "negative",
                "aspect_purity": 0.9,
                "polarity_purity": 0.8,
            }
        ],
        k_clusters=1,
        cache_path=tmp_path / "labels.json",
    )

    assert labeled[0]["label"] == "Small Portions"
    assert labeled[0]["label_source"] == "contextual"


def test_apply_labels_scopes_cache_by_model(monkeypatch, tmp_path) -> None:
    """Different text models should not share the same cache key."""

    responses = {
        "gpt-5-nano": "Small Portions",
        "gpt-5-mini": "Price Concerns",
    }

    def fake_call_openai(_prompt: str, model_name: str | None = None) -> str:
        return responses[str(model_name)]

    monkeypatch.setattr("services.labeling._call_openai_label", fake_call_openai)

    payload = [
        {
            "cluster_id": 1,
            "size": 7,
            "top_terms": ["small portions", "prices", "price"],
            "representatives": [
                "Portions were small considering the price",
                "Prices were higher than expected",
            ],
            "dominant_aspect": "",
            "dominant_polarity": "",
            "aspect_purity": 0.0,
            "polarity_purity": 0.0,
        }
    ]

    nano = apply_labels(
        cluster_summaries=[dict(payload[0])],
        k_clusters=1,
        cache_path=tmp_path / "labels.json",
        model_name="gpt-5-nano",
    )
    mini = apply_labels(
        cluster_summaries=[dict(payload[0])],
        k_clusters=1,
        cache_path=tmp_path / "labels.json",
        model_name="gpt-5-mini",
    )

    assert nano[0]["label"] == "Small Portions"
    assert nano[0]["label_source"] == "openai"
    assert mini[0]["label"] == "Price Concerns"
    assert mini[0]["label_source"] == "openai"


def test_apply_labels_keeps_openai_label_when_it_matches_cluster_aspect(monkeypatch, tmp_path) -> None:
    """Aspect-grounded OpenAI labels should not be overwritten by heuristic labels."""

    monkeypatch.setattr(
        "services.labeling._call_openai_label",
        lambda _prompt, model_name=None: "Higher Prices",
    )

    labeled = apply_labels(
        cluster_summaries=[
            {
                "cluster_id": 1,
                "size": 9,
                "top_terms": ["prices", "higher expected", "small portions"],
                "representatives": [
                    "Prices were higher than expected",
                    "Portions were small considering the price",
                ],
                "dominant_aspect": "value",
                "dominant_polarity": "negative",
                "aspect_purity": 0.9,
                "polarity_purity": 0.8,
            }
        ],
        k_clusters=1,
        cache_path=tmp_path / "labels.json",
        model_name="gpt-5-nano",
    )

    assert labeled[0]["label"] == "Higher Prices"
    assert labeled[0]["label_source"] == "openai"
