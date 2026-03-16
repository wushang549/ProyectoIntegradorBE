"""Regression tests for analysis insights summaries."""

from __future__ import annotations

from services.insights import _sanitize_summary, build_insights, build_overall_summary_fallback
from services.labeling import LabelingError


def test_build_insights_generates_llm_overall_summary(monkeypatch) -> None:
    """Insights should expose one cleaned overall summary when the LLM succeeds."""

    def fake_call_openai(_prompt: str, model_name: str) -> str:
        assert model_name == "gpt-5-nano"
        return (
            "<think>internal reasoning</think>\n"
            "Executive summary: Customers consistently praise food freshness, attentive staff, and a reliable dining "
            "experience, while the main friction points are long waits and a smaller set of slow-service complaints. "
            "Overall sentiment is positive, but operational consistency during busy periods remains the main risk."
        )

    monkeypatch.setattr("services.insights._call_openai_summary", fake_call_openai)

    insights = build_insights(
        total_items=40,
        clusters=[
            {
                "cluster_id": 1,
                "label": "Fresh Food",
                "size": 16,
                "top_terms": ["fresh", "delicious", "hot"],
                "representatives": [
                    "The food comes out quickly and is always hot and fresh",
                    "The flavors were great and everything tasted fresh",
                ],
                "dominant_aspect": "food",
                "dominant_polarity": "positive",
                "aspect_purity": 0.9,
            },
            {
                "cluster_id": 2,
                "label": "Wait Time",
                "size": 8,
                "top_terms": ["wait", "minutes", "reservation"],
                "representatives": [
                    "We also had to wait more than 40 minutes despite having a reservation",
                ],
                "dominant_aspect": "speed",
                "dominant_polarity": "negative",
                "aspect_purity": 0.86,
            },
        ],
        llm_model="gpt-5-nano",
    )

    assert insights["overall_summary_source"] == "llm"
    assert "<think>" not in insights["overall_summary"]
    assert "food freshness" in insights["overall_summary"].lower()


def test_build_insights_falls_back_when_llm_generation_fails(monkeypatch) -> None:
    """Insights should still return a general summary if OpenAI fails."""

    def fake_call_openai(_prompt: str, model_name: str) -> str:
        raise LabelingError(f"failed for {model_name}")

    monkeypatch.setattr("services.insights._call_openai_summary", fake_call_openai)

    insights = build_insights(
        total_items=30,
        clusters=[
            {
                "cluster_id": 1,
                "label": "Excellent Service",
                "size": 12,
                "top_terms": ["attentive", "friendly"],
                "representatives": ["Staff were attentive and friendly"],
                "dominant_polarity": "positive",
                "aspect_purity": 0.84,
            },
            {
                "cluster_id": 2,
                "label": "Slow Service",
                "size": 6,
                "top_terms": ["slow", "wait"],
                "representatives": ["We waited too long for the table"],
                "dominant_polarity": "negative",
                "aspect_purity": 0.8,
            },
        ],
        llm_model="gpt-5-nano",
    )

    assert insights["overall_summary_source"] == "heuristic"
    assert "Excellent Service" in insights["overall_summary"]
    assert "Slow Service" in insights["overall_summary"]


def test_build_overall_summary_fallback_handles_empty_input() -> None:
    """Fallback summary should stay stable for empty analyses."""

    assert build_overall_summary_fallback(total_items=0, clusters=[]) == "No analyzable comments were found."


def test_sanitize_summary_does_not_truncate_long_output() -> None:
    """Long LLM summaries should be preserved instead of being clipped."""

    long_text = " ".join(["detailed"] * 200)

    cleaned = _sanitize_summary(f"Executive summary: {long_text}")

    assert cleaned.endswith("detailed")
    assert "..." not in cleaned
