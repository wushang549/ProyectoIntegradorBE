from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import services.labeling as labeling


def _assert_valid_candidate_selection() -> None:
    original = labeling._ollama_generate
    try:
        labeling._ollama_generate = lambda _prompt, timeout_sec=20: (
            '{"label":"Shipping","confidence":0.91,"why":["keyword overlap","delivery context"]}'
        )
        snippets = [
            "Customer reports delayed shipping and wants a delivery date update.",
            "Tracking page has not moved for five days and shipment is late.",
            "Package delivery is delayed and support asked for shipping details.",
        ]
        with TemporaryDirectory() as td:
            cache_path = Path(td) / "cache.json"
            out = labeling.generate_label(
                snippets=snippets,
                cache_path=cache_path,
                prompt_tag="quickcheck_valid",
                top_keywords=["shipping delay", "delivery status", "tracking update"],
                candidates=["Shipping", "Billing", "Returns", "Customer Support"],
            )
        assert out == "Shipping", f"Expected Shipping, got {out}"
    finally:
        labeling._ollama_generate = original


def _assert_invalid_json_fallback() -> None:
    original = labeling._ollama_generate
    try:
        labeling._ollama_generate = lambda _prompt, timeout_sec=20: "this is not json"
        snippets = [
            "Billing issue: charged twice for the same invoice.",
            "Payment was duplicated and card was billed two times.",
            "Customer asks for refund after duplicate billing charge.",
        ]
        sanitized = [labeling.sanitize_text_for_llm(s) for s in snippets]
        expected = labeling.fallback_tfidf_label(sanitized)
        with TemporaryDirectory() as td:
            cache_path = Path(td) / "cache.json"
            out = labeling.generate_label(
                snippets=snippets,
                cache_path=cache_path,
                prompt_tag="quickcheck_fallback",
                top_keywords=["billing", "duplicate charge", "invoice"],
                candidates=["Billing", "Shipping", "Returns", "Other"],
            )
        assert out == expected, f"Expected fallback {expected}, got {out}"
    finally:
        labeling._ollama_generate = original


def main() -> None:
    _assert_valid_candidate_selection()
    _assert_invalid_json_fallback()
    print("labeling_quickcheck: ok")


if __name__ == "__main__":
    main()
