from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib import error, request

from services.storage import read_json, write_json

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"
WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _load_cache(path: Path) -> dict[str, str]:
    if path.exists():
        data = read_json(path)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    return {}


def _save_cache(path: Path, cache: dict[str, str]) -> None:
    write_json(path, cache)


def _postprocess_label(text: str) -> str:
    lines = (text or "").strip().splitlines()
    if not lines:
        return ""
    cleaned = lines[0]
    cleaned = cleaned.replace("`", "").replace('"', "").replace("'", "").strip()
    words = WORD_RE.findall(cleaned)
    if not words:
        return ""
    return " ".join(words[:3])


def _has_signal(text: str) -> bool:
    s = (text or "").strip()
    if len(s) < 3:
        return False
    return any(ch.isalpha() for ch in s)


def fallback_tfidf_label(snippets: list[str]) -> str:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = [s for s in snippets if s.strip()]
        if not docs:
            return "Topic"
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=3000)
        mat = vectorizer.fit_transform(docs)
        scores = mat.mean(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        if scores.size == 0:
            return "Topic"
        term = str(terms[int(scores.argmax())]).strip()
        return " ".join(term.split()[:3]).title() or "Topic"
    except Exception:
        return "Topic"


def _ollama_label(snippets: list[str], timeout_sec: int = 20) -> str:
    prompt = (
        "You label a text cluster.\n"
        "Return only 1 to 3 words.\n"
        "No punctuation. No quotes. No explanations.\n\n"
        "Snippets:\n"
        + "\n".join(f"- {s[:280]}" for s in snippets[:8])
        + "\n\nLabel:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 16},
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(OLLAMA_URL, method="POST", data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    parsed = json.loads(raw)
    response_text = str(parsed.get("response", "") or "")
    if not response_text.strip():
        return ""
    return _postprocess_label(response_text)


def generate_label(snippets: list[str], cache_path: Path, prompt_tag: str = "default") -> str:
    usable = [s.strip().replace("\n", " ")[:300] for s in snippets if _has_signal(s)]
    if not usable:
        return "Topic"

    cache = _load_cache(cache_path)
    key_payload: dict[str, Any] = {"model": OLLAMA_MODEL, "prompt_tag": prompt_tag, "snippets": usable[:8]}
    cache_key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    label = ""
    try:
        label = _ollama_label(usable)
    except (TimeoutError, error.URLError, json.JSONDecodeError, ValueError, IndexError):
        label = ""

    if not label:
        label = fallback_tfidf_label(usable)

    label = _postprocess_label(label) or "Topic"
    cache[cache_key] = label
    _save_cache(cache_path, cache)
    return label
