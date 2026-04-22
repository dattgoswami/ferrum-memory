"""Reranker — opt-in cross-encoder reranking.

Lazy-loads cross-encoder/ms-marco-MiniLM-L-6-v2.
Only activated by ?rerank=true query param — not applied by default.
"""
from __future__ import annotations

from typing import Any


class Reranker:
    """Opt-in reranker using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            self._loaded = True
        except ImportError:
            self._loaded = False

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank candidates by cross-encoder scores.

        If the model is not available, falls back to returning candidates
        unchanged (opt-in behavior).
        """
        if not self._loaded:
            self._ensure_loaded()
            if not self._loaded:
                return candidates

        pairs = [(query, c.get("content", "")) for c in candidates]
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        scores = self._model.predict(pairs)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]
