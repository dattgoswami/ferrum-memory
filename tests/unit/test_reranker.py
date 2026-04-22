"""Tests for ferrum_memory/retrieval/reranker.py."""
from __future__ import annotations

import pytest

from ferrum_memory.retrieval.reranker import Reranker


class TestReranker:
    """Tests for the Reranker class."""

    def test_init_default_model(self):
        reranker = Reranker()
        assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker._loaded is False

    def test_init_custom_model(self):
        reranker = Reranker(model_name="custom/model")
        assert reranker._model_name == "custom/model"

    def test_rerank_without_model_returns_candidates(self):
        """If model not loaded, return candidates unchanged (opt-in behavior)."""
        reranker = Reranker()
        candidates = [{"id": "a", "content": "hello"}, {"id": "b", "content": "world"}]
        result = reranker.rerank("query", candidates, top_k=2)
        assert result == candidates

    def test_rerank_top_k(self):
        """Top-k filtering with unloaded model returns all candidates."""
        reranker = Reranker()
        candidates = [{"id": str(i), "content": f"doc {i}"} for i in range(10)]
        result = reranker.rerank("query", candidates, top_k=5)
        assert len(result) == 10  # Without model, all returned

    def test_rerank_empty_candidates(self):
        reranker = Reranker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []
