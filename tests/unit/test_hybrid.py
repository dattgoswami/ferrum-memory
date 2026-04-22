"""Tests for ferrum_memory/retrieval/hybrid.py."""
from __future__ import annotations

import math

import pytest

from ferrum_memory.retrieval.hybrid import generate_dense, generate_sparse, rrf_fusion


class TestRrfFusion:
    def test_fusion_over_two_lists(self):
        list1 = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]
        list2 = [{"id": "b", "score": 0.85}, {"id": "c", "score": 0.7}]
        results = rrf_fusion([list1, list2], k=10)
        # 'b' appears in both -> highest RRF score
        assert results[0]["id"] == "b"

    def test_fusion_single_list(self):
        lst = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]
        results = rrf_fusion([lst], k=60)
        assert results[0]["id"] == "a"

    def test_fusion_empty(self):
        assert rrf_fusion([]) == []

    def test_k_parameter(self):
        list1 = [{"id": f"e{i}", "score": 1.0 - i * 0.1} for i in range(10)]
        results = rrf_fusion([list1], k=5)
        assert len(results) == 10  # k affects ranking, not count


class TestGenerateDense:
    def test_returns_correct_dim(self):
        vec = generate_dense("test", dim=384)
        assert len(vec) == 384

    def test_normalized(self):
        vec = generate_dense("test", dim=384)
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0, abs=0.001)

    def test_deterministic(self):
        v1 = generate_dense("test", dim=64)
        v2 = generate_dense("test", dim=64)
        assert v1 == v2

    def test_different_content_different_vectors(self):
        v1 = generate_dense("hello", dim=64)
        v2 = generate_dense("world", dim=64)
        assert v1 != v2


class TestGenerateSparse:
    def test_returns_tuple(self):
        indices, values = generate_sparse("hello world hello")
        assert isinstance(indices, list)
        assert isinstance(values, list)

    def test_same_length(self):
        indices, values = generate_sparse("hello world")
        assert len(indices) == len(values)

    def test_term_frequency(self):
        indices, values = generate_sparse("hello hello world")
        # 'hello' appears twice, 'world' once
        hello_count = sum(1 for v in values if v > 1.0)
        assert hello_count >= 1
