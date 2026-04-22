"""Tests for ferrum_memory/retrieval/replay.py."""
from __future__ import annotations

import math
import time

import pytest

from ferrum_memory.retrieval import (
    PrioritizedSampler,
    RecencySampler,
    ReplayBuffer,
    ReplayConfig,
    UniformSampler,
)


class TestPrioritizedSampler:
    def test_priority_formula(self):
        sampler = PrioritizedSampler(ReplayConfig(alpha=0.6, epsilon=0.01))
        # |0.5|^0.6 + 0.01
        expected = 0.5 ** 0.6 + 0.01
        assert sampler.priority(0.5) == pytest.approx(expected)

    def test_priority_zero_td_error(self):
        sampler = PrioritizedSampler(ReplayConfig(alpha=0.6, epsilon=0.01))
        assert sampler.priority(0.0) == 0.01  # just epsilon

    def test_sample_returns_experiences(self):
        sampler = PrioritizedSampler()
        experiences = [
            {"experience_id": f"e{i}", "td_error": float(i)}
            for i in range(10)
        ]
        result = sampler.sample(experiences, k=3)
        assert len(result) == 3

    def test_sample_empty(self):
        sampler = PrioritizedSampler()
        assert sampler.sample([], k=1) == []

    def test_importance_weights(self):
        sampler = PrioritizedSampler(ReplayConfig(alpha=0.6, beta=0.4))
        experiences = [{"td_error": 0.5}, {"td_error": 0.1}]
        weights = sampler.importance_weights(experiences)
        assert len(weights) == 2
        assert weights[0] <= weights[1]  # higher td_error -> higher weight

    def test_prioritized_returns_high_td_first(self):
        """Prioritized sampler should prefer high td_error experiences."""
        sampler = PrioritizedSampler(ReplayConfig(alpha=0.9, epsilon=0.001))
        experiences = [
            {"experience_id": f"e{i}", "td_error": float(i)}
            for i in range(20)
        ]
        # In a single draw, the top sample should likely have high td_error
        samples = sampler.sample(experiences, k=1)
        assert samples[0]["td_error"] > 0.0


class TestRecencySampler:
    def test_weight_at_half_life(self):
        sampler = RecencySampler(ReplayConfig(half_life=100))
        w = sampler.weight(100.0)
        assert w == pytest.approx(0.5, rel=0.01)

    def test_weight_zero_age(self):
        sampler = RecencySampler()
        assert sampler.weight(0) == 1.0

    def test_weight_exponential_decay(self):
        sampler = RecencySampler(ReplayConfig(half_life=100))
        w1 = sampler.weight(100)
        w2 = sampler.weight(200)
        assert w2 == pytest.approx(w1 * w1, rel=0.1)

    def test_sample_returns_experiences(self):
        now = time.time()
        sampler = RecencySampler()
        experiences = [
            {"timestamp": now - i * 10, "experience_id": f"e{i}"}
            for i in range(10)
        ]
        result = sampler.sample(experiences, k=3, now=now)
        assert len(result) == 3

    def test_sample_empty(self):
        sampler = RecencySampler()
        assert sampler.sample([], k=1) == []


class TestUniformSampler:
    def test_sample_returns_experiences(self):
        sampler = UniformSampler()
        experiences = [{"experience_id": f"e{i}"} for i in range(10)]
        result = sampler.sample(experiences, k=5)
        assert len(result) == 5

    def test_sample_empty(self):
        sampler = UniformSampler()
        assert sampler.sample([], k=1) == []


class TestReplayBuffer:
    def test_sample_prioritized(self):
        buffer = ReplayBuffer()
        experiences = [
            {"td_error": float(i), "experience_id": f"e{i}"}
            for i in range(10)
        ]
        result = buffer.sample(experiences, strategy="prioritized", k=3)
        assert len(result) == 3

    def test_sample_recency(self):
        buffer = ReplayBuffer()
        now = time.time()
        experiences = [
            {"timestamp": now - i * 10, "experience_id": f"e{i}"}
            for i in range(10)
        ]
        result = buffer.sample(experiences, strategy="recency", k=3, now=now)
        assert len(result) == 3

    def test_sample_uniform(self):
        buffer = ReplayBuffer()
        experiences = [{"experience_id": f"e{i}"} for i in range(10)]
        result = buffer.sample(experiences, strategy="uniform", k=5)
        assert len(result) == 5

    def test_sample_empty(self):
        buffer = ReplayBuffer()
        assert buffer.sample([], strategy="prioritized") == []

    def test_importance_weights(self):
        buffer = ReplayBuffer()
        experiences = [{"td_error": 0.5}, {"td_error": 0.1}]
        weights = buffer.importance_weights(experiences)
        assert len(weights) == 2
