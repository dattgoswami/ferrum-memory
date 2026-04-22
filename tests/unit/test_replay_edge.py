"""Edge case tests for ferrum_memory/retrieval/__init__.py."""
from __future__ import annotations

import time

import pytest

from ferrum_memory.retrieval import ReplayBuffer, ReplayConfig


def test_uniform_sample_edge_cases():
    """Test uniform sampler edge cases."""
    buffer = ReplayBuffer()
    # Empty list
    assert buffer.sample([], strategy="uniform") == []
    # k > len
    items = [{"id": "a"}]
    result = buffer.sample(items, strategy="uniform", k=10)
    assert len(result) == 1


def test_prioritized_zero_priorities():
    """Test prioritized sampler when all priorities are zero."""
    buffer = ReplayBuffer(ReplayConfig(alpha=0.0))
    items = [{"td_error": 0.0}, {"td_error": 0.0}]
    result = buffer.sample(items, strategy="prioritized", k=2)
    assert len(result) == 2


def test_replay_config_defaults():
    """Test ReplayConfig defaults."""
    cfg = ReplayConfig()
    assert cfg.alpha == 0.6
    assert cfg.beta == 0.4
    assert cfg.epsilon == 0.01
    assert cfg.half_life == 21600.0


def test_replay_buffer_all_strategies():
    """Test all three strategies return consistent type."""
    buffer = ReplayBuffer()
    items = [{"td_error": float(i), "id": i} for i in range(5)]
    for strategy in ("prioritized", "recency", "uniform"):
        result = buffer.sample(items, strategy=strategy, k=3)
        assert isinstance(result, list)
        assert len(result) == 3
