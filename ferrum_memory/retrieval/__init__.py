"""Replay buffer — PER, recency, uniform samplers + hybrid search."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReplayConfig:
    """Configuration for the replay buffer."""

    alpha: float = 0.6  # prioritization exponent
    beta: float = 0.4  # importance sampling correction
    epsilon: float = 0.01  # minimum priority
    half_life: float = 21600.0  # seconds for recency decay


class PrioritizedSampler:
    """Schaul et al. 2015 prioritized sampler.

    Sample proportionally to |td_error|^alpha.
    """

    def __init__(self, config: ReplayConfig | None = None):
        self._config = config or ReplayConfig()

    def priority(self, td_error: float) -> float:
        return abs(td_error) ** self._config.alpha + self._config.epsilon

    def sample(
        self,
        experiences: list[dict[str, Any]],
        k: int = 1,
    ) -> list[dict[str, Any]]:
        if not experiences:
            return []
        priorities = [self.priority(e.get("td_error", 0.0)) for e in experiences]
        total = sum(priorities)
        if total <= 0:
            return self._uniform_sample(experiences, k)
        probs = [p / total for p in priorities]
        import random
        selected = random.choices(experiences, weights=probs, k=min(k, len(experiences)))
        return selected

    def importance_weights(self, experiences: list[dict[str, Any]]) -> list[float]:
        if not experiences:
            return []
        priorities = [self.priority(e.get("td_error", 0.0)) for e in experiences]
        total = sum(priorities)
        if total <= 0:
            return [1.0] * len(experiences)
        return [
            (p / total) ** (-self._config.beta) / (min(priorities) / total) ** (-self._config.beta)
            for p in priorities
        ]

    def _uniform_sample(self, experiences: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        import random
        return random.sample(experiences, min(k, len(experiences)))


class RecencySampler:
    """Exponential decay sampler.

    Weight at t=half_life is ~0.5.
    """

    def __init__(self, config: ReplayConfig | None = None):
        self._config = config or ReplayConfig()

    def weight(self, age_seconds: float) -> float:
        if age_seconds <= 0:
            return 1.0
        return 0.5 ** (age_seconds / self._config.half_life)

    def sample(
        self,
        experiences: list[dict[str, Any]],
        k: int = 1,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        if not experiences:
            return []
        if now is None:
            import time
            now = time.time()
        weights = []
        for e in experiences:
            age = now - e.get("timestamp", now)
            weights.append(max(self.weight(age), 1e-10))
        total = sum(weights)
        if total <= 0:
            return self._uniform_sample(experiences, k)
        probs = [w / total for w in weights]
        import random
        return random.choices(experiences, weights=probs, k=min(k, len(experiences)))

    def _uniform_sample(self, experiences: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        import random
        return random.sample(experiences, min(k, len(experiences)))


class UniformSampler:
    """Uniform random sampler."""

    def sample(
        self,
        experiences: list[dict[str, Any]],
        k: int = 1,
    ) -> list[dict[str, Any]]:
        if not experiences:
            return []
        import random
        return random.sample(experiences, min(k, len(experiences)))


class ReplayBuffer:
    """Dispatcher that wraps all three samplers."""

    def __init__(self, config: ReplayConfig | None = None):
        self._config = config or ReplayConfig()
        self._prioritized = PrioritizedSampler(config)
        self._recency = RecencySampler(config)
        self._uniform = UniformSampler()

    def sample(
        self,
        experiences: list[dict[str, Any]],
        strategy: str = "prioritized",
        k: int = 1,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        strategies = {"prioritized": self._prioritized, "recency": self._recency, "uniform": self._uniform}
        sampler = strategies[strategy]
        if strategy == "recency":
            return sampler.sample(experiences, k, now=now)
        return sampler.sample(experiences, k)

    def importance_weights(self, experiences: list[dict[str, Any]]) -> list[float]:
        return self._prioritized.importance_weights(experiences)
