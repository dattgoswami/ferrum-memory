"""MemoryItem — the fundamental unit of stored memory."""
from __future__ import annotations

import time
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MemoryKind(str, Enum):
    """Type of memory being stored."""

    FACT = "fact"
    PATTERN = "pattern"
    LEARNING = "learning"
    ESCALATION = "escalation"


class MemorySource(str, Enum):
    """Origin of the memory entry."""

    DREAM_CYCLE = "dream_cycle"
    WAKE_CYCLE = "wake_cycle"
    HUMAN = "human"


class MemoryItem(BaseModel):
    """A single memory entry in the ferrum-memory system.

    Stores structured facts, patterns, learnings, or escalations
    produced by the dream cycle, wake cycle, or human operators.
    """

    memory_id: str = Field(description="Unique identifier for this memory entry.")
    tenant_id: str = Field(description="Tenant that owns this memory.")
    session_id: str = Field(description="Session that produced this memory.")
    kind: MemoryKind = Field(description="Type of memory.")
    content: str = Field(description="Human-readable memory content.")
    tags: List[str] = Field(default_factory=list, description="Tags for filtering and grouping.")
    source: MemorySource = Field(description="Origin of this memory entry.")
    created_at: float = Field(default_factory=time.time, description="Unix timestamp of creation.")
    ttl_seconds: int | None = Field(default=None, description="Optional TTL in seconds. None = permanent.")
    embedding_ref: str | None = Field(default=None, description="Reference to the stored embedding vector.")
    sparse_terms: List[str] = Field(default_factory=list, description="Sparse terms for BM25 indexing.")

    def is_expired(self) -> bool:
        """Return True if this memory entry has exceeded its TTL."""
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.created_at) >= self.ttl_seconds

    model_config = ConfigDict(use_enum_values=False)
