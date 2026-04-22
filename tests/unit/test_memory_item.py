"""Tests for contracts/memory_item.py."""
from __future__ import annotations

import json
import time

import pytest
from pydantic import ValidationError

from contracts.memory_item import MemoryItem, MemoryKind, MemorySource


class TestMemoryItem:
    """Tests for the MemoryItem schema."""

    def test_basic_construction(self):
        item = MemoryItem(
            memory_id="mem-001",
            tenant_id="tenant-1",
            session_id="session-1",
            kind=MemoryKind.FACT,
            content="test content",
            source=MemorySource.HUMAN,
        )
        assert item.memory_id == "mem-001"
        assert item.kind == MemoryKind.FACT
        assert item.tags == []
        assert item.ttl_seconds is None
        assert item.embedding_ref is None
        assert item.sparse_terms == []

    def test_serialization_round_trip(self):
        item = MemoryItem(
            memory_id="mem-002",
            tenant_id="tenant-2",
            session_id="session-2",
            kind=MemoryKind.PATTERN,
            content="pattern content",
            tags=["a", "b"],
            source=MemorySource.DREAM_CYCLE,
            ttl_seconds=3600,
            embedding_ref="emb-123",
            sparse_terms=["pattern"],
        )
        data = item.model_dump()
        restored = MemoryItem(**data)
        assert restored.memory_id == item.memory_id
        assert restored.kind == item.kind
        assert restored.tags == item.tags
        assert restored.sparse_terms == item.sparse_terms

    def test_json_schema_generation(self):
        schema = MemoryItem.model_json_schema()
        assert schema["title"] == "MemoryItem"
        assert "memory_id" in schema["properties"]
        assert "kind" in schema["properties"]
        assert "source" in schema["properties"]
        # Enums appear as $ref in pydantic v2 schema
        defs = schema.get("$defs", schema.get("definitions", {}))
        if isinstance(defs, dict):
            kind_def = defs.get("MemoryKind", {})
            assert kind_def.get("enum") == ["fact", "pattern", "learning", "escalation"]

    def test_required_field_raises(self):
        with pytest.raises(ValidationError):
            MemoryItem(
                memory_id="mem-003",
                # missing tenant_id
                session_id="session-1",
                kind=MemoryKind.FACT,
                content="missing tenant",
                source=MemorySource.HUMAN,
            )

    def test_validation_constraints(self):
        # ttl_seconds has no constraints, but kind must be a valid MemoryKind
        with pytest.raises(ValidationError):
            MemoryItem(
                memory_id="mem-004",
                tenant_id="t1",
                session_id="s1",
                kind="not_a_kind",
                content="c",
                source=MemorySource.HUMAN,
            )

    def test_optional_field_defaults(self):
        item = MemoryItem(
            memory_id="mem-005",
            tenant_id="t1",
            session_id="s1",
            kind=MemoryKind.FACT,
            content="c",
            source=MemorySource.HUMAN,
        )
        assert item.tags == []
        assert item.created_at == pytest.approx(time.time(), abs=2.0)
        assert item.ttl_seconds is None
        assert item.embedding_ref is None
        assert item.sparse_terms == []

    def test_is_expired_with_ttl(self):
        old_time = time.time() - 7200
        item = MemoryItem(
            memory_id="mem-006",
            tenant_id="t1",
            session_id="s1",
            kind=MemoryKind.FACT,
            content="c",
            source=MemorySource.HUMAN,
            created_at=old_time,
            ttl_seconds=3600,
        )
        assert item.is_expired() is True

    def test_is_not_expired_within_ttl(self):
        item = MemoryItem(
            memory_id="mem-007",
            tenant_id="t1",
            session_id="s1",
            kind=MemoryKind.PATTERN,
            content="c",
            source=MemorySource.WAKE_CYCLE,
            created_at=time.time(),
            ttl_seconds=3600,
        )
        assert item.is_expired() is False

    def test_is_not_expired_without_ttl(self):
        item = MemoryItem(
            memory_id="mem-008",
            tenant_id="t1",
            session_id="s1",
            kind=MemoryKind.LEARNING,
            content="c",
            source=MemorySource.HUMAN,
            ttl_seconds=None,
        )
        assert item.is_expired() is False

    def test_kind_enum_values(self):
        assert MemoryKind.FACT.value == "fact"
        assert MemoryKind.PATTERN.value == "pattern"
        assert MemoryKind.LEARNING.value == "learning"
        assert MemoryKind.ESCALATION.value == "escalation"

    def test_source_enum_values(self):
        assert MemorySource.DREAM_CYCLE.value == "dream_cycle"
        assert MemorySource.WAKE_CYCLE.value == "wake_cycle"
        assert MemorySource.HUMAN.value == "human"

    def test_to_json_round_trip(self):
        item = MemoryItem(
            memory_id="mem-009",
            tenant_id="t1",
            session_id="s1",
            kind=MemoryKind.ESCALATION,
            content="escalation content",
            source=MemorySource.HUMAN,
            tags=["critical"],
        )
        json_str = item.model_dump_json()
        parsed = MemoryItem.model_validate_json(json_str)
        assert parsed.memory_id == item.memory_id
        assert parsed.kind == item.kind
        assert parsed.content == item.content
