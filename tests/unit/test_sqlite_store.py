"""Tests for ferrum_memory/storage/sqlite_store.py."""
from __future__ import annotations

import asyncio
import json

import pytest

from ferrum_memory.config import Config
from ferrum_memory.storage.sqlite_store import SQLiteStore


@pytest.fixture
def store():
    return SQLiteStore(Config(sqlite_path=":memory:"))


@pytest.fixture(autouse=True)
async def init_store(store):
    await store.initialize()
    yield
    await store.close()


@pytest.mark.asyncio
async def test_store_and_retrieve(store):
    exp = {
        "experience_id": "exp-unit-001",
        "session_id": "s1",
        "task_id": "t1",
        "task_description": "Test task",
        "tool_call_sequence": [{"tool_name": "read", "arguments": {}}],
        "test_result": "pass",
        "reward": 0.5,
        "attempt_number": 1,
        "timestamp": 1000000.0,
        "td_error": 0.1,
    }
    await store.store(exp)
    result = await store.get_by_id("exp-unit-001")
    assert result is not None
    assert result["experience_id"] == "exp-unit-001"
    assert result["reward"] == 0.5


@pytest.mark.asyncio
async def test_wal_mode():
    """WAL mode works with file-based SQLite (not :memory:)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = SQLiteStore(Config(sqlite_path=f.name))
        await s.initialize()
        cursor = await s._db.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        assert row[0] == "wal"
        await s.close()


@pytest.mark.asyncio
async def test_update_td_error(store):
    exp = {
        "experience_id": "exp-td-001",
        "session_id": "s1",
        "task_id": "t1",
        "task_description": "d",
        "tool_call_sequence": [],
        "test_result": "pass",
        "reward": 0.0,
        "timestamp": 1000000.0,
        "td_error": 0.0,
    }
    await store.store(exp)
    await store.update_td_error("exp-td-001", 0.5)
    updated = await store.get_by_id("exp-td-001")
    assert updated["td_error"] == 0.5


@pytest.mark.asyncio
async def test_query_experiences(store):
    for i in range(5):
        exp = {
            "experience_id": f"exp-query-{i}",
            "session_id": "s-query",
            "task_id": f"t-{i}",
            "task_description": f"Task {i}",
            "tool_call_sequence": [],
            "test_result": "pass",
            "reward": 0.5,
            "timestamp": 1000000.0 + i,
            "td_error": float(i) * 0.1,
        }
        await store.store(exp)

    results = await store.query_experiences(session_id="s-query")
    assert len(results) == 5


@pytest.mark.asyncio
async def test_get_stats(store):
    for i in range(3):
        exp = {
            "experience_id": f"exp-stats-{i}",
            "session_id": "s-stats",
            "task_id": f"t-{i}",
            "task_description": f"d",
            "tool_call_sequence": [],
            "test_result": "pass",
            "reward": 0.5,
            "timestamp": 1000000.0 + i,
            "td_error": 0.1,
        }
        await store.store(exp)

    stats = await store.get_stats()
    assert stats["total_experiences"] == 3


@pytest.mark.asyncio
async def test_query_with_td_filter(store):
    for i in range(5):
        exp = {
            "experience_id": f"exp-tdq-{i}",
            "session_id": "s-tdq",
            "task_id": f"t-{i}",
            "task_description": f"d",
            "tool_call_sequence": [],
            "test_result": "pass",
            "reward": 0.5,
            "timestamp": 1000000.0 + i,
            "td_error": float(i) * 0.2,
        }
        await store.store(exp)

    results = await store.query_experiences(session_id="s-tdq", min_td_error=0.4)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_get_nonexistent_returns_none(store):
    result = await store.get_by_id("exp-nonexistent")
    assert result is None
