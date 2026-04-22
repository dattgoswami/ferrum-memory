"""Extra tests for ferrum_memory/storage/sqlite_store.py coverage."""
from __future__ import annotations

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
async def test_close(store):
    """Test that close works."""
    await store.close()
    assert store._db is None


@pytest.mark.asyncio
async def test_store_and_get(store):
    """Test store and retrieve path."""
    exp = {
        "experience_id": "exp-get-1",
        "session_id": "s1",
        "task_id": "t1",
        "task_description": "d",
        "tool_call_sequence": [],
        "test_result": "pass",
        "reward": 0.5,
        "timestamp": 1000000.0,
        "td_error": 0.1,
    }
    await store.store(exp)
    result = await store.get_by_id("exp-get-1")
    assert result is not None
    assert result["experience_id"] == "exp-get-1"


@pytest.mark.asyncio
async def test_update_td_error_path(store):
    """Test update td_error path."""
    exp = {
        "experience_id": "exp-td-2",
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
    await store.update_td_error("exp-td-2", 0.9)
    result = await store.get_by_id("exp-td-2")
    assert result["td_error"] == 0.9


@pytest.mark.asyncio
async def test_query_experiences_path(store):
    """Test query experiences path."""
    for i in range(3):
        exp = {
            "experience_id": f"exp-q-{i}",
            "session_id": "s1",
            "task_id": f"t-{i}",
            "task_description": f"d",
            "tool_call_sequence": [],
            "test_result": "pass",
            "reward": 0.5,
            "timestamp": 1000000.0 + i,
            "td_error": float(i) * 0.1,
        }
        await store.store(exp)
    results = await store.query_experiences(session_id="s1")
    assert len(results) == 3


@pytest.mark.asyncio
async def test_get_stats_path(store):
    """Test get_stats path."""
    exp = {
        "experience_id": "exp-s1",
        "session_id": "s1",
        "task_id": "t1",
        "task_description": "d",
        "tool_call_sequence": [],
        "test_result": "pass",
        "reward": 0.5,
        "timestamp": 1000000.0,
        "td_error": 0.1,
    }
    await store.store(exp)
    stats = await store.get_stats()
    assert stats["total_experiences"] == 1
