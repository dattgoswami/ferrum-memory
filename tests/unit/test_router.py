"""Tests for ferrum_memory/storage/router.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ferrum_memory.config import Config
from ferrum_memory.storage.router import StorageRouter


@pytest.fixture
def router():
    config = Config(sqlite_path=":memory:")
    router = StorageRouter(config)
    # Mock all three stores
    router._qdrant = AsyncMock()
    router._qdrant.upsert = AsyncMock()
    router._qdrant.search = AsyncMock(return_value=[])
    router._qdrant.count = AsyncMock(return_value=0)
    router._sqlite = AsyncMock()
    router._sqlite.store = AsyncMock()
    router._sqlite.get_by_id = AsyncMock(return_value=None)
    router._sqlite.update_td_error = AsyncMock()
    router._sqlite.query_experiences = AsyncMock(return_value=[])
    router._sqlite.get_stats = AsyncMock(return_value={"total_experiences": 0})
    router._redis = AsyncMock()
    router._redis.get = AsyncMock(return_value=None)
    router._redis.set = AsyncMock()
    router._redis.delete = AsyncMock(return_value=1)
    router._redis.health_check = AsyncMock(return_value=True)
    return router


@pytest.mark.asyncio
async def test_store_memory_delegates_to_qdrant(router):
    await router.store_memory("p1", [0.1] * 384, {"kind": "fact"})
    router._qdrant.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_search_memory_delegates_to_qdrant(router):
    await router.search_memory([0.1] * 384, limit=10)
    router._qdrant.search.assert_called_once()


@pytest.mark.asyncio
async def test_store_experience_delegates_to_sqlite(router):
    await router.store_experience({"experience_id": "e1", "session_id": "s1", "task_id": "t1",
                                   "task_description": "d", "tool_call_sequence": [], "test_result": "pass",
                                   "reward": 0.0, "timestamp": 0.0, "td_error": 0.0})
    router._sqlite.store.assert_called_once()


@pytest.mark.asyncio
async def test_get_experience_delegates_to_sqlite(router):
    await router.get_experience("e1")
    router._sqlite.get_by_id.assert_called_once()


@pytest.mark.asyncio
async def test_update_td_error(router):
    await router.update_td_error("e1", 0.5)
    router._sqlite.update_td_error.assert_called_once()


@pytest.mark.asyncio
async def test_query_experiences(router):
    await router.query_experiences(session_id="s1")
    router._sqlite.query_experiences.assert_called_once()


@pytest.mark.asyncio
async def test_get_experience_stats(router):
    stats = await router.get_experience_stats()
    assert stats["total_experiences"] == 0


@pytest.mark.asyncio
async def test_get_working_memory_delegates_to_redis(router):
    await router.get_working_memory("s1")
    router._redis.get.assert_called_once()


@pytest.mark.asyncio
async def test_set_working_memory_delegates_to_redis(router):
    await router.set_working_memory("s1", {"data": "test"})
    router._redis.set.assert_called_once()


@pytest.mark.asyncio
async def test_delete_working_memory(router):
    result = await router.delete_working_memory("s1")
    assert result


@pytest.mark.asyncio
async def test_health_check(router):
    status = await router.health_check()
    assert "qdrant" in status
    assert "sqlite" in status
    assert "redis" in status
