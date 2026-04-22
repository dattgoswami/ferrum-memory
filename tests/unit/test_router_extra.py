"""Extra tests for ferrum_memory/storage/router.py coverage."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ferrum_memory.config import Config
from ferrum_memory.storage.router import StorageRouter


@pytest.fixture
def router():
    config = Config(sqlite_path=":memory:")
    router = StorageRouter(config)
    router._qdrant = AsyncMock()
    router._qdrant.upsert = AsyncMock()
    router._qdrant.search = AsyncMock(return_value=[])
    router._qdrant.count = AsyncMock(return_value=42)
    router._qdrant.delete = AsyncMock()
    router._sqlite = AsyncMock()
    router._sqlite.store = AsyncMock()
    router._sqlite.get_by_id = AsyncMock(return_value=None)
    router._sqlite.update_td_error = AsyncMock()
    router._sqlite.query_experiences = AsyncMock(return_value=[])
    router._sqlite.get_stats = AsyncMock(return_value={"total_experiences": 99})
    router._redis = AsyncMock()
    router._redis.get = AsyncMock(return_value=None)
    router._redis.set = AsyncMock()
    router._redis.delete = AsyncMock(return_value=1)
    router._redis.health_check = AsyncMock(return_value=True)
    return router


@pytest.mark.asyncio
async def test_count_memory(router):
    count = await router.count_memory()
    assert count == 42


@pytest.mark.asyncio
async def test_health_check(router):
    status = await router.health_check()
    assert status["qdrant"] is True
    assert status["sqlite"] is True
    assert status["redis"] is True


@pytest.mark.asyncio
async def test_context_manager():
    """Test that StorageRouter supports async context manager."""
    config = Config(sqlite_path=":memory:")
    router = StorageRouter(config)
    router._qdrant = AsyncMock()
    router._qdrant.upsert = AsyncMock()
    router._qdrant.search = AsyncMock(return_value=[])
    router._qdrant.count = AsyncMock(return_value=0)
    router._qdrant.delete = AsyncMock()
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
    async with router as r:
        assert r is router
    router._qdrant.close.assert_called_once()
    router._sqlite.close.assert_called_once()
    router._redis.close.assert_called_once()
