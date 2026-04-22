"""Tests for ferrum_memory/storage/redis_store.py."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ferrum_memory.config import Config
from ferrum_memory.storage.redis_store import RedisStore


@pytest.fixture
def config():
    return Config(redis_url="redis://localhost:6379/0", default_working_mem_ttl=86400)


@pytest.fixture
def store(config):
    with patch("ferrum_memory.storage.redis_store.redis") as mock_redis:
        store = RedisStore(config)
        store._client = AsyncMock()
        store._client.setex = AsyncMock()
        store._client.get = AsyncMock()
        store._client.delete = AsyncMock()
        store._client.ping = AsyncMock(return_value=True)
        yield store


@pytest.mark.asyncio
async def test_set_with_ttl(store, config):
    await store.set("session-1", {"data": "test"})
    store._client.setex.assert_called_once()
    call_args = store._client.setex.call_args
    assert call_args[0][1] == config.DEFAULT_WORKING_MEM_TTL


@pytest.mark.asyncio
async def test_get_returns_data(store):
    data = {"session_id": "session-1", "notes": ["test"]}
    store._client.get = AsyncMock(return_value=json.dumps(data))
    result = await store.get("session-1")
    assert result == data


@pytest.mark.asyncio
async def test_get_returns_none_for_missing(store):
    store._client.get = AsyncMock(return_value=None)
    result = await store.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_delete(store):
    store._client.delete = AsyncMock(return_value=1)
    result = await store.delete("session-1")
    assert result is True


@pytest.mark.asyncio
async def test_health_check(store):
    result = await store.health_check()
    assert result is True


@pytest.mark.asyncio
async def test_key_format():
    store = RedisStore(Config())
    assert store._key("session-1") == "ferrum:wm:session-1"
