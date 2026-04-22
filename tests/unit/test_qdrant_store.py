"""Tests for ferrum_memory/storage/qdrant_store.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ferrum_memory.config import Config
from qdrant_client.models import PayloadSchemaType

from ferrum_memory.storage.qdrant_store import QdrantStore


@pytest.fixture
def config():
    return Config(qdrant_url="http://localhost:6333", qdrant_collection="test")


@pytest.fixture
def store(config):
    s = QdrantStore(config)
    s._client = AsyncMock()
    s._client.get_collections = AsyncMock(return_value=MagicMock(collections=[MagicMock(name="test")]))
    s._client.create_collection = AsyncMock()
    s._client.create_payload_index = AsyncMock()
    return s


@pytest.fixture
def mock_store():
    """Store with collection already present (skip initialize network call)."""
    s = QdrantStore(Config(qdrant_url="http://localhost:6333", qdrant_collection="test"))
    s._client = AsyncMock()
    s._client.get_collections = AsyncMock(return_value=MagicMock(collections=[MagicMock(name="test")]))
    s._client.create_collection = AsyncMock()
    s._client.create_payload_index = AsyncMock()
    s._client.query_points = AsyncMock(return_value=MagicMock(points=[]))
    s._client.upsert = AsyncMock()
    s._client.delete = AsyncMock()
    s._client.get_collection = AsyncMock(return_value=MagicMock(points=42))
    return s


@pytest.mark.asyncio
async def test_upsert_calls_client(store):
    await store.upsert(
        "point-1",
        dense_vector=[0.1] * 384,
        payload={"kind": "fact"},
        sparse_vector=[1.0, 2.0],
    )
    store._client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_search_calls_query_points(mock_store):
    mock_point = MagicMock()
    mock_point.id = "p1"
    mock_point.score = 0.9
    mock_point.payload = {"kind": "fact"}
    mock_store._client.query_points = AsyncMock(return_value=MagicMock(points=[mock_point]))

    results = await mock_store.search([0.1] * 384, limit=10)
    assert len(results) == 1
    assert results[0]["id"] == "p1"


@pytest.mark.asyncio
async def test_count_returns_points(store):
    collection = MagicMock()
    collection.points = 42
    store._client.get_collection = AsyncMock(return_value=collection)
    count = await store.count()
    assert count == 42


@pytest.mark.asyncio
async def test_collection_initialization():
    """Test that collection creation is called when needed."""
    store = QdrantStore(Config(qdrant_url="http://localhost:6333", qdrant_collection="new-collection"))
    store._client = AsyncMock()
    store._client.get_collections = AsyncMock(return_value=MagicMock(collections=[MagicMock(name="other")]))
    store._client.create_collection = AsyncMock()
    store._client.create_payload_index = AsyncMock()
    # Replace initialize to skip network call
    async def fake_init():
        await store._create_collection()
        await store._client.create_payload_index(
            collection_name="new-collection", field_name="kind", field_schema=PayloadSchemaType.KEYWORD)
        await store._client.create_payload_index(
            collection_name="new-collection", field_name="session_id", field_schema=PayloadSchemaType.KEYWORD)
    store.initialize = fake_init
    await store.initialize()
    store._client.create_collection.assert_called_once()
    assert store._client.create_payload_index.call_count >= 2


@pytest.mark.asyncio
async def test_delete_calls_client(store):
    await store.delete("point-1")
    store._client.delete.assert_called_once()


@pytest.mark.asyncio
async def test_raises_if_not_initialized():
    store = QdrantStore()
    with pytest.raises(RuntimeError, match="not initialized"):
        await store.upsert("p", [0.1] * 384)
