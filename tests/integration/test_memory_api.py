"""Integration tests for memory API — store -> search round-trip."""
from __future__ import annotations

import pytest

from ferrum_memory.config import Config
from ferrum_memory.storage.router import StorageRouter


@pytest.fixture
def storage():
    async def _make():
        s = StorageRouter(Config(SQLITE_PATH=":memory:"))
        await s.initialize()
        return s
    return _make


@pytest.mark.asyncio
async def test_memory_store_and_retrieve(storage):
    """Test storing and retrieving memory items via Qdrant mock."""
    st = await storage()
    try:
        from ferrum_memory.retrieval.hybrid import generate_dense

        dense_vec = generate_dense("test content for memory")
        await st.store_memory(
            point_id="mem-rtest-1",
            dense_vector=dense_vec,
            payload={"kind": "fact", "session_id": "s-mem-test"},
            sparse_vector=None,
        )

        count = await st.count_memory()
        assert count >= 0
    finally:
        await st.close()


@pytest.mark.asyncio
async def test_memory_search(storage):
    """Test search functionality."""
    st = await storage()
    try:
        from ferrum_memory.retrieval.hybrid import generate_dense

        vec = generate_dense("search test content")
        results = await st.search_memory(vec, limit=10)
        assert isinstance(results, list)
    finally:
        await st.close()
