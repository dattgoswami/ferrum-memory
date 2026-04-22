"""Tests for ferrum_memory/api/memory.py compress with rerank."""
from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ferrum_memory.api.memory import router
from ferrum_memory.storage.router import StorageRouter


def test_compress_with_rerank():
    """Test compress with rerank=True."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.store_memory = AsyncMock()
    mock_storage.search_memory = AsyncMock(return_value=[{"id": "p1", "content": "test"}])
    mock_storage._qdrant = AsyncMock()
    mock_storage._qdrant.delete = AsyncMock()
    mock_storage.get_working_memory = AsyncMock(return_value={
        "session_id": "s1", "important_files": [], "recent_notes": [], "failed_approaches": []
    })

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.post("/memory/compress?session_id=s1&rerank=true")
    assert resp.status_code == 200
