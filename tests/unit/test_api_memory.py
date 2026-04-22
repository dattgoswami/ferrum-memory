"""Tests for ferrum_memory/api/memory.py."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ferrum_memory.api.memory import router
from ferrum_memory.storage.router import StorageRouter


def test_store_memory_bad_input():
    """Test 422 on bad input (missing required fields)."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    resp = client.post("/memory/store?point_id=p1")
    assert resp.status_code == 422


def test_store_memory_success():
    """Test successful store."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.store_memory = AsyncMock()
    mock_storage.search_memory = AsyncMock(return_value=[])
    mock_storage._qdrant = AsyncMock()
    mock_storage._qdrant.delete = AsyncMock()

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.post(
        "/memory/store?point_id=p1",
        json={"dense_vector": [0.1] * 384, "payload": {"kind": "fact"}},
    )
    assert resp.status_code == 200


def test_search_memory_success():
    """Test successful search."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.store_memory = AsyncMock()
    mock_storage.search_memory = AsyncMock(return_value=[{"id": "p1", "score": 0.9}])
    mock_storage._qdrant = AsyncMock()
    mock_storage._qdrant.delete = AsyncMock()

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.post("/memory/search?query_dense=0.1,0.2", json={"query_dense": [0.1] * 384})
    assert resp.status_code == 200
    assert "results" in resp.json()
    assert len(resp.json()["results"]) == 1


def test_delete_memory_success():
    """Test successful delete."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.store_memory = AsyncMock()
    mock_storage.search_memory = AsyncMock(return_value=[])
    mock_storage._qdrant = AsyncMock()
    mock_storage._qdrant.delete = AsyncMock()

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.delete("/memory/p1")
    assert resp.status_code == 200


def test_compress_memory_success():
    """Test session compression."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.store_memory = AsyncMock()
    mock_storage.search_memory = AsyncMock(return_value=[])
    mock_storage._qdrant = AsyncMock()
    mock_storage._qdrant.delete = AsyncMock()
    mock_storage.get_working_memory = AsyncMock(return_value={
        "session_id": "s1", "important_files": [], "recent_notes": [], "failed_approaches": []
    })

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.post("/memory/compress?session_id=s1")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "s1"
