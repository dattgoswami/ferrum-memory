"""Tests for ferrum_memory/api/session.py."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ferrum_memory.api.session import router
from ferrum_memory.storage.router import StorageRouter


def test_start_session():
    """Test session start."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.set_working_memory = AsyncMock()
    mock_storage.get_working_memory = AsyncMock(return_value=None)
    mock_storage.delete_working_memory = AsyncMock(return_value=True)
    mock_storage.store_memory = AsyncMock()
    mock_storage.search_memory = AsyncMock(return_value=[])
    mock_storage._qdrant = AsyncMock()
    mock_storage._qdrant.delete = AsyncMock()

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.post("/session/start?session_id=s1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "started"


def test_get_session():
    """Test get session."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.set_working_memory = AsyncMock()
    mock_storage.get_working_memory = AsyncMock(return_value={
        "session_id": "s1", "current_task_id": "t1", "important_files": [],
        "recent_notes": [], "failed_approaches": [],
    })
    mock_storage.delete_working_memory = AsyncMock(return_value=True)
    mock_storage.store_memory = AsyncMock()
    mock_storage.search_memory = AsyncMock(return_value=[])
    mock_storage._qdrant = AsyncMock()
    mock_storage._qdrant.delete = AsyncMock()

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.get("/session/s1")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "s1"


def test_update_session():
    """Test update session."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.set_working_memory = AsyncMock()
    mock_storage.get_working_memory = AsyncMock(return_value=None)
    mock_storage.delete_working_memory = AsyncMock(return_value=True)
    mock_storage.store_memory = AsyncMock()
    mock_storage.search_memory = AsyncMock(return_value=[])
    mock_storage._qdrant = AsyncMock()
    mock_storage._qdrant.delete = AsyncMock()

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.put("/session/s1", json={"notes": ["test"]})
    assert resp.status_code == 200
    assert resp.json()["status"] == "updated"


def test_close_session():
    """Test session close with consolidation."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.set_working_memory = AsyncMock()
    mock_storage.get_working_memory = AsyncMock(return_value={
        "session_id": "s1", "important_files": ["src/main.py"],
        "recent_notes": ["note 1"], "failed_approaches": [],
    })
    mock_storage.delete_working_memory = AsyncMock(return_value=True)
    mock_storage.store_memory = AsyncMock()
    mock_storage.search_memory = AsyncMock(return_value=[])
    mock_storage._qdrant = AsyncMock()
    mock_storage._qdrant.delete = AsyncMock()

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.post("/session/close?session_id=s1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "closed"
    assert "summary" in resp.json()
