"""Tests for ferrum_memory/api/experience.py."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ferrum_memory.api.experience import router
from ferrum_memory.storage.router import StorageRouter


def test_store_experience_success():
    """Test successful experience store."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.store_experience = AsyncMock()
    mock_storage.get_experience_stats = AsyncMock(return_value={"total_experiences": 0})
    mock_storage.update_td_error = AsyncMock()
    mock_storage.query_experiences = AsyncMock(return_value=[])

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.post("/experience/", json={
        "experience_id": "e1", "session_id": "s1", "task_id": "t1",
        "task_description": "d", "tool_call_sequence": [], "test_result": "pass",
        "reward": 0.5, "timestamp": 1000000.0, "td_error": 0.1,
    })
    assert resp.status_code == 200


def test_store_experience_bad_input():
    """Test 422 on bad input — query param for td-error update must be numeric."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    resp = client.put("/experience/td-error?experience_id=e1&td_error=not-a-number")
    assert resp.status_code == 422


def test_get_stats():
    """Test stats endpoint."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.store_experience = AsyncMock()
    mock_storage.get_experience_stats = AsyncMock(return_value={"total_experiences": 42})
    mock_storage.update_td_error = AsyncMock()
    mock_storage.query_experiences = AsyncMock(return_value=[])

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.get("/experience/stats")
    assert resp.status_code == 200
    assert resp.json()["total_experiences"] == 42


def test_update_td_error():
    """Test TD error update."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.store_experience = AsyncMock()
    mock_storage.get_experience_stats = AsyncMock(return_value={"total_experiences": 0})
    mock_storage.update_td_error = AsyncMock()
    mock_storage.query_experiences = AsyncMock(return_value=[])

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.put("/experience/td-error?experience_id=e1&td_error=0.5")
    assert resp.status_code == 200
    assert resp.json()["td_error"] == 0.5


def test_replay_prioritized():
    """Test replay with prioritized strategy."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.store_experience = AsyncMock()
    mock_storage.get_experience_stats = AsyncMock(return_value={"total_experiences": 0})
    mock_storage.update_td_error = AsyncMock()
    mock_storage.query_experiences = AsyncMock(return_value=[
        {"experience_id": "e1", "td_error": 0.5},
        {"experience_id": "e2", "td_error": 0.1},
    ])

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.get("/experience/replay?strategy=prioritized&k=2")
    assert resp.status_code == 200
    assert resp.json()["strategy"] == "prioritized"
    assert "samples" in resp.json()


def test_replay_uniform():
    """Test replay with uniform strategy."""
    app = FastAPI()
    app.include_router(router)

    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.store_experience = AsyncMock()
    mock_storage.get_experience_stats = AsyncMock(return_value={"total_experiences": 0})
    mock_storage.update_td_error = AsyncMock()
    mock_storage.query_experiences = AsyncMock(return_value=[
        {"experience_id": "e1", "td_error": 0.5},
    ])

    app.dependency_overrides[StorageRouter] = lambda: mock_storage

    client = TestClient(app)
    resp = client.get("/experience/replay?strategy=uniform&k=1")
    assert resp.status_code == 200
    assert resp.json()["strategy"] == "uniform"
