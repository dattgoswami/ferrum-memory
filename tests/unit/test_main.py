"""Tests for ferrum_memory/main.py."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ferrum_memory.main import create_app


def test_health_healthy():
    """Test /health returns healthy when all backends are up."""
    app = create_app()
    mock_storage = AsyncMock()
    mock_storage.health_check = AsyncMock(return_value={
        "qdrant": True, "sqlite": True, "redis": True
    })
    app.state.storage = mock_storage
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_health_degraded():
    """Test /health returns degraded when a backend is down."""
    app = create_app()
    mock_storage = AsyncMock()
    mock_storage.health_check = AsyncMock(return_value={
        "qdrant": False, "sqlite": True, "redis": True
    })
    app.state.storage = mock_storage
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


def test_stats():
    """Test /stats endpoint."""
    app = create_app()
    mock_storage = AsyncMock()
    mock_storage.health_check = AsyncMock(return_value={"qdrant": True, "sqlite": True, "redis": True})
    mock_storage.count_memory = AsyncMock(return_value=100)
    mock_storage.get_experience_stats = AsyncMock(return_value={"total_experiences": 50})
    app.state.storage = mock_storage
    client = TestClient(app)
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert resp.json()["memory_items"] == 100
    assert resp.json()["total_experiences"] == 50


def test_openapi_exists():
    """Test OpenAPI schema is generated."""
    app = create_app()
    client = TestClient(app)
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert "paths" in resp.json()
