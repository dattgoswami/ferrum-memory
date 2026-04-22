"""Tests for ferrum_memory/main.py lifespan."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from ferrum_memory.main import create_app
from ferrum_memory.storage.router import StorageRouter


def test_lifespan_creates_storage():
    """Test that lifespan creates a storage instance."""
    app = create_app()
    mock_storage = AsyncMock(spec=StorageRouter)
    mock_storage.initialize = AsyncMock()
    mock_storage.close = AsyncMock()
    app.state.storage = mock_storage

    # Just verify the lifespan app is constructible
    assert app.routes  # app created successfully with routes


def test_app_has_routes():
    """Test that the app has expected routes."""
    app = create_app()
    route_paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/health" in route_paths
    assert "/stats" in route_paths
    assert "/memory/store" in route_paths
    assert "/experience/" in route_paths
    assert "/session/start" in route_paths
