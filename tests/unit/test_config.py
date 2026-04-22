"""Tests for ferrum_memory/config.py."""
from __future__ import annotations

import os

import pytest

from ferrum_memory.config import Config


class TestConfig:
    def test_default_qdrant_url(self):
        assert Config.QDRANT_URL == "http://localhost:6333"

    def test_default_redis_url(self):
        assert Config.REDIS_URL == "redis://localhost:6379/0"

    def test_default_sqlite_path(self):
        assert Config.SQLITE_PATH == ":memory:"

    def test_default_dense_dim(self):
        assert Config.FASTEMBED_DENSE_DIM == 384

    def test_default_replay_ttl(self):
        assert Config.DEFAULT_REPLAY_TTL == 86400

    def test_default_working_mem_ttl(self):
        assert Config.DEFAULT_WORKING_MEM_TTL == 86400

    def test_default_collection(self):
        assert Config.QDRANT_COLLECTION == "ferrum_memory"

    def test_default_bm25_enabled(self):
        assert Config.FASTEMBED_BM25 is True

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://custom:6333")
        monkeypatch.setenv("REDIS_URL", "redis://custom:6379")
        monkeypatch.setenv("SQLITE_PATH", "/tmp/ferrum.db")
        c = Config()
        assert c.QDRANT_URL == "http://custom:6333"
        assert c.REDIS_URL == "redis://custom:6379"
        assert c.SQLITE_PATH == "/tmp/ferrum.db"

    def test_otel_defaults(self):
        assert Config.OTEL_SERVICE_NAME == "ferrum-memory"
        assert Config.OTEL_EXPORTER_OTLP_ENDPOINT == "http://localhost:4317"
