"""Configuration — env var loading with spec defaults."""
from __future__ import annotations

import os


def _get(key: str, default: str) -> str:
    return os.getenv(key, default)


class AppSettings:
    """Application settings loaded from environment variables with spec defaults.

    Supports class-level access for defaults and instance-level overrides.
    Renamed from Config to avoid FastAPI dependency injection conflicts.
    """

    QDRANT_URL: str = _get("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = _get("QDRANT_COLLECTION", "ferrum_memory")
    QDRANT_PREFER_RECOMMEND: bool = _get("QDRANT_PREFER_RECOMMEND", "true").lower() == "true"
    REDIS_URL: str = _get("REDIS_URL", "redis://localhost:6379/0")
    SQLITE_PATH: str = _get("SQLITE_PATH", ":memory:")
    FASTEMBED_MODEL: str = _get("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
    FASTEMBED_DENSE_DIM: int = 384
    FASTEMBED_BM25: bool = _get("FASTEMBED_BM25", "true").lower() == "true"
    DEFAULT_REPLAY_TTL: int = int(_get("DEFAULT_REPLAY_TTL", "86400"))
    DEFAULT_WORKING_MEM_TTL: int = int(_get("DEFAULT_WORKING_MEM_TTL", "86400"))
    OTEL_SERVICE_NAME: str = _get("OTEL_SERVICE_NAME", "ferrum-memory")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = _get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    def __init__(
        self,
        qdrant_url: str | None = None,
        qdrant_collection: str | None = None,
        redis_url: str | None = None,
        sqlite_path: str | None = None,
        default_working_mem_ttl: int | None = None,
    ):
        self.QDRANT_URL = qdrant_url or _get("QDRANT_URL", AppSettings.QDRANT_URL)
        self.QDRANT_COLLECTION = qdrant_collection or _get("QDRANT_COLLECTION", AppSettings.QDRANT_COLLECTION)
        self.QDRANT_PREFER_RECOMMEND = _get("QDRANT_PREFER_RECOMMEND", "true").lower() == "true"
        self.REDIS_URL = redis_url or _get("REDIS_URL", AppSettings.REDIS_URL)
        self.SQLITE_PATH = sqlite_path or _get("SQLITE_PATH", AppSettings.SQLITE_PATH)
        self.FASTEMBED_MODEL = _get("FASTEMBED_MODEL", AppSettings.FASTEMBED_MODEL)
        self.FASTEMBED_DENSE_DIM = AppSettings.FASTEMBED_DENSE_DIM
        self.FASTEMBED_BM25 = _get("FASTEMBED_BM25", "true").lower() == "true"
        self.DEFAULT_REPLAY_TTL = int(_get("DEFAULT_REPLAY_TTL", str(AppSettings.DEFAULT_REPLAY_TTL)))
        self.DEFAULT_WORKING_MEM_TTL = default_working_mem_ttl if default_working_mem_ttl is not None else int(_get("DEFAULT_WORKING_MEM_TTL", str(AppSettings.DEFAULT_WORKING_MEM_TTL)))
        self.OTEL_SERVICE_NAME = _get("OTEL_SERVICE_NAME", AppSettings.OTEL_SERVICE_NAME)
        self.OTEL_EXPORTER_OTLP_ENDPOINT = _get("OTEL_EXPORTER_OTLP_ENDPOINT", AppSettings.OTEL_EXPORTER_OTLP_ENDPOINT)


# Backwards compat alias
Config = AppSettings
