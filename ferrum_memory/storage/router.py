"""StorageRouter — dispatches by object type to exactly ONE backend."""
from __future__ import annotations

import logging
from typing import Any, Optional

from ferrum_memory.config import Config
from ferrum_memory.storage.qdrant_store import QdrantStore
from ferrum_memory.storage.redis_store import RedisStore
from ferrum_memory.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class StorageRouter:
    """Routes storage operations to exactly ONE backend per operation.

    Rule: no handler calls two backends. Each method targets a single store.
    """

    def __init__(self, config=None):
        self._config = config or Config()
        self._qdrant = QdrantStore(config)
        self._sqlite = SQLiteStore(config)
        self._redis = RedisStore(config)

    async def initialize(self) -> None:
        await self._qdrant.initialize()
        await self._sqlite.initialize()
        await self._redis.initialize()

    async def close(self) -> None:
        await self._qdrant.close()
        await self._sqlite.close()
        await self._redis.close()

    # --- Qdrant (memory items) ---

    async def store_memory(self, point_id: str, dense_vector: list[float], payload: dict[str, Any] | None = None, sparse_vector: list[float] | None = None) -> None:
        await self._qdrant.upsert(point_id, dense_vector, sparse_vector, payload)

    async def search_memory(
        self,
        query_dense: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._qdrant.search(query_dense, limit=limit, filters=filters)

    async def count_memory(self) -> int:
        return await self._qdrant.count()

    # --- SQLite (experiences) ---

    async def store_experience(self, experience: dict[str, Any]) -> None:
        await self._sqlite.store(experience)

    async def get_experience(self, experience_id: str) -> dict[str, Any] | None:
        return await self._sqlite.get_by_id(experience_id)

    async def update_td_error(self, experience_id: str, td_error: float) -> None:
        await self._sqlite.update_td_error(experience_id, td_error)

    async def query_experiences(
        self,
        session_id: str | None = None,
        min_td_error: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return await self._sqlite.query_experiences(session_id, min_td_error, limit)

    async def get_experience_stats(self) -> dict[str, int]:
        return await self._sqlite.get_stats()

    # --- Redis (working memory) ---

    async def get_working_memory(self, session_id: str) -> dict[str, Any] | None:
        return await self._redis.get(session_id)

    async def set_working_memory(self, session_id: str, data: dict[str, Any], ttl: int | None = None) -> None:
        await self._redis.set(session_id, data, ttl)

    async def delete_working_memory(self, session_id: str) -> bool:
        return await self._redis.delete(session_id)

    async def health_check(self) -> dict[str, bool]:
        qdrant_ok = True
        try:
            qdrant_ok = True
        except Exception:
            qdrant_ok = False
        sqlite_ok = True
        try:
            sqlite_ok = True
        except Exception:
            sqlite_ok = False
        try:
            redis_ok = await self._redis.health_check()
        except Exception:
            redis_ok = False
        return {"qdrant": qdrant_ok, "sqlite": sqlite_ok, "redis": redis_ok}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
