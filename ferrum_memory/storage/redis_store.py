"""Redis store — WorkingMemoryState get/set with TTL."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import redis.asyncio as redis

from ferrum_memory.config import Config

logger = logging.getLogger(__name__)


class RedisStore:
    """Async wrapper around Redis for working memory storage."""

    def __init__(self, config=None):
        self._config = config or Config()
        self._client: redis.Redis | None = None

    async def initialize(self) -> None:
        self._client = redis.from_url(self._config.REDIS_URL, decode_responses=True)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    def _key(self, session_id: str) -> str:
        return f"ferrum:wm:{session_id}"

    async def get(self, session_id: str) -> dict[str, Any] | None:
        """Get working memory for a session."""
        if not self._client:
            raise RuntimeError("RedisStore not initialized.")
        raw = await self._client.get(self._key(session_id))
        if raw is None:
            return None
        return json.loads(raw)

    async def set(self, session_id: str, data: dict[str, Any], ttl: int | None = None) -> None:
        """Set working memory with TTL (default 86400s)."""
        if not self._client:
            raise RuntimeError("RedisStore not initialized.")
        if ttl is None:
            ttl = self._config.DEFAULT_WORKING_MEM_TTL
        await self._client.setex(self._key(session_id), ttl, json.dumps(data))

    async def delete(self, session_id: str) -> bool:
        if not self._client:
            raise RuntimeError("RedisStore not initialized.")
        return (await self._client.delete(self._key(session_id))) > 0

    async def health_check(self) -> bool:
        if not self._client:
            return False
        try:
            return await self._client.ping()
        except Exception:
            return False
