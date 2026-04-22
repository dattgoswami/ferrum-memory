"""SQLite store — experience table CRUD with aiosqlite and WAL mode."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import aiosqlite

from ferrum_memory.config import Config

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS experience (
    experience_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_description TEXT NOT NULL,
    tool_call_sequence TEXT NOT NULL,
    test_result TEXT NOT NULL,
    reward REAL NOT NULL,
    attempt_number INTEGER NOT NULL DEFAULT 1,
    duration_seconds REAL,
    timestamp REAL NOT NULL,
    td_error REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_experience_session ON experience(session_id);
CREATE INDEX IF NOT EXISTS idx_experience_td_error ON experience(td_error);
"""


class SQLiteStore:
    """Async wrapper around SQLite for experience tuple storage."""

    def __init__(self, config=None):
        self._config = config or Config()
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open connection and enable WAL mode."""
        self._db = await aiosqlite.connect(self._config.SQLITE_PATH)
        await self._db.execute("PRAGMA journal_mode=WAL")
        # Split multi-statement SQL for aiosqlite
        for stmt in CREATE_TABLE_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                await self._db.execute(stmt)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def store(self, experience: dict[str, Any]) -> None:
        """Store an experience tuple."""
        if not self._db:
            raise RuntimeError("SQLiteStore not initialized.")
        await self._db.execute(
            """INSERT OR REPLACE INTO experience
               (experience_id, session_id, task_id, task_description,
                tool_call_sequence, test_result, reward, attempt_number,
                duration_seconds, timestamp, td_error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experience["experience_id"],
                experience["session_id"],
                experience["task_id"],
                experience["task_description"],
                json.dumps(experience.get("tool_call_sequence", [])),
                experience["test_result"],
                experience["reward"],
                experience.get("attempt_number", 1),
                experience.get("duration_seconds"),
                experience["timestamp"],
                experience.get("td_error", 0.0),
            ),
        )
        await self._db.commit()

    async def update_td_error(self, experience_id: str, td_error: float) -> None:
        if not self._db:
            raise RuntimeError("SQLiteStore not initialized.")
        await self._db.execute(
            "UPDATE experience SET td_error = ? WHERE experience_id = ?",
            (td_error, experience_id),
        )
        await self._db.commit()

    async def get_by_id(self, experience_id: str) -> dict[str, Any] | None:
        if not self._db:
            raise RuntimeError("SQLiteStore not initialized.")
        cursor = await self._db.execute(
            "SELECT * FROM experience WHERE experience_id = ?",
            (experience_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row, cursor.description)

    async def query_experiences(
        self,
        session_id: str | None = None,
        min_td_error: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query experiences with optional filters."""
        if not self._db:
            raise RuntimeError("SQLiteStore not initialized.")
        where_parts = []
        params: list[Any] = []
        if session_id:
            where_parts.append("session_id = ?")
            params.append(session_id)
        if min_td_error is not None:
            where_parts.append("td_error >= ?")
            params.append(min_td_error)
        where_clause = " WHERE " + " AND ".join(where_parts) if where_parts else ""
        query = f"SELECT * FROM experience{where_clause} ORDER BY td_error DESC LIMIT ?"
        params.append(limit)
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_dict(row, cursor.description) for row in rows]

    async def get_stats(self) -> dict[str, int]:
        """Get experience counts via SQL aggregation."""
        if not self._db:
            raise RuntimeError("SQLiteStore not initialized.")
        cursor = await self._db.execute("SELECT COUNT(*) FROM experience")
        total = (await cursor.fetchone())[0]
        return {"total_experiences": total}

    def _row_to_dict(self, row, description) -> dict[str, Any]:
        keys = [desc[0] for desc in description]
        result = dict(zip(keys, row))
        if "tool_call_sequence" in result and isinstance(result["tool_call_sequence"], str):
            result["tool_call_sequence"] = json.loads(result["tool_call_sequence"])
        return result
