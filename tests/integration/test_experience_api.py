"""Integration tests for experience API — store -> replay cycle."""
from __future__ import annotations

import pytest

from ferrum_memory.config import Config
from ferrum_memory.storage.router import StorageRouter


@pytest.fixture
def storage():
    async def _make():
        s = StorageRouter(Config(SQLITE_PATH=":memory:"))
        await s.initialize()
        return s
    return _make


@pytest.mark.asyncio
async def test_store_and_replay(storage):
    """Test full store -> replay cycle with in-memory SQLite."""
    st = await storage()
    try:
        for i in range(5):
            exp = {
                "experience_id": f"exp-replay-{i}",
                "session_id": "s-replay",
                "task_id": f"task-{i}",
                "task_description": f"Task {i}",
                "tool_call_sequence": [],
                "test_result": "pass" if i % 2 == 0 else "fail",
                "reward": float(i) / 4.0,
                "attempt_number": 1,
                "duration_seconds": 10.0,
                "timestamp": 1000000.0 + i,
                "td_error": float(i) * 0.1,
            }
            await st.store_experience(exp)

        experiences = await st.query_experiences(session_id="s-replay")
        assert len(experiences) == 5

        from ferrum_memory.retrieval import ReplayBuffer
        buffer = ReplayBuffer()
        samples = buffer.sample(experiences, strategy="prioritized", k=3)
        assert len(samples) == 3
    finally:
        await st.close()


@pytest.mark.asyncio
async def test_td_error_update(storage):
    """Test TD error update and retrieval."""
    st = await storage()
    try:
        exp = {
            "experience_id": "exp-td-update",
            "session_id": "s-td",
            "task_id": "task-td",
            "task_description": "TD test",
            "tool_call_sequence": [],
            "test_result": "pass",
            "reward": 0.5,
            "attempt_number": 1,
            "duration_seconds": 5.0,
            "timestamp": 1000000.0,
            "td_error": 0.0,
        }
        await st.store_experience(exp)
        await st.update_td_error("exp-td-update", 0.9)

        updated = await st.get_experience("exp-td-update")
        assert updated is not None
        assert updated["td_error"] == 0.9
    finally:
        await st.close()


@pytest.mark.asyncio
async def test_experience_stats(storage):
    """Test stats aggregation."""
    st = await storage()
    try:
        for i in range(10):
            exp = {
                "experience_id": f"exp-stats-{i}",
                "session_id": "s-stats",
                "task_id": f"task-{i}",
                "task_description": f"Task {i}",
                "tool_call_sequence": [],
                "test_result": "pass",
                "reward": 0.5,
                "attempt_number": 1,
                "duration_seconds": 10.0,
                "timestamp": 1000000.0 + i,
                "td_error": 0.1,
            }
            await st.store_experience(exp)

        stats = await st.get_experience_stats()
        assert stats["total_experiences"] == 10
    finally:
        await st.close()
