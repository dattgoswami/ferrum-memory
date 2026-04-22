"""Integration tests for session API — start -> close consolidation."""
from __future__ import annotations

import pytest

from ferrum_memory.config import Config
from ferrum_memory.storage.router import StorageRouter
from ferrum_memory.lifecycle.consolidation import consolidate


@pytest.fixture
def storage():
    async def _make():
        s = StorageRouter(Config(SQLITE_PATH=":memory:"))
        await s.initialize()
        return s
    return _make


@pytest.mark.asyncio
async def test_session_start_close(storage):
    """Test full session lifecycle with consolidation."""
    st = await storage()
    try:
        session_id = "session-lifecycle-1"

        await st.set_working_memory(session_id, {
            "session_id": session_id,
            "current_task_id": "task-1",
            "important_files": ["src/main.py"],
            "recent_notes": ["Started task"],
            "failed_approaches": [],
        })

        wm = await st.get_working_memory(session_id)
        assert wm is not None
        assert wm["session_id"] == session_id

        summary = await consolidate(session_id, st)
        assert summary["session_id"] == session_id
        assert "Started task" in summary["key_patterns"]

        await st.delete_working_memory(session_id)
        wm_after = await st.get_working_memory(session_id)
        assert wm_after is None
    finally:
        await st.close()


@pytest.mark.asyncio
async def test_consolidate_nonexistent_session(storage):
    """Test consolidation on a session with no working memory."""
    st = await storage()
    try:
        summary = await consolidate("nonexistent-session", st)
        assert "No working memory found" in summary["summary"]
    finally:
        await st.close()
