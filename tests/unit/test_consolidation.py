"""Tests for ferrum_memory/lifecycle/consolidation.py."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ferrum_memory.lifecycle.consolidation import consolidate


@pytest.mark.asyncio
async def test_consolidate_with_working_memory():
    storage = AsyncMock()
    storage.get_working_memory = AsyncMock(return_value={
        "session_id": "s1",
        "important_files": ["src/main.py", "tests/test_main.py"],
        "recent_notes": ["note 1", "note 2"],
        "failed_approaches": ["approach 1"],
    })
    storage.delete_working_memory = AsyncMock(return_value=True)

    summary = await consolidate("s1", storage)
    assert summary["session_id"] == "s1"
    assert "s1" in summary["summary"]
    assert "src/main.py" in summary["produced_files"]
    assert "note 1" in summary["key_patterns"]


@pytest.mark.asyncio
async def test_consolidate_without_working_memory():
    storage = AsyncMock()
    storage.get_working_memory = AsyncMock(return_value=None)
    storage.delete_working_memory = AsyncMock(return_value=True)

    summary = await consolidate("s2", storage)
    assert "No working memory found" in summary["summary"]


@pytest.mark.asyncio
async def test_consolidate_empty_working_memory():
    storage = AsyncMock()
    storage.get_working_memory = AsyncMock(return_value={
        "session_id": "s3",
        "important_files": [],
        "recent_notes": [],
        "failed_approaches": [],
    })
    storage.delete_working_memory = AsyncMock(return_value=True)

    summary = await consolidate("s3", storage)
    assert summary["session_id"] == "s3"
    assert summary["produced_files"] == []
