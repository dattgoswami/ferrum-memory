"""Shared fixtures for integration tests."""
from __future__ import annotations

import pytest


@pytest.fixture
def sample_experience_data():
    return {
        "experience_id": "exp-int-001",
        "session_id": "session-int-1",
        "task_id": "task-int-1",
        "task_description": "Integration test task",
        "tool_call_sequence": [],
        "test_result": "pass",
        "reward": 0.5,
        "attempt_number": 1,
        "duration_seconds": 10.0,
        "timestamp": 1000000.0,
        "td_error": 0.1,
    }


@pytest.fixture
def sample_memory_data():
    return {
        "point_id": "mem-int-001",
        "dense_vector": [0.0] * 384,
        "payload": {"kind": "fact", "session_id": "session-int-1"},
    }


@pytest.fixture
def sample_working_memory_data():
    return {
        "session_id": "session-int-1",
        "current_task_id": "task-1",
        "important_files": ["src/main.py"],
        "recent_notes": ["integration test note"],
        "failed_approaches": [],
    }
