"""Shared fixtures for the ferrum-memory test suite."""
from __future__ import annotations

import time

import pytest

from contracts.experience_tuple import ExperienceTuple, TestResult, ToolCallRecord
from contracts.memory_item import MemoryItem, MemoryKind, MemorySource
from contracts.working_memory import SessionSummary, WorkingMemoryState


# ------ MemoryItem fixtures ------


@pytest.fixture
def sample_memory_item() -> MemoryItem:
    return MemoryItem(
        memory_id="mem-001",
        tenant_id="tenant-1",
        session_id="session-1",
        kind=MemoryKind.FACT,
        content="The agent prefers ruff over black for formatting.",
        tags=["formatter", "ruff"],
        source=MemorySource.DREAM_CYCLE,
        created_at=time.time(),
        ttl_seconds=3600,
        embedding_ref="emb-001",
        sparse_terms=["ruff", "formatter", "agent"],
    )


@pytest.fixture
def permanent_memory_item() -> MemoryItem:
    return MemoryItem(
        memory_id="mem-002",
        tenant_id="tenant-1",
        session_id="session-1",
        kind=MemoryKind.PATTERN,
        content="Always check git status before committing.",
        source=MemorySource.WAKE_CYCLE,
        created_at=time.time(),
        ttl_seconds=None,
    )


# ------ ExperienceTuple fixtures ------


@pytest.fixture
def sample_tool_call() -> ToolCallRecord:
    return ToolCallRecord(
        tool_name="write_file",
        arguments={"path": "src/main.py", "content": "hello"},
        result="File written successfully.",
        duration_seconds=0.5,
    )


@pytest.fixture
def sample_experience() -> ExperienceTuple:
    return ExperienceTuple(
        experience_id="exp-001",
        session_id="session-1",
        task_id="task-001",
        task_description="Add null check to process()",
        tool_call_sequence=[
            ToolCallRecord(tool_name="read_file", arguments={"path": "src/utils.py"}),
            ToolCallRecord(
                tool_name="write_file",
                arguments={"path": "src/utils.py", "content": "# added null check"},
            ),
        ],
        test_result=TestResult.PASS,
        reward=0.8,
        attempt_number=1,
        duration_seconds=12.5,
        td_error=0.1,
    )


# ------ WorkingMemoryState fixtures ------


@pytest.fixture
def sample_working_memory() -> WorkingMemoryState:
    return WorkingMemoryState(
        session_id="session-1",
        current_task_id="task-001",
        important_files=["src/utils.py", "tests/test_utils.py"],
        recent_notes=["Added null check", "Tests passing"],
        failed_approaches=["try-except wrapper"],
    )


@pytest.fixture
def sample_session_summary() -> SessionSummary:
    return SessionSummary(
        session_id="session-1",
        summary="Completed null check implementation in utils.py.",
        key_patterns=["null checks prevent attribute errors"],
        key_learnings=["Always validate input before calling methods"],
        failed_approaches=["try-except wrapper was too broad"],
        produced_files=["src/utils.py"],
    )
