"""Tests for contracts/working_memory.py."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from contracts.working_memory import SessionSummary, WorkingMemoryState


class TestWorkingMemoryState:
    """Tests for the WorkingMemoryState schema."""

    def test_basic_construction(self):
        state = WorkingMemoryState(session_id="session-1")
        assert state.session_id == "session-1"
        assert state.current_task_id is None
        assert state.important_files == []
        assert state.recent_notes == []
        assert state.failed_approaches == []

    def test_full_construction(self):
        state = WorkingMemoryState(
            session_id="session-1",
            current_task_id="task-001",
            important_files=["src/main.py"],
            recent_notes=["note 1"],
            failed_approaches=["approach 1"],
        )
        assert state.current_task_id == "task-001"
        assert state.important_files == ["src/main.py"]

    def test_serialization_round_trip(self):
        state = WorkingMemoryState(
            session_id="session-2",
            current_task_id="task-002",
            important_files=["a.py", "b.py"],
            recent_notes=["n1", "n2"],
            failed_approaches=["f1"],
        )
        data = state.model_dump()
        restored = WorkingMemoryState(**data)
        assert restored.session_id == state.session_id
        assert restored.important_files == state.important_files

    def test_json_schema_generation(self):
        schema = WorkingMemoryState.model_json_schema()
        assert schema["title"] == "WorkingMemoryState"
        assert "session_id" in schema["properties"]
        assert "important_files" in schema["properties"]

    def test_required_field_raises(self):
        with pytest.raises(ValidationError):
            WorkingMemoryState()

    def test_optional_field_defaults(self):
        state = WorkingMemoryState(session_id="session-3")
        assert state.current_task_id is None
        assert state.important_files == []
        assert state.recent_notes == []
        assert state.failed_approaches == []

    def test_to_json_round_trip(self):
        state = WorkingMemoryState(session_id="session-4")
        json_str = state.model_dump_json()
        parsed = WorkingMemoryState.model_validate_json(json_str)
        assert parsed.session_id == state.session_id


class TestSessionSummary:
    """Tests for the SessionSummary schema."""

    def test_basic_construction(self):
        summary = SessionSummary(
            session_id="session-1",
            summary="Completed task.",
        )
        assert summary.session_id == "session-1"
        assert summary.summary == "Completed task."
        assert summary.key_patterns == []
        assert summary.key_learnings == []
        assert summary.failed_approaches == []
        assert summary.produced_files == []

    def test_full_construction(self):
        summary = SessionSummary(
            session_id="session-1",
            summary="Completed task.",
            key_patterns=["pattern 1"],
            key_learnings=["learning 1"],
            failed_approaches=["failed 1"],
            produced_files=["file.py"],
        )
        assert summary.key_patterns == ["pattern 1"]
        assert summary.produced_files == ["file.py"]

    def test_serialization_round_trip(self):
        summary = SessionSummary(
            session_id="session-2",
            summary="Summary content",
            key_patterns=["p1", "p2"],
            key_learnings=["l1"],
        )
        data = summary.model_dump()
        restored = SessionSummary(**data)
        assert restored.session_id == summary.session_id
        assert restored.key_patterns == summary.key_patterns

    def test_json_schema_generation(self):
        schema = SessionSummary.model_json_schema()
        assert schema["title"] == "SessionSummary"
        assert "session_id" in schema["properties"]
        assert "summary" in schema["properties"]
        assert "key_learnings" in schema["properties"]

    def test_required_field_raises(self):
        with pytest.raises(ValidationError):
            SessionSummary()

    def test_optional_field_defaults(self):
        summary = SessionSummary(session_id="session-3", summary="s")
        assert summary.key_patterns == []
        assert summary.key_learnings == []
        assert summary.failed_approaches == []
        assert summary.produced_files == []

    def test_to_json_round_trip(self):
        summary = SessionSummary(session_id="session-4", summary="s")
        json_str = summary.model_dump_json()
        parsed = SessionSummary.model_validate_json(json_str)
        assert parsed.session_id == summary.session_id
