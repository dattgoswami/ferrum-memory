"""Tests for contracts/experience_tuple.py."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from contracts.experience_tuple import ExperienceTuple, TestResult, ToolCallRecord


class TestToolCallRecord:
    """Tests for the ToolCallRecord schema."""

    def test_basic_construction(self):
        record = ToolCallRecord(tool_name="write_file", arguments={"path": "x.py"})
        assert record.tool_name == "write_file"
        assert record.result is None
        assert record.duration_seconds is None

    def test_serialization_round_trip(self):
        record = ToolCallRecord(
            tool_name="read_file",
            arguments={"path": "src/main.py"},
            result="file contents",
            duration_seconds=0.3,
        )
        data = record.model_dump()
        restored = ToolCallRecord(**data)
        assert restored.tool_name == record.tool_name
        assert restored.result == record.result

    def test_json_schema_generation(self):
        schema = ToolCallRecord.model_json_schema()
        assert schema["title"] == "ToolCallRecord"
        assert "tool_name" in schema["properties"]
        assert "arguments" in schema["properties"]

    def test_required_field_raises(self):
        with pytest.raises(ValidationError):
            ToolCallRecord(arguments={"x": 1})

    def test_optional_field_defaults(self):
        record = ToolCallRecord(tool_name="tool")
        assert record.arguments == {}
        assert record.result is None
        assert record.duration_seconds is None


class TestExperienceTuple:
    """Tests for the ExperienceTuple schema."""

    def test_basic_construction(self):
        exp = ExperienceTuple(
            experience_id="exp-001",
            session_id="session-1",
            task_id="task-1",
            task_description="fix bug",
            test_result=TestResult.PASS,
        )
        assert exp.experience_id == "exp-001"
        assert exp.reward == 0.0
        assert exp.attempt_number == 1
        assert exp.td_error == 0.0
        assert exp.tool_call_sequence == []

    def test_serialization_round_trip(self):
        exp = ExperienceTuple(
            experience_id="exp-002",
            session_id="s1",
            task_id="t1",
            task_description="desc",
            test_result=TestResult.FAIL,
            reward=-0.5,
            attempt_number=3,
            td_error=0.2,
            tool_call_sequence=[
                ToolCallRecord(tool_name="write_file", arguments={"path": "x.py"}),
            ],
        )
        data = exp.model_dump()
        restored = ExperienceTuple(**data)
        assert restored.experience_id == exp.experience_id
        assert restored.test_result == exp.test_result
        assert restored.reward == exp.reward
        assert len(restored.tool_call_sequence) == 1

    def test_json_schema_generation(self):
        schema = ExperienceTuple.model_json_schema()
        assert schema["title"] == "ExperienceTuple"
        assert "experience_id" in schema["properties"]
        assert "reward" in schema["properties"]
        assert "test_result" in schema["properties"]

    def test_required_field_raises(self):
        with pytest.raises(ValidationError):
            ExperienceTuple(
                experience_id="exp-003",
                # missing session_id
                task_id="t1",
                task_description="d",
                test_result=TestResult.PASS,
            )

    def test_validation_constraints_reward_bounds(self):
        with pytest.raises(ValidationError):
            ExperienceTuple(
                experience_id="exp-004",
                session_id="s1",
                task_id="t1",
                task_description="d",
                test_result=TestResult.PASS,
                reward=1.5,
            )

        with pytest.raises(ValidationError):
            ExperienceTuple(
                experience_id="exp-005",
                session_id="s1",
                task_id="t1",
                task_description="d",
                test_result=TestResult.PASS,
                reward=-1.5,
            )

    def test_validation_constraints_td_error(self):
        # td_error has no bounds constraint, any float is valid
        exp = ExperienceTuple(
            experience_id="exp-006",
            session_id="s1",
            task_id="t1",
            task_description="d",
            test_result=TestResult.PASS,
            td_error=999.0,
        )
        assert exp.td_error == 999.0

    def test_optional_field_defaults(self):
        exp = ExperienceTuple(
            experience_id="exp-007",
            session_id="s1",
            task_id="t1",
            task_description="d",
            test_result=TestResult.TIMEOUT,
        )
        assert exp.reward == 0.0
        assert exp.attempt_number == 1
        assert exp.duration_seconds is None
        assert exp.tool_call_sequence == []

    def test_serialization_to_json(self):
        exp = ExperienceTuple(
            experience_id="exp-008",
            session_id="s1",
            task_id="t1",
            task_description="d",
            test_result=TestResult.ESCALATED,
            reward=1.0,
        )
        json_str = exp.model_dump_json()
        parsed = ExperienceTuple.model_validate_json(json_str)
        assert parsed.experience_id == exp.experience_id
        assert parsed.reward == 1.0

    def test_test_result_enum_values(self):
        assert TestResult.PASS.value == "pass"
        assert TestResult.FAIL.value == "fail"
        assert TestResult.TIMEOUT.value == "timeout"
        assert TestResult.ESCALATED.value == "escalated"

    def test_tool_call_sequence(self):
        exp = ExperienceTuple(
            experience_id="exp-009",
            session_id="s1",
            task_id="t1",
            task_description="d",
            test_result=TestResult.PASS,
            tool_call_sequence=[
                ToolCallRecord(tool_name="read_file", arguments={"path": "a.py"}),
                ToolCallRecord(tool_name="write_file", arguments={"path": "b.py"}),
            ],
        )
        assert len(exp.tool_call_sequence) == 2
        assert exp.tool_call_sequence[0].tool_name == "read_file"
        assert exp.tool_call_sequence[1].tool_name == "write_file"
