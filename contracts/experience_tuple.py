"""ExperienceTuple — RL experience storage contract."""
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class TestResult(str, Enum):
    """Outcome of running tests after a tool-call sequence."""

    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    ESCALATED = "escalated"


class ToolCallRecord(BaseModel):
    """Record of a single tool call made during an experience."""

    tool_name: str = Field(description="Name of the tool that was called.")
    arguments: dict = Field(default_factory=dict, description="Arguments passed to the tool.")
    result: str | None = Field(default=None, description="Result of the tool call, if any.")
    duration_seconds: float | None = Field(default=None, description="How long the tool call took.")


class ExperienceTuple(BaseModel):
    """An RL experience tuple for prioritized experience replay.

    Captures a full task execution including tool calls, test outcomes,
    and a numeric reward for prioritizing replay.
    """

    experience_id: str = Field(description="Unique identifier for this experience.")
    session_id: str = Field(description="Session this experience belongs to.")
    task_id: str = Field(description="Task that was executed.")
    task_description: str = Field(description="Human-readable description of the task.")
    tool_call_sequence: List[ToolCallRecord] = Field(default_factory=list, description="Sequence of tool calls.")
    test_result: TestResult = Field(description="Outcome of running tests.")
    reward: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Reward signal in [-1, 1]. Higher is better.",
    )
    attempt_number: int = Field(default=1, description="Which attempt this was for the task.")
    duration_seconds: float | None = Field(default=None, description="Total duration of the task execution.")
    timestamp: float = Field(default_factory=lambda: __import__("time").time(), description="Unix timestamp.")
    td_error: float = Field(
        default=0.0,
        description="Temporal-difference error for prioritized sampling.",
    )
