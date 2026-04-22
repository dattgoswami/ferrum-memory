"""WorkingMemoryState and SessionSummary — working memory contracts."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class WorkingMemoryState(BaseModel):
    """Active working memory for a session, persisted in Redis.

    Compressed on session close by the consolidation lifecycle.
    """

    session_id: str = Field(description="Unique session identifier.")
    current_task_id: str | None = Field(default=None, description="ID of the task currently being worked on.")
    important_files: List[str] = Field(default_factory=list, description="Files the agent is actively touching.")
    recent_notes: List[str] = Field(default_factory=list, description="Recent notes from the agent.")
    failed_approaches: List[str] = Field(default_factory=list, description="Approaches that failed for this session.")
    last_updated: float = Field(
        default_factory=lambda: __import__("time").time(),
        description="Unix timestamp of last update.",
    )


class SessionSummary(BaseModel):
    """Compressed summary of a session, produced on close.

    Stored as a MemoryItem with kind=PATTERN or kind=LEARNING.
    """

    session_id: str = Field(description="The session this summary covers.")
    summary: str = Field(description="Human-readable summary of the session.")
    key_patterns: List[str] = Field(default_factory=list, description="Patterns observed during the session.")
    key_learnings: List[str] = Field(default_factory=list, description="Learnings that should persist.")
    failed_approaches: List[str] = Field(default_factory=list, description="Approaches that failed.")
    produced_files: List[str] = Field(default_factory=list, description="Files produced or modified.")
