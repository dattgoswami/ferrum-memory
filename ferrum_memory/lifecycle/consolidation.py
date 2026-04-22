"""Session consolidation — compress working memory on session close."""
from __future__ import annotations

import logging
from typing import Any

from ferrum_memory.storage.router import StorageRouter
from ferrum_memory.config import Config

logger = logging.getLogger(__name__)


async def consolidate(session_id: str, storage: StorageRouter) -> dict[str, Any]:
    """Compress a session's working memory into a summary.

    Called on session close. Produces a SessionSummary-style dict
    that can be stored as a MemoryItem.
    """
    wm = await storage.get_working_memory(session_id)
    if not wm:
        return {
            "session_id": session_id,
            "summary": "No working memory found for session.",
            "key_patterns": [],
            "key_learnings": [],
            "produced_files": [],
        }

    files = wm.get("important_files", [])
    notes = wm.get("recent_notes", [])
    failed = wm.get("failed_approaches", [])

    summary_parts = [f"Session {session_id} involved {len(files)} files and {len(notes)} notes."]
    if failed:
        summary_parts.append(f"Failed approaches: {', '.join(failed)}.")

    summary = " ".join(summary_parts)

    return {
        "session_id": session_id,
        "summary": summary,
        "key_patterns": list(set(notes))[:10],
        "key_learnings": [f"Note: {n}" for n in notes[:5]],
        "produced_files": files,
    }
