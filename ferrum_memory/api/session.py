"""Session API — POST/GET/PUT /session/*."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Response

from ferrum_memory.storage.router import StorageRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/session")


@router.post("/start", response_model=None)
async def start_session(
    session_id: str,
    storage: StorageRouter = Depends(),
) -> Response:
    """Start a new session."""
    await storage.set_working_memory(session_id, {"session_id": session_id, "notes": []})
    return Response(status_code=200, content=json.dumps({"status": "started", "session_id": session_id}))


@router.get("/{session_id}", response_model=None)
async def get_session(
    session_id: str,
    storage: StorageRouter = Depends(),
) -> Response:
    """Get current working memory for a session."""
    data = await storage.get_working_memory(session_id)
    return Response(status_code=200, content=json.dumps({"session_id": session_id, "data": data}))


@router.put("/{session_id}", response_model=None)
async def update_session(
    session_id: str,
    data: dict,
    storage: StorageRouter = Depends(),
) -> Response:
    """Update working memory for a session."""
    data["session_id"] = session_id
    await storage.set_working_memory(session_id, data)
    return Response(status_code=200, content=json.dumps({"status": "updated", "session_id": session_id}))


@router.post("/close", response_model=None)
async def close_session(
    session_id: str,
    storage: StorageRouter = Depends(),
) -> Response:
    """Close a session, triggering consolidation."""
    from ferrum_memory.lifecycle.consolidation import consolidate
    summary = await consolidate(session_id, storage)
    await storage.delete_working_memory(session_id)
    return Response(status_code=200, content=json.dumps({"status": "closed", "session_id": session_id, "summary": summary}))
