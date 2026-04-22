"""Experience API — POST /experience, GET /replay, /experience/stats, td_error update."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Response

from ferrum_memory.storage.router import StorageRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experience")


@router.post("/", response_model=None)
async def store_experience(
    experience: dict,
    storage: StorageRouter = Depends(),
) -> Response:
    """Store an experience tuple."""
    await storage.store_experience(experience)
    return Response(status_code=200, content=json.dumps({"status": "stored", "experience_id": experience.get("experience_id")}))


@router.get("/stats", response_model=None)
async def get_experience_stats(
    storage: StorageRouter = Depends(),
) -> Response:
    """Get experience statistics."""
    stats = await storage.get_experience_stats()
    return Response(status_code=200, content=json.dumps(stats))


@router.put("/td-error", response_model=None)
async def update_td_error(
    experience_id: str,
    td_error: float,
    storage: StorageRouter = Depends(),
) -> Response:
    """Update the TD error for an experience."""
    await storage.update_td_error(experience_id, td_error)
    return Response(status_code=200, content=json.dumps({"status": "updated", "experience_id": experience_id, "td_error": td_error}))


@router.get("/replay", response_model=None)
async def get_replay(
    strategy: str = "prioritized",
    k: int = 10,
    session_id: str | None = None,
    storage: StorageRouter = Depends(),
) -> Response:
    """Get replay samples."""
    experiences = await storage.query_experiences(session_id=session_id)
    from ferrum_memory.retrieval import ReplayBuffer
    buffer = ReplayBuffer()
    samples = buffer.sample(experiences, strategy=strategy, k=k)
    return Response(status_code=200, content=json.dumps({"samples": samples, "strategy": strategy}))
