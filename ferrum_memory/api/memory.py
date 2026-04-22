"""Memory API — POST/DELETE /memory/*, POST /memory/search, /memory/compress."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Response

from ferrum_memory.storage.router import StorageRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory")


@router.post("/store", response_model=None)
async def store_memory(
    point_id: str,
    dense_vector: list[float],
    payload: dict | None = None,
    sparse_vector: list[float] | None = None,
    storage: StorageRouter = Depends(),
) -> Response:
    """Store a memory item."""
    await storage.store_memory(point_id, dense_vector, payload, sparse_vector)
    return Response(status_code=200, content='{"status": "stored"}')


@router.delete("/{point_id}", response_model=None)
async def delete_memory(
    point_id: str,
    storage: StorageRouter = Depends(),
) -> Response:
    """Delete a memory item."""
    await storage._qdrant.delete(point_id)
    return Response(status_code=200, content='{"status": "deleted"}')


@router.post("/search", response_model=None)
async def search_memory(
    body: dict,
    limit: int = 10,
    rerank: bool = False,
    storage: StorageRouter = Depends(),
) -> Response:
    query_dense = body.get("query_dense", [])
    """Search memory items with optional reranking."""
    results = await storage.search_memory(query_dense, limit=limit)
    if rerank:
        from ferrum_memory.retrieval.reranker import Reranker
        reranker = Reranker()
        results = reranker.rerank("", results, top_k=limit)
    import json
    return Response(status_code=200, content=json.dumps({"results": results}))


@router.post("/compress", response_model=None)
async def compress_session(
    session_id: str,
    storage: StorageRouter = Depends(),
) -> Response:
    """Compress a session's working memory into a summary."""
    from ferrum_memory.lifecycle.consolidation import consolidate
    summary = await consolidate(session_id, storage)
    import json
    return Response(status_code=200, content=json.dumps({"session_id": session_id, "summary": summary}))
