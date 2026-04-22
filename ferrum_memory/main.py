"""FastAPI application — app factory, lifespan, /health, /stats."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from ferrum_memory.api import memory, experience, session
from ferrum_memory.config import Config
from ferrum_memory.storage.router import StorageRouter

logger = logging.getLogger(__name__)

config = Config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: initialize and teardown all backends."""
    app.state.storage = StorageRouter(config)
    try:
        await app.state.storage.initialize()
        logger.info("All backends initialized.")
    except Exception:
        logger.exception("Failed to initialize backends.")
        raise
    yield
    try:
        await app.state.storage.close()
    except Exception:
        logger.exception("Failed to close backends.")
    logger.info("All backends closed.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ferrum-memory",
        description="Agent working memory and experience replay buffer.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(memory.router)
    app.include_router(experience.router)
    app.include_router(session.router)

    @app.get("/health")
    async def health() -> dict:
        status = await app.state.storage.health_check()
        all_ok = all(status.values())
        return {"status": "healthy" if all_ok else "degraded", "backends": status}

    @app.get("/stats")
    async def stats() -> dict:
        qdrant_count = await app.state.storage.count_memory()
        exp_stats = await app.state.storage.get_experience_stats()
        return {
            "memory_items": qdrant_count,
            **exp_stats,
        }

    return app


app = create_app()
