"""Qdrant storage — hybrid BM25+dense ingest + RRF search."""
from __future__ import annotations

import logging
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    VectorParams,
)

from ferrum_memory.config import Config

logger = logging.getLogger(__name__)


class QdrantStore:
    """Async wrapper around Qdrant for hybrid dense+sparse search."""

    def __init__(self, config=None):
        self._config = config or Config()
        self._client: AsyncQdrantClient | None = None

    async def initialize(self) -> None:
        """Connect to Qdrant and ensure collection exists."""
        self._client = AsyncQdrantClient(url=self._config.QDRANT_URL)

        if not await self._collection_exists():
            await self._create_collection()

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def upsert(
        self,
        point_id: str,
        dense_vector: list[float],
        sparse_vector: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Ingest a memory item with dense + sparse vectors."""
        if not self._client:
            raise RuntimeError("QdrantStore not initialized. Call initialize() first.")

        sparse = None
        if sparse_vector is not None:
            sparse = SparseVector(indices=sparse_vector, values=self._extract_values(sparse_vector))

        point = PointStruct(
            id=point_id,
            vector={
                "dense": dense_vector,
                "sparse": sparse,
            },
            payload=payload or {},
        )
        await self._client.upsert(self._config.QDRANT_COLLECTION, points=[point])

    async def search(
        self,
        query_dense: list[float],
        query_sparse_indices: list[int] | None = None,
        query_sparse_values: list[float] | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search with RRF fusion of BM25 + dense prefetch."""
        if not self._client:
            raise RuntimeError("QdrantStore not initialized. Call initialize() first.")

        dense_params = VectorParams(distance=Distance.COSINE, size=self._config.FASTEMBED_DENSE_DIM)
        sparse_params = None

        prefetch: list[Any] = []

        if query_sparse_indices:
            sparse_vec = SparseVector(
                indices=query_sparse_indices,
                values=query_sparse_values or [1.0] * len(query_sparse_indices),
            )
            prefetch.append({
                "vector": {"dense": query_dense, "sparse": sparse_vec},
                "re_score": True,
            })

        if not prefetch:
            prefetch.append({"vector": {"dense": query_dense}, "re_score": True})

        result = await self._client.query_points(
            self._config.QDRANT_COLLECTION,
            query= {"dense": query_dense},
            prefetch=prefetch if len(prefetch) > 1 else [],
            query_filter=self._build_filter(filters),
            limit=limit,
        )

        return [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            }
            for point in result.points
        ]

    async def delete(self, point_id: str) -> bool:
        if not self._client:
            raise RuntimeError("QdrantStore not initialized.")
        await self._client.delete(self._config.QDRANT_COLLECTION, points_selector=[point_id])
        return True

    async def count(self) -> int:
        if not self._client:
            raise RuntimeError("QdrantStore not initialized.")
        return (await self._client.get_collection(self._config.QDRANT_COLLECTION)).points

    async def _collection_exists(self) -> bool:
        if not self._client:
            return False
        collections = await self._client.get_collections()
        return any(c.name == self._config.QDRANT_COLLECTION for c in collections.collections)

    async def _create_collection(self) -> None:
        if not self._client:
            return
        dense_cfg = VectorParams(
            distance=Distance.COSINE,
            size=self._config.FASTEMBED_DENSE_DIM,
        )
        await self._client.create_collection(
            collection_name=self._config.QDRANT_COLLECTION,
            vectors_config={"dense": dense_cfg},
        )
        await self._client.create_payload_index(
            collection_name=self._config.QDRANT_COLLECTION,
            field_name="kind",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        await self._client.create_payload_index(
            collection_name=self._config.QDRANT_COLLECTION,
            field_name="session_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def _build_filter(self, filters: dict[str, Any] | None) -> Filter | None:
        if not filters:
            return None
        conditions = [FieldCondition(key=k, match=MatchValue(v), ) for k, v in filters.items()]
        return Filter(must=conditions)

    @staticmethod
    def _extract_values(sparse: list[float]) -> list[float]:
        return sparse
