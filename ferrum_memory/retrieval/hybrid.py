"""Hybrid search — BM25 + dense + RRF fusion pipeline."""
from __future__ import annotations

import math
from typing import Any


def rrf_fusion(results: list[dict[str, Any]], k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion (RRF) over ranked candidate lists.

    Fuses multiple retrieval results into a single ranked list
    using the RRF formula: score(d) = sum(1 / (k + rank(d))).
    """
    score_map: dict[str, float] = {}
    for result_list in results:
        for rank, item in enumerate(result_list, 1):
            item_id = item.get("id", str(item))
            score_map[item_id] = score_map.get(item_id, 0.0) + 1.0 / (k + rank)

    fused = []
    for item_id, score in score_map.items():
        matching = next((item for item_list in results for item in item_list if item.get("id", str(item)) == item_id))
        fused.append({**matching, "rrf_score": score})
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


def generate_dense(content: str, dim: int = 384) -> list[float]:
    """Generate a dense embedding vector from text content.

    Uses a simple hash-based placeholder. Replace with fastembed
    in production.
    """
    import hashlib
    import struct

    vector = []
    for i in range(dim):
        h = hashlib.sha256(f"{content}:{i}".encode()).hexdigest()
        # Map hex to [-1, 1] range for better normalization
        val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
        vector.append(val)
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector


def generate_sparse(content: str) -> tuple[list[int], list[float]]:
    """Generate sparse BM25-style terms from text content.

    Returns (indices, values) where indices are hashed term positions
    and values are simple term frequencies.
    """
    import hashlib

    terms = content.lower().split()
    term_counts: dict[str, int] = {}
    for term in terms:
        term_counts[term] = term_counts.get(term, 0) + 1

    sparse_indices = []
    sparse_values = []
    for term, count in term_counts.items():
        h = int(hashlib.md5(term.encode()).hexdigest(), 16)
        sparse_indices.append(h % 10000)
        sparse_values.append(float(count))
    return sparse_indices, sparse_values
