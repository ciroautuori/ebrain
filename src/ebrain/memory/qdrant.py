"""Async Qdrant integration for ebrain — embeddings, upsert, vector search.

Single collection `ebrain_memories` with session_id payload filter.
AsyncQdrantClient + fastembed via asyncio.to_thread (non-blocking).
Graceful degradation: returns empty/False when Qdrant unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import struct

_log = logging.getLogger("ebrain.memory.qdrant")

QDRANT_HOST = os.environ.get("EBRAIN_QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("EBRAIN_QDRANT_PORT", "6333"))
COLLECTION = "ebrain_memories"
VECTOR_SIZE = 384  # bge-small-en-v1.5

_client = None
_embedder = None
_collection_ready = False


def _stable_id(mem_id: str) -> int:
    """Deterministic uint64 from memory ID string (MD5 first 8 bytes)."""
    raw = hashlib.md5(mem_id.encode()).digest()
    return struct.unpack(">Q", raw[:8])[0]


async def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        from qdrant_client import AsyncQdrantClient
        _client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
    except ImportError:
        _log.debug("qdrant_client not installed")
        return None
    return _client


async def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        from fastembed import TextEmbedding
        _embedder = await asyncio.to_thread(TextEmbedding)
    except ImportError:
        _log.debug("fastembed not installed")
        return None
    return _embedder


async def _ensure_collection() -> bool:
    global _collection_ready
    if _collection_ready:
        return True
    client = await _get_client()
    if client is None:
        return False
    try:
        from qdrant_client.models import Distance
        from qdrant_client.models import VectorParams
        existing = await client.get_collections()
        names = {c.name for c in existing.collections}
        if COLLECTION not in names:
            await client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            await client.create_payload_index(
                collection_name=COLLECTION,
                field_name="session_id",
                field_schema="keyword",
            )
        _collection_ready = True
        return True
    except Exception as exc:
        _log.debug("qdrant collection init failed: %s", exc)
        return False


async def embed(text: str) -> list[float] | None:
    """Embed text via fastembed (non-blocking via asyncio.to_thread)."""
    embedder = await _get_embedder()
    if embedder is None:
        return None
    try:
        vecs = await asyncio.to_thread(lambda: list(embedder.embed([text])))
        return [float(v) for v in vecs[0]]
    except Exception as exc:
        _log.debug("embed failed: %s", exc)
        return None


async def upsert_memory(mem_id: str, session_id: str, content: str) -> bool:
    """Embed and index a memory. Returns True on success."""
    if not await _ensure_collection():
        return False
    client = await _get_client()
    vec = await embed(content)
    if vec is None or client is None:
        return False
    try:
        from qdrant_client.models import PointStruct
        await client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(
                id=_stable_id(mem_id),
                vector=vec,
                payload={"mem_id": mem_id, "session_id": session_id},
            )],
        )
        return True
    except Exception as exc:
        _log.debug("qdrant upsert failed: %s", exc)
        return False


async def search_memories(
    query: str,
    session_id: str,
    limit: int = 5,
    score_threshold: float = 0.3,
) -> list[str]:
    """Vector search over memories for a session. Returns list of mem_ids."""
    if not await _ensure_collection():
        return []
    client = await _get_client()
    vec = await embed(query)
    if vec is None or client is None:
        return []
    try:
        from qdrant_client.models import FieldCondition
        from qdrant_client.models import Filter
        from qdrant_client.models import MatchValue
        response = await client.query_points(
            collection_name=COLLECTION,
            query=vec,
            query_filter=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            ),
            limit=limit,
            score_threshold=score_threshold,
        )
        return [hit.payload["mem_id"] for hit in response.points if "mem_id" in hit.payload]
    except Exception as exc:
        _log.debug("qdrant search failed: %s", exc)
        return []


async def is_near_duplicate(content: str, session_id: str, threshold: float = 0.85) -> bool:
    """True if content has cosine similarity >= threshold to any existing memory in session."""
    if not await _ensure_collection():
        return False
    client = await _get_client()
    vec = await embed(content)
    if vec is None or client is None:
        return False
    try:
        from qdrant_client.models import FieldCondition
        from qdrant_client.models import Filter
        from qdrant_client.models import MatchValue
        response = await client.query_points(
            collection_name=COLLECTION,
            query=vec,
            query_filter=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            ),
            limit=1,
            score_threshold=threshold,
        )
        return len(response.points) > 0
    except Exception as exc:
        _log.debug("qdrant dedup check failed: %s", exc)
        return False


def reset() -> None:
    """Reset all singletons (for testing)."""
    global _client, _embedder, _collection_ready
    _client = None
    _embedder = None
    _collection_ready = False
