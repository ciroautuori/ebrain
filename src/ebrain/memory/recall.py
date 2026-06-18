"""Memory Recall — vector search for relevant memories at session start.

Uses Qdrant for vector similarity search (when available) with
fallback to PostgreSQL keyword search.
"""

from __future__ import annotations

import logging
import time

from ebrain.memory.config import MemoryConfig
from ebrain.memory.l1_extractor import get_memories
from ebrain.memory.l2l3 import get_persona
from ebrain.memory.l2l3 import get_scenes
from ebrain.memory.types import L1Memory
from ebrain.memory.types import RecallResult

_log = logging.getLogger("ebrain.memory.recall")


async def _vector_recall(
    query: str,
    session_id: str,
    config: MemoryConfig,
) -> list[L1Memory]:
    """Recall memories via Qdrant vector similarity."""
    try:
        from fastembed import TextEmbedding
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance
        from qdrant_client.models import VectorParams
    except ImportError:
        _log.debug("qdrant/fastembed not available, falling back to keyword recall")
        return []

    try:
        client = QdrantClient(host="127.0.0.1", port=6333)
        embedder = TextEmbedding()
    except Exception as exc:
        _log.debug("qdrant client init failed: %s", exc)
        return []

    collection = f"ebrain.memory_{session_id}"

    # Ensure collection exists
    try:
        client.get_collection(collection)
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=384,  # bge-small-en-v1.5
                distance=Distance.COSINE,
            ),
        )

    # Get all memories for this session
    all_memories = await get_memories(session_id, limit=500)
    if not all_memories:
        return []

    # Embed query and memories
    query_vec = list(embedder.embed([query]))[0]

    # Search
    try:
        results = client.search(
            collection_name=collection,
            query_vector=query_vec,
            limit=config.recall_max_results,
            score_threshold=config.recall_score_threshold,
        )
    except Exception:
        # Fallback: index memories first if collection is empty
        return _keyword_recall(query, all_memories, config)

    if not results:
        return []

    # Map back to memories
    hit_ids = {hit.id for hit in results}
    return [m for m in all_memories if m.id in hit_ids][: config.recall_max_results]


def _keyword_recall(
    query: str,
    memories: list[L1Memory],
    config: MemoryConfig,
) -> list[L1Memory]:
    """Simple keyword-based recall fallback."""
    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored: list[tuple[float, L1Memory]] = []
    for mem in memories:
        content_lower = mem.content.lower()
        # Score: exact phrase + keyword overlap
        score = 0.0
        if query_lower in content_lower:
            score += 0.5
        # Keyword overlap
        kw_overlap = sum(
            1 for w in query_words
            if w in content_lower or any(w in kw.lower() for kw in mem.keywords)
        )
        score += kw_overlap * 0.1

        if score >= config.recall_score_threshold:
            scored.append((score, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[: config.recall_max_results]]


async def recall(
    query: str,
    session_id: str,
    *,
    config: MemoryConfig,
) -> RecallResult:
    """Recall relevant memories, scenes, and persona for a query.

    Primary: Qdrant vector search.
    Fallback: PostgreSQL keyword search.

    Returns a RecallResult ready for context injection.
    """
    t0 = time.time()

    if not config.recall_enabled:
        return RecallResult(elapsed_ms=0, strategy="disabled")

    # Try vector recall first, fallback to keyword
    strategy = "vector"
    memories = await _vector_recall(query, session_id, config)
    if not memories:
        strategy = "keyword"
        all_memories = await get_memories(session_id, limit=500)
        memories = _keyword_recall(query, all_memories, config)

    # Get persona and relevant scenes
    persona = await get_persona(session_id)
    scenes = await get_scenes(session_id, limit=5)

    elapsed = (time.time() - t0) * 1000

    if memories:
        _log.debug(
            "recall [%s]: %d memories, persona=%s, scenes=%d (%.0fms)",
            strategy,
            len(memories),
            "yes" if persona else "no",
            len(scenes),
            elapsed,
        )

    return RecallResult(
        memories=memories,
        persona=persona,
        scenes=scenes,
        elapsed_ms=elapsed,
        strategy=strategy,
    )
