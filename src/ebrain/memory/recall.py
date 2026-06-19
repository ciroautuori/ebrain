"""Memory Recall — vector search for relevant memories at session start.

Primary: Qdrant vector search via ebrain.memory.qdrant (async, non-blocking).
Fallback: PostgreSQL keyword search when Qdrant unavailable.
"""

from __future__ import annotations

import logging
import time

from ebrain.memory.config import MemoryConfig
from ebrain.memory.l1_extractor import get_memories
from ebrain.memory.l2l3 import get_persona
from ebrain.memory.l2l3 import get_scenes
from ebrain.memory.qdrant import search_memories as qdrant_search
from ebrain.memory.types import L1Memory
from ebrain.memory.types import RecallResult

_log = logging.getLogger("ebrain.memory.recall")


def _keyword_recall(
    query: str,
    memories: list[L1Memory],
    config: MemoryConfig,
) -> list[L1Memory]:
    """Keyword-based recall fallback when Qdrant unavailable."""
    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored: list[tuple[float, L1Memory]] = []
    for mem in memories:
        content_lower = mem.content.lower()
        score = 0.0
        if query_lower in content_lower:
            score += 0.5
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

    Primary: Qdrant vector search (pre-indexed by L1 extractor).
    Fallback: PostgreSQL keyword search.
    """
    t0 = time.time()

    if not config.recall_enabled:
        return RecallResult(elapsed_ms=0, strategy="disabled")

    strategy = "vector"
    mem_ids = await qdrant_search(
        query,
        session_id,
        limit=config.recall_max_results,
        score_threshold=config.recall_score_threshold,
    )

    memories: list[L1Memory] = []
    if mem_ids:
        all_memories = await get_memories(session_id, limit=500)
        id_set = set(mem_ids)
        memories = [m for m in all_memories if m.id in id_set][: config.recall_max_results]
    else:
        strategy = "keyword"
        all_memories = await get_memories(session_id, limit=500)
        memories = _keyword_recall(query, all_memories, config)

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
