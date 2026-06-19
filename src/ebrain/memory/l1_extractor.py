"""L1 Memory Extractor — LLM-powered memory extraction with vector dedup.

Inspired by TencentDB Agent Memory L1 layer:
1. Takes recent conversation turns (L0)
2. Calls LLM to extract structured memories (facts, preferences, decisions)
3. Deduplicates against existing memories via Qdrant vector similarity
4. Stores new memories in PG + Qdrant
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from ebrain.db import execute
from ebrain.db import fetch
from ebrain.db import fetchone
from ebrain.llm import ask_json
from ebrain.llm import get_default_model
from ebrain.memory.config import MemoryConfig
from ebrain.memory.qdrant import is_near_duplicate
from ebrain.memory.qdrant import upsert_memory
from ebrain.memory.types import L1Memory

_log = logging.getLogger("ebrain.memory.l1")

L1_SCHEMA = """
CREATE TABLE IF NOT EXISTS ebrain_memory_l1_extractions (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    content         TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'fact',
    keywords        JSONB DEFAULT '[]',
    source_l0_ids   JSONB DEFAULT '[]',
    source_turn     INT DEFAULT 0,
    confidence      REAL DEFAULT 0.8,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ebrain_memory_l1_checkpoints (
    session_id      TEXT PRIMARY KEY,
    last_l0_id      BIGINT NOT NULL DEFAULT 0,
    total_extracted INT NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ebrain_memory_l1_session
    ON ebrain_memory_l1_extractions (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_ebrain_memory_l1_kind
    ON ebrain_memory_l1_extractions (kind);
"""

EXTRACTION_PROMPT = """You are a memory extraction engine. From the conversation
below, extract up to {max_memories} structured observations as JSON.

Each observation must be a fact, preference, decision, or pattern about the user or project.
IMPORTANT: Do NOT extract generic AI assistant responses or tool output logs.
Only extract information that would be useful to remember across sessions.

Output format:
{{
  "memories": [
    {{
      "content": "The user prefers dark mode for all dashboards",
      "kind": "preference",
      "keywords": ["dark mode", "dashboard", "ui"],
      "confidence": 0.9
    }}
  ]
}}

Kinds: fact | preference | decision | pattern | question

Conversation:
{conversation}
"""



async def ensure_schema() -> None:
    """Create L1 tables (idempotent)."""
    await execute(L1_SCHEMA)


def _build_extraction_prompt(turns: list[dict], max_memories: int) -> str:
    """Build the extraction prompt from conversation turns."""
    conversation_parts: list[str] = []
    for t in turns:
        role = t.get("role", "unknown")
        content = t.get("content", "")[:2000]  # truncate long contents
        conversation_parts.append(f"[{role}]: {content}")

    return EXTRACTION_PROMPT.format(
        max_memories=max_memories,
        conversation="\n".join(conversation_parts),
    )




async def extract_memories(
    turns: list[dict],
    session_id: str,
    *,
    config: MemoryConfig,
    model: str | None = None,
) -> list[L1Memory]:
    """Extract L1 memories from conversation turns.

    Dedup via Qdrant cosine similarity (threshold: config.l1_dedup_threshold).
    Stores each new memory in PG + indexes in Qdrant for future recall.
    """
    if not config.l1_enabled or not turns:
        return []

    model = model or config.l1_model or get_default_model()
    prompt = _build_extraction_prompt(turns, config.l1_max_memories_per_run)

    try:
        result = await ask_json(prompt, model=model)
    except Exception as exc:
        _log.warning("l1 extraction LLM call failed: %s", exc)
        return []

    if not isinstance(result, dict) or "memories" not in result:
        _log.warning("l1 extraction returned unexpected format: %s", type(result))
        return []

    new_memories: list[L1Memory] = []
    for raw in result.get("memories", []):
        if not isinstance(raw, dict) or not raw.get("content"):
            continue
        content = str(raw["content"]).strip()
        if len(content) < 10:
            continue

        if await is_near_duplicate(content, session_id, threshold=config.l1_dedup_threshold):
            _log.debug("l1 dedup: skipped near-duplicate '%s'", content[:80])
            continue

        mem = L1Memory(
            id=f"l1_{session_id}_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            content=content,
            kind=str(raw.get("kind", "fact")),
            keywords=list(raw.get("keywords", []))[:8],
            source_turn=turns[-1].get("turn", 0) if turns else 0,
            confidence=float(raw.get("confidence", 0.8)),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        await execute(
            """INSERT INTO ebrain_memory_l1_extractions (id, session_id, content, kind, keywords, source_turn, confidence)
               VALUES ($1, $2, $3, $4, $5, $6, $7)
               ON CONFLICT (id) DO NOTHING""",
            mem.id,
            mem.session_id,
            mem.content,
            mem.kind,
            json.dumps(mem.keywords),
            mem.source_turn,
            mem.confidence,
        )
        await upsert_memory(mem.id, session_id, content)
        new_memories.append(mem)

    # Update checkpoint
    if new_memories and turns:
        last_l0_id = max(t.get("id", 0) for t in turns)
        await execute(
            """INSERT INTO ebrain_memory_l1_checkpoints (session_id, last_l0_id, total_extracted)
               VALUES ($1, $2, $3)
               ON CONFLICT (session_id) DO UPDATE
               SET last_l0_id = EXCLUDED.last_l0_id,
                   total_extracted = ebrain_memory_l1_checkpoints.total_extracted + EXCLUDED.total_extracted,
                   updated_at = NOW()""",
            session_id,
            last_l0_id,
            len(new_memories),
        )

    if new_memories:
        _log.info("l1 extraction: %d new memories from session %s", len(new_memories), session_id)

    return new_memories


async def get_memories(
    session_id: str,
    *,
    limit: int = 50,
    kind: str | None = None,
) -> list[L1Memory]:
    """Retrieve stored L1 memories for a session."""
    query = """SELECT id, session_id, content, kind, keywords, source_turn, confidence, created_at
               FROM ebrain_memory_l1_extractions
               WHERE session_id = $1"""
    params: list[Any] = [session_id]

    if kind:
        query += " AND kind = $2"
        params.append(kind)

    query += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1)
    params.append(limit)

    rows = await fetch(query, *params)
    return [
        L1Memory(
            id=r["id"],
            session_id=r["session_id"],
            content=r["content"],
            kind=r["kind"],
            keywords=json.loads(r["keywords"]) if isinstance(r["keywords"], str) else r["keywords"],
            source_turn=r["source_turn"],
            confidence=float(r["confidence"]) if r["confidence"] else 0.8,
            created_at=str(r["created_at"]),
        )
        for r in rows
    ]


async def count_memories(session_id: str) -> int:
    """Count L1 memories for a session."""
    row = await fetchone(
        "SELECT COUNT(*) as cnt FROM ebrain_memory_l1_extractions WHERE session_id = $1",
        session_id,
    )
    return row["cnt"] if row else 0
