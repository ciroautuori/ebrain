"""L0 Conversation Recorder — persist raw conversation turns to PostgreSQL.

Inspired by TencentDB Agent Memory L0 layer: auto-captures conversation JSONL
into a structured PG table for downstream extraction (L1) and audit.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ebrain.db import execute
from ebrain.db import fetch

_log = logging.getLogger("ebrain.memory.l0")

L0_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_l0_conversations (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    turn_number     INT NOT NULL DEFAULT 0,
    role            TEXT NOT NULL,  -- user | assistant | system
    content         TEXT NOT NULL,
    tool_calls      JSONB DEFAULT '[]',
    tool_results    JSONB DEFAULT '[]',
    metadata        JSONB DEFAULT '{}',
    token_count     INT DEFAULT 0,
    recorded_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_l0_session
    ON memory_l0_conversations (session_id, turn_number);

CREATE INDEX IF NOT EXISTS idx_memory_l0_recorded
    ON memory_l0_conversations (recorded_at);
"""


async def ensure_schema() -> None:
    """Create L0 tables if they don't exist (idempotent)."""
    try:
        await execute(L0_SCHEMA)
    except Exception as exc:
        _log.warning("l0 schema init failed (may already exist): %s", exc)


async def record_turn(
    session_id: str,
    role: str,
    content: str,
    *,
    turn_number: int = 0,
    tool_calls: list[dict] | None = None,
    tool_results: list[dict] | None = None,
    metadata: dict | None = None,
    token_count: int = 0,
) -> int:
    """Record a single conversation turn to L0.

    Returns the row id.
    """
    row = await fetch(
        """INSERT INTO memory_l0_conversations
               (session_id, turn_number, role, content, tool_calls, tool_results, metadata, token_count)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
           RETURNING id""",
        session_id,
        turn_number,
        role,
        content,
        json.dumps(tool_calls or []),
        json.dumps(tool_results or []),
        json.dumps(metadata or {}),
        token_count,
    )
    return row[0]["id"] if row else 0


async def get_recent_turns(
    session_id: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Get recent conversation turns for a session (for L1 extraction)."""
    rows = await fetch(
        """SELECT id, turn_number, role, content, tool_calls, metadata, recorded_at
           FROM memory_l0_conversations
           WHERE session_id = $1
           ORDER BY turn_number DESC
           LIMIT $2""",
        session_id,
        limit,
    )
    return [
        {
            "id": r["id"],
            "turn": r["turn_number"],
            "role": r["role"],
            "content": r["content"],
            "tool_calls": json.loads(r["tool_calls"]) if isinstance(r["tool_calls"], str) else r["tool_calls"],
            "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
            "recorded_at": str(r["recorded_at"]),
        }
        for r in rows
    ]


async def count_turns_since_last_extraction(session_id: str) -> int:
    """Count turns since last L1 extraction checkpoint for this session."""
    row = await fetch(
        """SELECT COUNT(*) as cnt
           FROM memory_l0_conversations
           WHERE session_id = $1
             AND id > COALESCE(
                 (SELECT MAX(last_l0_id)
                  FROM memory_l1_extractions
                  WHERE session_id = $1),
                 0
             )""",
        session_id,
    )
    return row[0]["cnt"] if row else 0


async def cleanup_old_conversations(retention_days: int = 90) -> int:
    """Remove conversations older than retention_days. Returns count deleted."""
    if retention_days <= 0:
        return 0
    cutoff = time.time() - retention_days * 86400
    rows = await execute(
        "DELETE FROM memory_l0_conversations WHERE EXTRACT(EPOCH FROM recorded_at) < $1",
        cutoff,
    )
    deleted = int(rows.replace("DELETE ", "")) if rows and rows.startswith("DELETE") else 0
    if deleted:
        _log.info("l0 cleanup: removed %d old conversations", deleted)
    return deleted
