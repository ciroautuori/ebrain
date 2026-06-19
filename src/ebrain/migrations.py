"""EBrain versioned migrations — lightweight, no external dependency.

Each migration is an idempotent SQL block tagged with a version number.
`run_migrations()` replaces scattered `ensure_schema()` calls.
`schema_version` table tracks applied versions.
"""

from __future__ import annotations

import logging

from ebrain.db import execute
from ebrain.db import fetch

_log = logging.getLogger("ebrain.migrations")

_MIGRATIONS: list[tuple[int, str]] = [
    (1, """
        CREATE TABLE IF NOT EXISTS schema_version (
            version     INT PRIMARY KEY,
            applied_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS ebrain_entities (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            kind        TEXT NOT NULL DEFAULT 'concept',
            tags        JSONB DEFAULT '[]',
            metadata    JSONB DEFAULT '{}',
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS ebrain_edges (
            id          BIGSERIAL PRIMARY KEY,
            source_id   TEXT NOT NULL REFERENCES ebrain_entities(id) ON DELETE CASCADE,
            target_id   TEXT NOT NULL REFERENCES ebrain_entities(id) ON DELETE CASCADE,
            kind        TEXT NOT NULL DEFAULT 'relates_to',
            weight      REAL DEFAULT 1.0,
            metadata    JSONB DEFAULT '{}',
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_ebrain_entities_kind ON ebrain_entities(kind);
        CREATE INDEX IF NOT EXISTS idx_ebrain_edges_source ON ebrain_edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_ebrain_edges_target ON ebrain_edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_ebrain_edges_kind ON ebrain_edges(kind);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ebrain_edges_unique
            ON ebrain_edges (source_id, target_id, kind);

        CREATE TABLE IF NOT EXISTS ebrain_memory_l0_conversations (
            id              BIGSERIAL PRIMARY KEY,
            session_id      TEXT NOT NULL,
            turn_number     INT NOT NULL DEFAULT 0,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            tool_calls      JSONB DEFAULT '[]',
            tool_results    JSONB DEFAULT '[]',
            metadata        JSONB DEFAULT '{}',
            token_count     INT DEFAULT 0,
            recorded_at     TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_ebrain_memory_l0_session
            ON ebrain_memory_l0_conversations (session_id, turn_number);
        CREATE INDEX IF NOT EXISTS idx_ebrain_memory_l0_recorded
            ON ebrain_memory_l0_conversations (recorded_at);

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

        CREATE TABLE IF NOT EXISTS ebrain_memory_l2_scenes (
            id              TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL,
            title           TEXT NOT NULL,
            summary         TEXT NOT NULL,
            memory_ids      JSONB DEFAULT '[]',
            tags            JSONB DEFAULT '[]',
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_ebrain_memory_l2_session
            ON ebrain_memory_l2_scenes (session_id, created_at);

        CREATE TABLE IF NOT EXISTS ebrain_memory_l3_personas (
            session_id          TEXT PRIMARY KEY,
            name                TEXT DEFAULT '',
            role                TEXT DEFAULT '',
            traits              JSONB DEFAULT '[]',
            preferences         JSONB DEFAULT '[]',
            recurring_topics    JSONB DEFAULT '[]',
            tools_used          JSONB DEFAULT '[]',
            summary             TEXT DEFAULT '',
            total_memories      INT DEFAULT 0,
            total_conversations INT DEFAULT 0,
            updated_at          TIMESTAMPTZ DEFAULT NOW()
        );
    """),
]


async def run_migrations() -> int:
    """Apply all pending migrations. Returns count of newly applied migrations."""
    await execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version     INT PRIMARY KEY,
            applied_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    applied = {r["version"] for r in await fetch("SELECT version FROM schema_version")}
    count = 0

    for version, sql in _MIGRATIONS:
        if version in applied:
            continue
        await execute(sql)
        await execute("INSERT INTO schema_version (version) VALUES ($1) ON CONFLICT DO NOTHING", version)
        _log.info("migration v%d applied", version)
        count += 1

    return count


async def current_version() -> int:
    """Return highest applied migration version, or 0 if none."""
    try:
        rows = await fetch("SELECT MAX(version) as v FROM schema_version")
        return int(rows[0]["v"] or 0) if rows else 0
    except Exception:
        return 0
