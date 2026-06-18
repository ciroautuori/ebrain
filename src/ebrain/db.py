"""EBrain Database — asyncpg connection pool (PostgreSQL 17+).

Standalone, zero EROS dependencies. Singleton pool with configurable limits.
Idempotent schema migrations versioned via `schema_version` table.

Usage:
    from ebrain.db import pool, fetch, fetchone, execute

    rows = await fetch("SELECT * FROM entities WHERE kind = $1", "person")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import asyncpg

_log = logging.getLogger("ebrain.db")

DATABASE_URL = os.environ.get(
    "EBRAIN_DATABASE_URL",
    "postgresql://eros:eros_dev_2026@127.0.0.1:5433/eros",
)

# ── Pool singleton ────────────────────────────────────────────────────────

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Get or create the asyncpg connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=4,
            max_size=20,
            command_timeout=30,
            statement_cache_size=100,
        )
        _log.info("ebrain db pool ready (%s)", DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL)
    return _pool


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def execute(query: str, *args: Any) -> str:
    """Execute a statement. Returns status string."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def fetch(query: str, *args: Any) -> list[asyncpg.Record]:
    """Fetch all rows."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def fetchone(query: str, *args: Any) -> asyncpg.Record | None:
    """Fetch one row or None."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetchval(query: str, *args: Any) -> Any:
    """Fetch a single value."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(query, *args)


# ── Domain types ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Entity:
    """A typed entity in the knowledge graph."""
    id: str
    name: str
    kind: str = "concept"
    tags: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass
class Edge:
    """A typed edge connecting two entities."""
    source_id: str
    target_id: str
    kind: str = "relates_to"
    weight: float = 1.0
    created_at: str = ""


# ── Migrations ─────────────────────────────────────────────────────────────

async def ensure_schema() -> None:
    """Create core tables if they don't exist (idempotent)."""
    await execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INT PRIMARY KEY,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS entities (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            kind        TEXT NOT NULL DEFAULT 'concept',
            tags        JSONB DEFAULT '[]',
            metadata    JSONB DEFAULT '{}',
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS edges (
            id          BIGSERIAL PRIMARY KEY,
            source_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            target_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            kind        TEXT NOT NULL DEFAULT 'relates_to',
            weight      REAL DEFAULT 1.0,
            metadata    JSONB DEFAULT '{}',
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind);
        CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges(kind);
    """)
