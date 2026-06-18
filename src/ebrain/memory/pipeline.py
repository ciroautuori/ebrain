"""Memory Pipeline — 4-layer orchestration (L0→L1→L2→L3).

Inspired by TencentDB Agent Memory's TdaiCore:
- Auto-records conversations (L0)
- Triggers L1 extraction after every N conversations
- Triggers L2 profile building after every N new L1 memories
- Triggers L3 persona generation after every N L2 scenes
- Provides auto-recall at session start via recall()

Usage:
    pipeline = MemoryPipeline()

    # At session start — recall relevant context
    ctx = await pipeline.recall("user query about X", session_id)

    # After each turn — record
    await pipeline.record(session_id, "user", "Hello")

    # After session — trigger extraction chain
    await pipeline.maybe_extract(session_id)
"""

from __future__ import annotations

import logging
from typing import Any

from ebrain.memory.config import DEFAULT_CONFIG
from ebrain.memory.config import MemoryConfig
from ebrain.memory.l0_recorder import count_turns_since_last_extraction
from ebrain.memory.l0_recorder import ensure_schema as ensure_l0
from ebrain.memory.l0_recorder import record_turn
from ebrain.memory.l1_extractor import count_memories
from ebrain.memory.l1_extractor import ensure_schema as ensure_l1
from ebrain.memory.l1_extractor import extract_memories
from ebrain.memory.l1_extractor import get_memories
from ebrain.memory.l2l3 import build_scenes
from ebrain.memory.l2l3 import ensure_schema as ensure_l2l3
from ebrain.memory.l2l3 import generate_persona
from ebrain.memory.l2l3 import get_persona
from ebrain.memory.l2l3 import get_scenes
from ebrain.memory.recall import recall as _recall
from ebrain.memory.types import L1Memory
from ebrain.memory.types import Persona
from ebrain.memory.types import RecallResult
from ebrain.memory.types import Scene

_log = logging.getLogger("ebrain.memory")


class MemoryPipeline:
    """Orchestrates the 4-layer memory pipeline.

    Thread-safe (uses asyncpg connection pool under the hood).
    All operations are async.
    """

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self._schema_ensured = False

    async def _init_schema(self) -> None:
        """Ensure all tables exist (lazy, once)."""
        if self._schema_ensured:
            return
        await ensure_l0()
        await ensure_l1()
        await ensure_l2l3()
        self._schema_ensured = True

    # ── L0: Recording ──────────────────────────────────────────

    async def record(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        turn_number: int = 0,
        metadata: dict | None = None,
        token_count: int = 0,
    ) -> int:
        """Record a conversation turn (L0)."""
        await self._init_schema()
        return await record_turn(
            session_id=session_id,
            role=role,
            content=content,
            turn_number=turn_number,
            metadata=metadata,
            token_count=token_count,
        )

    # ── L1: Extraction ─────────────────────────────────────────

    async def maybe_extract(self, session_id: str) -> list[L1Memory]:
        """Trigger L1 extraction if enough turns have accumulated.

        Returns newly extracted memories (empty if threshold not met).
        """
        await self._init_schema()
        if not self.config.l1_enabled:
            return []

        count = await count_turns_since_last_extraction(session_id)
        threshold = self.config.l1_every_n_conversations

        if count < threshold:
            _log.debug("l1 skip: %d turns (< threshold %d) for %s", count, threshold, session_id)
            return []

        from ebrain.memory.l0_recorder import get_recent_turns

        turns = await get_recent_turns(session_id, limit=max(10, count))
        return await extract_memories(turns, session_id, config=self.config)

    # ── L2/L3: Profile + Persona ───────────────────────────────

    async def maybe_build_profile(self, session_id: str) -> list[Scene]:
        """Trigger L2 scene building if enough L1 memories exist."""
        await self._init_schema()
        if not self.config.l2_enabled:
            return []

        total = await count_memories(session_id)
        if total < self.config.l2_trigger_every_n_memories:
            _log.debug("l2 skip: %d memories (< threshold %d)", total, self.config.l2_trigger_every_n_memories)
            return []

        return await build_scenes(session_id, config=self.config)

    async def maybe_generate_persona(self, session_id: str) -> Persona | None:
        """Trigger L3 persona generation if sufficient scenes exist."""
        await self._init_schema()
        if not self.config.l3_enabled:
            return None

        scenes = await get_scenes(session_id, limit=100)
        if len(scenes) < self.config.l3_trigger_every_n_scenes:
            _log.debug("l3 skip: %d scenes (< threshold %d)", len(scenes), self.config.l3_trigger_every_n_scenes)
            return None

        return await generate_persona(session_id, config=self.config)

    # ── Full pipeline run ──────────────────────────────────────

    async def run_pipeline(self, session_id: str) -> dict[str, Any]:
        """Run the full L0→L1→L2→L3 pipeline for a session.

        Best called at session end or after N turns.
        """
        await self._init_schema()

        l1_result = await self.maybe_extract(session_id)
        l2_result = await self.maybe_build_profile(session_id)
        l3_result = await self.maybe_generate_persona(session_id)

        return {
            "session_id": session_id,
            "l1_extracted": len(l1_result),
            "l2_scenes": len(l2_result),
            "l3_persona": l3_result is not None,
        }

    # ── Recall ─────────────────────────────────────────────────

    async def recall(self, query: str, session_id: str) -> RecallResult:
        """Recall relevant context for a query.

        Use at session start to inject memories into the agent context.
        """
        await self._init_schema()
        return await _recall(query, session_id, config=self.config)

    # ── Getters ────────────────────────────────────────────────

    async def get_memories(self, session_id: str, limit: int = 50) -> list[L1Memory]:
        """Get stored L1 memories."""
        await self._init_schema()
        return await get_memories(session_id, limit=limit)

    async def get_persona(self, session_id: str) -> Persona | None:
        """Get the L3 persona."""
        await self._init_schema()
        return await get_persona(session_id)

    async def get_scenes(self, session_id: str, limit: int = 10) -> list[Scene]:
        """Get L2 scenes."""
        await self._init_schema()
        return await get_scenes(session_id, limit=limit)


# Singleton
_pipeline: MemoryPipeline | None = None


def get_pipeline(config: MemoryConfig | None = None) -> MemoryPipeline:
    """Get or create the global MemoryPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = MemoryPipeline(config=config)
    return _pipeline
