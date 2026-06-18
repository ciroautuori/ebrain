"""L2 Scene/Profile Builder + L3 Persona Generator.

Inspired by TencentDB Agent Memory L2/L3 layers:
- L2: Groups L1 memories into coherent "scenes" (thematic clusters)
- L3: Synthesizes a persona from accumulated profiles and memories
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from ebrain.db import execute
from ebrain.db import fetch
from ebrain.llm import ask_json
from ebrain.llm import get_default_model
from ebrain.memory.config import MemoryConfig
from ebrain.memory.l1_extractor import count_memories
from ebrain.memory.l1_extractor import get_memories
from ebrain.memory.types import Persona
from ebrain.memory.types import Scene

_log = logging.getLogger("ebrain.memory.l2l3")

L2_SCHEMA = """
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
"""

L3_SCHEMA = """
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
"""

SCENE_PROMPT = """You are a knowledge organizer. Given a list of extracted
memories from a session, group them into coherent "scenes" (thematic clusters).

Each scene should have:
- title: short descriptive label (max 8 words)
- summary: 2-3 sentences synthesizing the related memories
- memory_ids: which memory ids belong to this scene
- tags: 2-4 keywords

Max scenes: {max_scenes}

Memories:
{memories}

Return JSON:
{{
  "scenes": [
    {{
      "title": "...",
      "summary": "...",
      "memory_ids": ["id1", "id2"],
      "tags": ["tag1", "tag2"]
    }}
  ]
}}"""

PERSONA_PROMPT = """You are a persona analyst. From the accumulated knowledge about a user/agent, synthesize a persona profile.

Scenes (thematic clusters):
{scenes}

Recent memories:
{memories}

Total memories: {total_memories}

Extract:
- name: how the user/agent refers to themselves
- role: primary function (e.g. "content creator", "dev ops", "commerce manager")
- traits: 3-6 personality/habit descriptors
- preferences: 3-8 things the user prefers or avoids
- recurring_topics: 3-6 topics/areas mentioned repeatedly
- tools_used: tools/frameworks/services mentioned
- summary: 3-5 sentence narrative describing this persona

Return JSON with these fields."""


async def ensure_schema() -> None:
    """Create L2/L3 tables (idempotent)."""
    await execute(L2_SCHEMA)
    await execute(L3_SCHEMA)


async def build_scenes(
    session_id: str,
    *,
    config: MemoryConfig,
    model: str | None = None,
) -> list[Scene]:
    """L2: Build scene/profile blocks from L1 memories."""
    if not config.l2_enabled:
        return []

    memories = await get_memories(session_id, limit=200)
    if len(memories) < 5:
        return []

    model = model or config.l2_model or get_default_model()

    memories_str = "\n".join(
        f"ID:{m.id} [{m.kind}] {m.content}" for m in memories
    )

    try:
        result = await ask_json(
            SCENE_PROMPT.format(
                max_scenes=config.l2_max_scenes,
                memories=memories_str[:8000],
            ),
            model=model,
        )
    except Exception as exc:
        _log.warning("l2 scene extraction failed: %s", exc)
        return []

    if not isinstance(result, dict) or "scenes" not in result:
        return []

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    new_scenes: list[Scene] = []

    for raw in result.get("scenes", []):
        if not isinstance(raw, dict):
            continue

        scene = Scene(
            id=f"l2_{session_id}_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            title=str(raw.get("title", "Untitled Scene")),
            summary=str(raw.get("summary", "")),
            memory_ids=list(raw.get("memory_ids", [])),
            tags=list(raw.get("tags", [])),
            created_at=now,
        )

        await execute(
            """INSERT INTO ebrain_memory_l2_scenes (id, session_id, title, summary, memory_ids, tags)
               VALUES ($1, $2, $3, $4, $5, $6)
               ON CONFLICT (id) DO NOTHING""",
            scene.id,
            session_id,
            scene.title,
            scene.summary,
            json.dumps(scene.memory_ids),
            json.dumps(scene.tags),
        )
        new_scenes.append(scene)

    if new_scenes:
        _log.info("l2 scenes: %d new from session %s", len(new_scenes), session_id)

    return new_scenes


async def generate_persona(
    session_id: str,
    *,
    config: MemoryConfig,
    model: str | None = None,
) -> Persona | None:
    """L3: Generate or refresh the persona for a session."""
    if not config.l3_enabled:
        return None

    model = model or config.l3_model or get_default_model()

    # Get existing scenes and recent memories
    scene_rows = await fetch(
        """SELECT title, summary, tags
           FROM ebrain_memory_l2_scenes
           WHERE session_id = $1
           ORDER BY created_at DESC
           LIMIT 15""",
        session_id,
    )
    scenes_str = "\n".join(
        f"## {r['title']}\n{r['summary']}" for r in scene_rows
    ) or "(no scenes yet)"

    recent = await get_memories(session_id, limit=30)
    memories_str = "\n".join(
        f"[{m.kind}] {m.content}" for m in recent
    ) or "(no memories)"

    total = await count_memories(session_id)

    try:
        result = await ask_json(
            PERSONA_PROMPT.format(
                scenes=scenes_str[:5000],
                memories=memories_str[:5000],
                total_memories=total,
            ),
            model=model,
        )
    except Exception as exc:
        _log.warning("l3 persona generation failed: %s", exc)
        return None

    if not isinstance(result, dict):
        return None

    persona = Persona(
        session_id=session_id,
        name=str(result.get("name", "")),
        role=str(result.get("role", "")),
        traits=list(result.get("traits", [])),
        preferences=list(result.get("preferences", [])),
        recurring_topics=list(result.get("recurring_topics", [])),
        tools_used=list(result.get("tools_used", [])),
        summary=str(result.get("summary", "")),
        total_memories=total,
        updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # Upsert persona
    await execute(
        """INSERT INTO ebrain_memory_l3_personas
               (session_id, name, role, traits, preferences, recurring_topics, tools_used, summary, total_memories)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
           ON CONFLICT (session_id) DO UPDATE
           SET name = EXCLUDED.name,
               role = EXCLUDED.role,
               traits = EXCLUDED.traits,
               preferences = EXCLUDED.preferences,
               recurring_topics = EXCLUDED.recurring_topics,
               tools_used = EXCLUDED.tools_used,
               summary = EXCLUDED.summary,
               total_memories = EXCLUDED.total_memories,
               updated_at = NOW()""",
        persona.session_id,
        persona.name,
        persona.role,
        json.dumps(persona.traits),
        json.dumps(persona.preferences),
        json.dumps(persona.recurring_topics),
        json.dumps(persona.tools_used),
        persona.summary,
        persona.total_memories,
    )

    _log.info("l3 persona: updated for session %s (%d memories)", session_id, total)
    return persona


async def get_persona(session_id: str) -> Persona | None:
    """Retrieve the L3 persona for a session."""
    row = await fetch(
        """SELECT * FROM ebrain_memory_l3_personas WHERE session_id = $1""",
        session_id,
    )
    if not row:
        return None
    r = row[0]
    return Persona(
        session_id=r["session_id"],
        name=r["name"] or "",
        role=r["role"] or "",
        traits=json.loads(r["traits"]) if isinstance(r["traits"], str) else r["traits"] or [],
        preferences=json.loads(r["preferences"]) if isinstance(r["preferences"], str) else r["preferences"] or [],
        recurring_topics=json.loads(r["recurring_topics"]) if isinstance(r["recurring_topics"], str) else r["recurring_topics"] or [],  # noqa: E501
        tools_used=json.loads(r["tools_used"]) if isinstance(r["tools_used"], str) else r["tools_used"] or [],
        summary=r["summary"] or "",
        total_memories=int(r["total_memories"] or 0),
        total_conversations=int(r["total_conversations"] or 0),
        updated_at=str(r["updated_at"]),
    )


async def get_scenes(session_id: str, limit: int = 10) -> list[Scene]:
    """Retrieve L2 scenes for a session."""
    rows = await fetch(
        """SELECT * FROM ebrain_memory_l2_scenes
           WHERE session_id = $1
           ORDER BY created_at DESC
           LIMIT $2""",
        session_id,
        limit,
    )
    return [
        Scene(
            id=r["id"],
            session_id=r["session_id"],
            title=r["title"],
            summary=r["summary"],
            memory_ids=json.loads(r["memory_ids"]) if isinstance(r["memory_ids"], str) else r["memory_ids"],
            tags=json.loads(r["tags"]) if isinstance(r["tags"], str) else r["tags"],
            created_at=str(r["created_at"]),
        )
        for r in rows
    ]
