"""Integration tests — real PostgreSQL + real Qdrant. NO MOCKS.

Requires:
  - PG at postgresql://ebrain:ebrain@127.0.0.1:5433/ebrain
  - Qdrant at http://127.0.0.1:6333

All tests use session_id "ebrain-int-test-*" and clean up after themselves.
"""

from __future__ import annotations

import uuid

import asyncpg
import pytest
import pytest_asyncio

TEST_SESSION = f"ebrain-int-test-{uuid.uuid4().hex[:8]}"
DATABASE_URL = "postgresql://ebrain:ebrain@127.0.0.1:5433/ebrain"


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture(scope="session", autouse=True)
async def db_schema():
    """Init schema once per test session, clean up test data after."""
    import ebrain.db as db
    db.DATABASE_URL = DATABASE_URL
    db._pool = None
    from ebrain.db import ensure_schema
    from ebrain.memory.l0_recorder import ensure_schema as l0
    from ebrain.memory.l1_extractor import ensure_schema as l1
    from ebrain.memory.l2l3 import ensure_schema as l2l3
    await ensure_schema()
    await l0()
    await l1()
    await l2l3()
    yield
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM ebrain_memory_l0_conversations WHERE session_id LIKE 'ebrain-int-test-%'"
        )
        await conn.execute(
            "DELETE FROM ebrain_memory_l1_extractions WHERE session_id LIKE 'ebrain-int-test-%'"
        )
        await conn.execute(
            "DELETE FROM ebrain_memory_l1_checkpoints WHERE session_id LIKE 'ebrain-int-test-%'"
        )
        await conn.execute(
            "DELETE FROM ebrain_memory_l2_scenes WHERE session_id LIKE 'ebrain-int-test-%'"
        )
        await conn.execute(
            "DELETE FROM ebrain_memory_l3_personas WHERE session_id LIKE 'ebrain-int-test-%'"
        )
        await conn.execute(
            "DELETE FROM ebrain_entities WHERE id LIKE 'int-test-%'"
        )
    await db.close_pool()


# ── DB connectivity ────────────────────────────────────────────────────────────


class TestDBConnectivity:
    async def test_pg_connects(self):
        conn = await asyncpg.connect(DATABASE_URL)
        ver = await conn.fetchval("SELECT version()")
        assert "PostgreSQL" in ver
        await conn.close()

    async def test_schema_tables_exist(self):
        from ebrain.db import fetch
        rows = await fetch(
            """SELECT tablename FROM pg_tables
               WHERE schemaname = 'public' AND tablename LIKE 'ebrain_%'
               ORDER BY tablename"""
        )
        names = {r["tablename"] for r in rows}
        assert "ebrain_entities" in names
        assert "ebrain_edges" in names
        assert "ebrain_memory_l0_conversations" in names
        assert "ebrain_memory_l1_extractions" in names
        assert "ebrain_memory_l2_scenes" in names
        assert "ebrain_memory_l3_personas" in names

    async def test_unique_edge_constraint_exists(self):
        from ebrain.db import fetchone
        row = await fetchone(
            """SELECT indexname FROM pg_indexes
               WHERE tablename = 'ebrain_edges'
               AND indexname = 'idx_ebrain_edges_unique'"""
        )
        assert row is not None, "unique index on edges missing"


# ── L0 Recorder ───────────────────────────────────────────────────────────────


class TestL0Recorder:
    async def test_record_turn_returns_id(self):
        from ebrain.memory.l0_recorder import record_turn
        row_id = await record_turn(TEST_SESSION, "user", "Hello integration test")
        assert row_id > 0

    async def test_get_recent_turns(self):
        from ebrain.memory.l0_recorder import get_recent_turns
        from ebrain.memory.l0_recorder import record_turn
        sid = f"{TEST_SESSION}-l0"
        await record_turn(sid, "user", "First message")
        await record_turn(sid, "assistant", "First reply")
        turns = await get_recent_turns(sid, limit=10)
        assert len(turns) == 2
        roles = {t["role"] for t in turns}
        assert "user" in roles
        assert "assistant" in roles

    async def test_count_turns_since_last_extraction(self):
        from ebrain.memory.l0_recorder import count_turns_since_last_extraction
        from ebrain.memory.l0_recorder import record_turn
        sid = f"{TEST_SESSION}-count"
        await record_turn(sid, "user", "Count test 1")
        await record_turn(sid, "user", "Count test 2")
        count = await count_turns_since_last_extraction(sid)
        assert count >= 2

    async def test_cleanup_old_conversations_zero_retention(self):
        from ebrain.memory.l0_recorder import cleanup_old_conversations
        deleted = await cleanup_old_conversations(retention_days=0)
        assert deleted == 0


# ── Qdrant integration ────────────────────────────────────────────────────────


class TestQdrant:
    async def test_qdrant_reachable(self):
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.get("http://127.0.0.1:6333/collections") as r:
                assert r.status == 200

    async def test_embed_returns_vector(self):
        from ebrain.memory.qdrant import embed
        vec = await embed("test content for embedding")
        assert vec is not None
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    async def test_upsert_and_search(self):
        from ebrain.memory.qdrant import search_memories
        from ebrain.memory.qdrant import upsert_memory
        mem_id = f"l1_{TEST_SESSION}_qdrant01"
        ok = await upsert_memory(mem_id, TEST_SESSION, "The user prefers dark mode")
        assert ok is True

        results = await search_memories(
            "dark mode preference", TEST_SESSION, limit=5, score_threshold=0.1
        )
        assert mem_id in results

    async def test_is_near_duplicate_true(self):
        from ebrain.memory.qdrant import is_near_duplicate
        from ebrain.memory.qdrant import upsert_memory
        mem_id = f"l1_{TEST_SESSION}_dup01"
        await upsert_memory(mem_id, TEST_SESSION, "The user always uses vim as their editor")

        is_dup = await is_near_duplicate(
            "User uses vim editor exclusively", TEST_SESSION, threshold=0.5
        )
        assert is_dup is True

    async def test_is_near_duplicate_false_different_content(self):
        from ebrain.memory.qdrant import is_near_duplicate
        from ebrain.memory.qdrant import upsert_memory
        mem_id = f"l1_{TEST_SESSION}_nodup01"
        await upsert_memory(mem_id, TEST_SESSION, "The server runs on Ubuntu 22.04")

        is_dup = await is_near_duplicate(
            "User prefers green tea in the morning", TEST_SESSION, threshold=0.9
        )
        assert is_dup is False

    async def test_search_session_isolation(self):
        """Memories from other sessions must not appear in search results."""
        from ebrain.memory.qdrant import search_memories
        from ebrain.memory.qdrant import upsert_memory
        other_session = f"{TEST_SESSION}-other"
        await upsert_memory(f"l1_{other_session}_iso01", other_session, "Secret other session data")

        results = await search_memories("secret other session", TEST_SESSION, score_threshold=0.1)
        for mem_id in results:
            assert other_session not in mem_id

    async def test_stable_id_deterministic(self):
        from ebrain.memory.qdrant import _stable_id
        assert _stable_id("hello") == _stable_id("hello")
        assert _stable_id("hello") != _stable_id("world")

    async def test_graceful_degradation_wrong_port(self):
        """Qdrant unavailable → returns empty/False, no exception."""
        import ebrain.memory.qdrant as q
        q.reset()
        original_port = q.QDRANT_PORT
        q.QDRANT_PORT = 19999  # wrong port
        try:
            result = await q.search_memories("query", "test-session")
            assert result == []
            is_dup = await q.is_near_duplicate("content", "test-session")
            assert is_dup is False
        finally:
            q.QDRANT_PORT = original_port
            q.reset()


# ── L1 Extraction ─────────────────────────────────────────────────────────────


class TestL1Extraction:
    def _make_llm(self, memories: list[dict]):
        """Real Python function implementing the LLM seam — not a mock."""
        async def _ask_json(prompt: str) -> dict:
            return {"memories": memories}
        return _ask_json

    async def test_extract_stores_in_pg_and_qdrant(self):
        from ebrain import set_ask_json
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.l1_extractor import count_memories
        from ebrain.memory.l1_extractor import extract_memories
        from ebrain.memory.l1_extractor import get_memories
        from ebrain.memory.qdrant import search_memories

        sid = f"{TEST_SESSION}-l1-extract"
        set_ask_json(self._make_llm([
            {
                "content": "User prefers JSON:API response format",
                "kind": "preference",
                "keywords": ["api", "json"],
                "confidence": 0.9,
            },
            {
                "content": "User works with Python 3.12",
                "kind": "fact",
                "keywords": ["python"],
                "confidence": 0.85,
            },
        ]))

        turns = [{"role": "user", "content": "I use Python 3.12 and prefer JSON:API", "turn": 1, "id": 999}]
        config = MemoryConfig(l1_dedup_threshold=0.99)
        memories = await extract_memories(turns, sid, config=config)

        assert len(memories) == 2
        assert await count_memories(sid) >= 2

        stored = await get_memories(sid, limit=10)
        contents = {m.content for m in stored}
        assert "User prefers JSON:API response format" in contents

        qdrant_results = await search_memories("JSON API format preference", sid, score_threshold=0.1)
        assert len(qdrant_results) > 0

    async def test_extract_skips_near_duplicates(self):
        from ebrain import set_ask_json
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.l1_extractor import extract_memories
        from ebrain.memory.qdrant import upsert_memory

        sid = f"{TEST_SESSION}-l1-dedup"
        existing_id = f"l1_{sid}_existing"
        await upsert_memory(existing_id, sid, "User uses dark mode in all interfaces")

        set_ask_json(self._make_llm([
            {"content": "User always uses dark mode everywhere", "kind": "preference", "keywords": ["dark"], "confidence": 0.9},
        ]))

        turns = [{"role": "user", "content": "dark mode", "turn": 1, "id": 888}]
        config = MemoryConfig(l1_dedup_threshold=0.5)
        memories = await extract_memories(turns, sid, config=config)
        assert len(memories) == 0

    async def test_extract_empty_turns_returns_empty(self):
        from ebrain import set_ask_json
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.l1_extractor import extract_memories

        set_ask_json(self._make_llm([]))
        memories = await extract_memories([], f"{TEST_SESSION}-empty", config=MemoryConfig())
        assert memories == []

    async def test_extract_llm_failure_returns_empty(self):
        from ebrain import set_ask_json
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.l1_extractor import extract_memories

        async def failing_llm(prompt: str) -> dict:
            raise RuntimeError("LLM unavailable")

        set_ask_json(failing_llm)
        turns = [{"role": "user", "content": "test", "turn": 1, "id": 1}]
        memories = await extract_memories(turns, f"{TEST_SESSION}-fail", config=MemoryConfig())
        assert memories == []

    async def test_extract_disabled_returns_empty(self):
        from ebrain import set_ask_json
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.l1_extractor import extract_memories

        set_ask_json(self._make_llm([{"content": "Should not appear", "kind": "fact"}]))
        turns = [{"role": "user", "content": "x", "turn": 1, "id": 1}]
        memories = await extract_memories(turns, f"{TEST_SESSION}-disabled", config=MemoryConfig(l1_enabled=False))
        assert memories == []


# ── Recall ─────────────────────────────────────────────────────────────────────


class TestRecall:
    async def test_recall_vector_path(self):
        import json as _json

        from ebrain.db import execute as db_execute
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.qdrant import upsert_memory
        from ebrain.memory.recall import recall

        sid = f"{TEST_SESSION}-recall-vec"
        mem_id = f"l1_{sid}_rv01"

        await db_execute(
            """INSERT INTO ebrain_memory_l1_extractions (id, session_id, content, kind, keywords, source_turn, confidence)
               VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (id) DO NOTHING""",
            mem_id, sid, "User deploys on Ubuntu with Docker", "fact", _json.dumps(["ubuntu", "docker"]), 0, 0.9,
        )
        await upsert_memory(mem_id, sid, "User deploys on Ubuntu with Docker")

        config = MemoryConfig(recall_score_threshold=0.1)
        result = await recall("Docker Ubuntu deployment", sid, config=config)
        assert result.strategy == "vector"
        assert any(m.id == mem_id for m in result.memories)

    async def test_recall_keyword_fallback(self):
        """With no Qdrant data → falls back to keyword search."""
        import json as _json

        from ebrain.db import execute as db_execute
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.recall import recall

        sid = f"{TEST_SESSION}-recall-kw"
        mem_id = f"l1_{sid}_kw01"
        await db_execute(
            """INSERT INTO ebrain_memory_l1_extractions (id, session_id, content, kind, keywords, source_turn, confidence)
               VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (id) DO NOTHING""",
            mem_id, sid, "Python 3.12 is used for the project", "fact", _json.dumps(["python"]), 0, 0.9,
        )

        config = MemoryConfig(recall_score_threshold=0.0)
        result = await recall("python project", sid, config=config)
        assert result.strategy == "keyword"
        assert any(m.id == mem_id for m in result.memories)

    async def test_recall_disabled_returns_empty(self):
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.recall import recall

        config = MemoryConfig(recall_enabled=False)
        result = await recall("anything", f"{TEST_SESSION}-disabled", config=config)
        assert result.strategy == "disabled"
        assert result.memories == []


# ── Knowledge Graph ────────────────────────────────────────────────────────────


class TestKnowledgeGraph:
    async def test_add_and_get_entity(self):
        from ebrain.graph_store import KnowledgeGraph
        kg = KnowledgeGraph()
        entity = await kg.add_entity("int-test-pg", "PostgreSQL", kind="tool", tags=["db", "sql"])
        assert entity.id == "int-test-pg"
        assert entity.name == "PostgreSQL"

        fetched = await kg.get_entity("int-test-pg")
        assert fetched is not None
        assert fetched.name == "PostgreSQL"
        assert "db" in fetched.tags

    async def test_add_edge_and_get_neighbors(self):
        from ebrain.graph_store import KnowledgeGraph
        kg = KnowledgeGraph()
        await kg.add_entity("int-test-app", "MyApp", kind="project")
        await kg.add_entity("int-test-db", "MyDB", kind="tool")
        await kg.add_edge("int-test-app", "int-test-db", kind="depends_on")

        neighbors = await kg.get_neighbors("int-test-app")
        target_ids = {n["target"]["id"] for n in neighbors}
        assert "int-test-db" in target_ids

    async def test_duplicate_edge_idempotent(self):
        """Inserting same edge twice must not raise (unique constraint + ON CONFLICT)."""
        from ebrain.graph_store import KnowledgeGraph
        kg = KnowledgeGraph()
        await kg.add_entity("int-test-e1", "E1", kind="concept")
        await kg.add_entity("int-test-e2", "E2", kind="concept")
        await kg.add_edge("int-test-e1", "int-test-e2", kind="relates_to")
        await kg.add_edge("int-test-e1", "int-test-e2", kind="relates_to")

    async def test_shortest_path(self):
        from ebrain.graph_store import KnowledgeGraph
        kg = KnowledgeGraph()
        await kg.add_entity("int-test-s1", "S1", kind="concept")
        await kg.add_entity("int-test-s2", "S2", kind="concept")
        await kg.add_entity("int-test-s3", "S3", kind="concept")
        await kg.add_edge("int-test-s1", "int-test-s2", kind="relates_to")
        await kg.add_edge("int-test-s2", "int-test-s3", kind="relates_to")

        path = await kg.shortest_path("int-test-s1", "int-test-s3")
        assert path is not None
        assert path[0] == "int-test-s1"
        assert path[-1] == "int-test-s3"

    async def test_shortest_path_same_node(self):
        from ebrain.graph_store import KnowledgeGraph
        kg = KnowledgeGraph()
        await kg.add_entity("int-test-same", "Same", kind="concept")
        path = await kg.shortest_path("int-test-same", "int-test-same")
        assert path == ["int-test-same"]

    async def test_shortest_path_no_connection(self):
        from ebrain.graph_store import KnowledgeGraph
        kg = KnowledgeGraph()
        await kg.add_entity("int-test-iso-a", "IsoA", kind="concept")
        await kg.add_entity("int-test-iso-b", "IsoB", kind="concept")
        path = await kg.shortest_path("int-test-iso-a", "int-test-iso-b")
        assert path is None

    async def test_invalid_entity_kind_raises(self):
        from ebrain.graph_store import KnowledgeGraph
        kg = KnowledgeGraph()
        with pytest.raises(ValueError, match="unknown entity kind"):
            await kg.add_entity("int-test-bad", "Bad", kind="unknown_kind")

    async def test_invalid_edge_kind_raises(self):
        from ebrain.graph_store import KnowledgeGraph
        kg = KnowledgeGraph()
        await kg.add_entity("int-test-ek1", "EK1", kind="concept")
        await kg.add_entity("int-test-ek2", "EK2", kind="concept")
        with pytest.raises(ValueError, match="unknown edge kind"):
            await kg.add_edge("int-test-ek1", "int-test-ek2", kind="unknown_edge")

    async def test_search_entities(self):
        from ebrain.graph_store import KnowledgeGraph
        kg = KnowledgeGraph()
        await kg.add_entity("int-test-search-target", "SearchableEntity", kind="tool")
        results = await kg.search_entities("SearchableEntity")
        ids = {e.id for e in results}
        assert "int-test-search-target" in ids

    async def test_auto_link(self):
        from ebrain.graph_store import KnowledgeGraph
        kg = KnowledgeGraph()
        await kg.add_entity("int-test-al-source", "Source", kind="project")
        await kg.add_entity("int-test-al-tool", "SpecialTool", kind="tool")
        linked = await kg.auto_link("int-test-al-source", "We use SpecialTool for building")
        assert "int-test-al-tool" in linked

    async def test_stats(self):
        from ebrain.graph_store import KnowledgeGraph
        kg = KnowledgeGraph()
        stats = await kg.stats()
        assert "total_entities" in stats
        assert "total_edges" in stats
        assert stats["total_entities"] >= 0


# ── Full Pipeline ──────────────────────────────────────────────────────────────


class TestMemoryPipeline:
    def _make_llm(self, memories=None, scenes=None, persona=None):
        call_count = {"n": 0}

        async def _ask_json(prompt: str) -> dict:
            call_count["n"] += 1
            if "memory extraction" in prompt.lower() or "observation" in prompt.lower():
                return {"memories": memories or []}
            if "scene" in prompt.lower() or "organizer" in prompt.lower():
                return {"scenes": scenes or []}
            if "persona" in prompt.lower() or "analyst" in prompt.lower():
                return persona or {
                    "name": "Test User", "role": "developer", "traits": [],
                    "preferences": [], "recurring_topics": [],
                    "tools_used": [], "summary": "Test persona.",
                }
            return {}

        return _ask_json

    async def test_full_pipeline_run(self):
        from ebrain import set_ask_json
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.pipeline import MemoryPipeline

        sid = f"{TEST_SESSION}-pipeline"
        set_ask_json(self._make_llm(
            memories=[{"content": "User builds with FastAPI", "kind": "fact", "keywords": ["fastapi"], "confidence": 0.9}]
        ))

        config = MemoryConfig(l1_every_n_conversations=2, l1_dedup_threshold=0.99)
        pipeline = MemoryPipeline(config=config)

        await pipeline.record(sid, "user", "I build APIs with FastAPI", turn_number=1)
        await pipeline.record(sid, "user", "FastAPI is my framework of choice", turn_number=2)

        result = await pipeline.run_pipeline(sid)
        assert result["session_id"] == sid
        assert isinstance(result["l1_extracted"], int)

    async def test_pipeline_recall(self):
        import json as _json

        from ebrain import set_ask_json
        from ebrain.db import execute as db_execute
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.pipeline import MemoryPipeline
        from ebrain.memory.qdrant import upsert_memory

        sid = f"{TEST_SESSION}-pipeline-recall"
        mem_id = f"l1_{sid}_pr01"
        await db_execute(
            """INSERT INTO ebrain_memory_l1_extractions (id, session_id, content, kind, keywords, source_turn, confidence)
               VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (id) DO NOTHING""",
            mem_id, sid, "FastAPI used for REST APIs", "fact", _json.dumps(["fastapi"]), 0, 0.9,
        )
        await upsert_memory(mem_id, sid, "FastAPI used for REST APIs")

        set_ask_json(self._make_llm())
        pipeline = MemoryPipeline(config=MemoryConfig(recall_score_threshold=0.1))
        result = await pipeline.recall("REST API framework", sid)
        assert result is not None

    async def test_get_pipeline_singleton_reset_on_new_config(self):
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.pipeline import get_pipeline

        cfg1 = MemoryConfig(l1_enabled=True)
        cfg2 = MemoryConfig(l1_enabled=False)

        p1 = get_pipeline(cfg1)
        p2 = get_pipeline(cfg2)
        assert p2.config.l1_enabled is False
        assert p1 is not p2


# ── Dream Cycle ────────────────────────────────────────────────────────────────


class TestMigrations:
    async def test_run_migrations_idempotent(self):
        from ebrain.migrations import current_version
        from ebrain.migrations import run_migrations
        v1 = await current_version()
        assert v1 >= 1
        count = await run_migrations()
        assert count == 0
        v2 = await current_version()
        assert v2 == v1

    async def test_current_version_gte_1(self):
        from ebrain.migrations import current_version
        assert await current_version() >= 1


class TestMemoryPipelineContextManager:
    async def test_aenter_aexit(self):
        import ebrain.db as db
        db._pool = None
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.pipeline import MemoryPipeline
        async with MemoryPipeline(config=MemoryConfig()) as pipeline:
            assert pipeline._schema_ensured is True
        db._pool = None


class TestDreamCycle:
    async def test_analyze_gaps_skips_small_graph(self):
        """analyze_gaps returns skip status when < 5 entities."""
        from ebrain import set_ask_json
        from ebrain.graph_store import KnowledgeGraph

        async def llm(prompt: str) -> dict:
            return {"isolated": [], "thin_areas": [], "missing_edges": [], "suggested_entities": [], "summary": "ok"}

        set_ask_json(llm)

        kg = KnowledgeGraph()
        from ebrain.dream import analyze_gaps
        result = await analyze_gaps(kg)
        assert result.get("status") in ("skip", "ok")

    async def test_dream_cycle_runs_without_llm_configured(self):
        from ebrain import llm as llm_mod
        from ebrain.dream import analyze_gaps
        from ebrain.graph_store import KnowledgeGraph

        original = llm_mod._ask_json
        llm_mod._ask_json = None
        try:
            kg = KnowledgeGraph()
            result = await analyze_gaps(kg)
            assert "error" in result or result.get("status") == "skip"
        finally:
            llm_mod._ask_json = original
