"""TDD for ebrain — standalone agent memory + knowledge graph.

Tests are $0: no Qdrant, no PostgreSQL needed for unit tests.
Uses in-memory mocks where needed.
"""

from __future__ import annotations


class TestDBTypes:
    """Entity and Edge data classes."""

    def test_entity_creation(self):
        from ebrain.db import Entity

        e = Entity(id="e1", name="Test Entity", kind="tool", tags=["ai", "cli"])
        assert e.id == "e1"
        assert e.name == "Test Entity"
        assert e.kind == "tool"
        assert e.tags == ["ai", "cli"]

    def test_edge_creation(self):
        from ebrain.db import Edge

        e = Edge(source_id="a", target_id="b", kind="depends_on", weight=0.8)
        assert e.source_id == "a"
        assert e.target_id == "b"
        assert e.kind == "depends_on"
        assert e.weight == 0.8


class TestMemoryConfig:
    """MemoryConfig defaults and env parsing."""

    def test_defaults(self):
        from ebrain.memory.config import MemoryConfig

        cfg = MemoryConfig()
        assert cfg.l0_enabled is True
        assert cfg.l1_enabled is True
        assert cfg.l1_every_n_conversations == 5
        assert cfg.recall_max_results == 5

    def test_from_env(self, monkeypatch):
        from ebrain.memory.config import MemoryConfig

        monkeypatch.setenv("EROS_MEMORY_L1_ENABLED", "false")
        monkeypatch.setenv("EROS_MEMORY_L1_EVERY_N", "10")
        cfg = MemoryConfig.from_env()
        assert cfg.l1_enabled is False
        assert cfg.l1_every_n_conversations == 10


class TestL1Memory:
    """L1Memory formatting."""

    def test_to_injection(self):
        from ebrain.memory.types import L1Memory

        mem = L1Memory(
            id="m1", session_id="s1",
            content="User prefers dark mode",
            kind="preference", keywords=["dark mode"],
        )
        text = mem.to_injection()
        assert "[preference]" in text
        assert "dark mode" in text

    def test_to_injection_truncation(self):
        from ebrain.memory.types import L1Memory

        mem = L1Memory(
            id="m2", session_id="s1",
            content="A very long memory that should be truncated",
            kind="fact",
        )
        text = mem.to_injection(max_chars=20)
        assert len(text) <= 23
        assert text.endswith("...")


class TestScene:
    """Scene formatting."""

    def test_to_injection(self):
        from ebrain.memory.types import Scene

        scene = Scene(
            id="s1", session_id="s1",
            title="Dashboard", summary="Dark theme.",
            tags=["ui"],
        )
        text = scene.to_injection()
        assert "Dashboard" in text
        assert "Dark theme" in text


class TestPersona:
    """Persona formatting."""

    def test_to_injection(self):
        from ebrain.memory.types import Persona

        p = Persona(
            session_id="s1", name="Dev", role="engineer",
            traits=["fast"], preferences=["vim"],
            summary="Works on AI.",
        )
        text = p.to_injection()
        assert "Persona: Dev" in text
        assert "engineer" in text
        assert "vim" in text


class TestRecallResult:
    """RecallResult context formatting."""

    def test_empty(self):
        from ebrain.memory.types import RecallResult

        r = RecallResult()
        assert r.format_context() == ""

    def test_with_persona(self):
        from ebrain.memory.types import Persona
        from ebrain.memory.types import RecallResult

        p = Persona(session_id="s1", name="X", summary="Works.")
        r = RecallResult(persona=p)
        assert "Persona: X" in r.format_context()

    def test_with_memories(self):
        from ebrain.memory.types import L1Memory
        from ebrain.memory.types import RecallResult

        mems = [
            L1Memory(id="m1", session_id="s1", content="Fact 1", kind="fact"),
            L1Memory(id="m2", session_id="s1", content="Fact 2", kind="fact"),
        ]
        r = RecallResult(memories=mems)
        ctx = r.format_context()
        assert "Fact 1" in ctx
        assert "Fact 2" in ctx


class TestOffload:
    """Symbolic compression helpers."""

    def test_offload_none(self):
        from ebrain.memory.offload import offload_tool_result

        assert offload_tool_result(None) == "[EMPTY]"

    def test_offload_error_dict(self):
        from ebrain.memory.offload import offload_tool_result

        r = offload_tool_result({"error": "Connection refused"})
        assert "[ERR]" in r
        assert "Connection refused" in r

    def test_offload_status_dict(self):
        from ebrain.memory.offload import offload_tool_result

        r = offload_tool_result({"status": "ok", "id": "abc"})
        assert "[ok]" in r

    def test_offload_list(self):
        from ebrain.memory.offload import offload_tool_result

        r = offload_tool_result(["a", "b", "c"])
        assert "[3]" in r
        assert "a, b, c" in r

    def test_offload_long_text(self):
        from ebrain.memory.offload import offload_tool_result

        r = offload_tool_result("x" * 3000, max_chars=50)
        assert len(r) <= 53
        assert r.endswith("...")

    def test_estimate_tokens(self):
        from ebrain.memory.offload import estimate_tokens

        assert estimate_tokens("") == 1
        assert estimate_tokens("hello world") == 2


class TestPipeline:
    """MemoryPipeline initialization."""

    def test_default_pipeline(self):
        from ebrain.memory.pipeline import MemoryPipeline

        p = MemoryPipeline()
        assert p.config.l1_enabled is True

    def test_custom_pipeline(self):
        from ebrain.memory.config import MemoryConfig
        from ebrain.memory.pipeline import MemoryPipeline

        cfg = MemoryConfig(l1_enabled=False, l2_enabled=False)
        p = MemoryPipeline(config=cfg)
        assert p.config.l1_enabled is False


class TestEntityExtractor:
    """Entity extraction from text."""

    def test_extract_tools(self):
        from ebrain.entities import EntityExtractor

        extractor = EntityExtractor()
        entities = extractor.extract("I use claude and docker to build the app")
        names = {e.name for e in entities}
        assert "Claude" in names
        assert "Docker" in names

    def test_extract_no_duplicates(self):
        from ebrain.entities import EntityExtractor

        extractor = EntityExtractor()
        entities = extractor.extract("claude is great. claude helps me.", existing_ids={"claude"})
        assert len(entities) == 0

    def test_to_slug(self):
        from ebrain.entities import EntityExtractor

        assert EntityExtractor._to_slug("Hello World!") == "hello-world"
        assert EntityExtractor._to_slug("FastAPI") == "fastapi"
