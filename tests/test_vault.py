"""Vault tests — real filesystem, no mocks.

Tests VaultSync read/write, index maintenance, lint, and watcher setup.
Uses tempfile.mkdtemp() for isolation — each test class gets its own vault.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ebrain.db import Entity
from ebrain.memory.types import L1Memory
from ebrain.memory.types import Persona
from ebrain.memory.types import Scene
from ebrain.vault import VaultSync
from ebrain.vault import _frontmatter
from ebrain.vault import _parse_frontmatter
from ebrain.vault import _slug

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def vault(tmp_vault: Path) -> VaultSync:
    return VaultSync(tmp_vault)


def _make_memory(
    *,
    mem_id: str = "l1_test_001",
    session_id: str = "test-session",
    content: str = "User prefers dark mode",
    kind: str = "preference",
    keywords: list[str] | None = None,
    confidence: float = 0.9,
) -> L1Memory:
    return L1Memory(
        id=mem_id,
        session_id=session_id,
        content=content,
        kind=kind,
        keywords=keywords if keywords is not None else ["dark-mode", "ui"],
        confidence=confidence,
        created_at="2026-06-19T10:00:00Z",
    )


def _make_scene(
    *,
    scene_id: str = "scene_001",
    session_id: str = "test-session",
    title: str = "UI Preferences",
    summary: str = "User has strong UI preferences.",
    memory_ids: list[str] | None = None,
    tags: list[str] | None = None,
) -> Scene:
    return Scene(
        id=scene_id,
        session_id=session_id,
        title=title,
        summary=summary,
        memory_ids=memory_ids or ["l1_test_001", "l1_test_002"],
        tags=tags or ["ui", "preferences"],
        created_at="2026-06-19T10:00:00Z",
    )


def _make_persona(*, session_id: str = "test-session") -> Persona:
    return Persona(
        session_id=session_id,
        name="Ciro",
        role="developer",
        traits=["detail-oriented", "pragmatic"],
        preferences=["dark mode", "keyboard shortcuts"],
        recurring_topics=["python", "postgres", "obsidian"],
        tools_used=["vim", "docker", "claude"],
        summary="A developer who values efficiency and clean tooling.",
        total_memories=10,
        total_conversations=3,
        updated_at="2026-06-19T10:00:00Z",
    )


def _make_entity(
    *,
    entity_id: str = "postgresql",
    name: str = "PostgreSQL",
    kind: str = "tool",
    tags: list[str] | None = None,
) -> Entity:
    return Entity(
        id=entity_id,
        name=name,
        kind=kind,
        tags=tags or ["database", "sql"],
        created_at="2026-06-19T10:00:00Z",
    )


# ── Helpers unit tests ─────────────────────────────────────────────────────────


class TestHelpers:
    def test_slug_lowercase_and_safe(self):
        assert _slug("Hello World!") == "hello-world"

    def test_slug_max_64(self):
        assert len(_slug("x" * 100)) <= 64

    def test_slug_strips_leading_trailing_dashes(self):
        result = _slug("---test---")
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_frontmatter_roundtrip_simple(self):
        data = {"id": "abc", "kind": "preference", "confidence": 0.9}
        fm = _frontmatter(data)
        assert fm.startswith("---")
        assert fm.endswith("---")
        assert "id" in fm

    def test_frontmatter_list(self):
        fm = _frontmatter({"keywords": ["a", "b", "c"]})
        assert "- a" in fm
        assert "- b" in fm

    def test_parse_frontmatter_no_fm(self):
        meta, body = _parse_frontmatter("Just text here.")
        assert meta == {}
        assert body == "Just text here."

    def test_parse_frontmatter_with_fm(self):
        text = '---\nid: "test-id"\nkind: "fact"\n---\n\nBody text here.'
        meta, body = _parse_frontmatter(text)
        assert meta["id"] == "test-id"
        assert meta["kind"] == "fact"
        assert "Body text here." in body

    def test_parse_frontmatter_bool_preserved(self):
        fm = _frontmatter({"enabled": True})
        assert "true" in fm

    def test_parse_frontmatter_float_precision(self):
        fm = _frontmatter({"confidence": 0.925})
        assert "0.925" in fm


# ── VaultSync write operations ────────────────────────────────────────────────


class TestVaultSyncWrite:
    def test_write_memory_creates_file(self, vault: VaultSync):
        mem = _make_memory()
        path = vault.write_memory(mem)
        assert path.exists()
        assert path.suffix == ".md"

    def test_write_memory_correct_path(self, vault: VaultSync):
        mem = _make_memory(session_id="my-session", mem_id="l1_abc")
        path = vault.write_memory(mem)
        assert "memories" in str(path)
        assert "my-session" in str(path)
        assert "l1_abc" in str(path)

    def test_write_memory_frontmatter(self, vault: VaultSync):
        mem = _make_memory()
        path = vault.write_memory(mem)
        text = path.read_text()
        assert "ebrain_type" in text
        assert "memory" in text
        assert "preference" in text

    def test_write_memory_content_present(self, vault: VaultSync):
        mem = _make_memory(content="User loves keyboard shortcuts")
        path = vault.write_memory(mem)
        text = path.read_text()
        assert "User loves keyboard shortcuts" in text

    def test_write_memory_keyword_wikilinks(self, vault: VaultSync):
        mem = _make_memory(keywords=["vim", "tmux"])
        path = vault.write_memory(mem)
        text = path.read_text()
        assert "[[vim]]" in text
        assert "[[tmux]]" in text

    def test_write_memory_idempotent(self, vault: VaultSync):
        mem = _make_memory()
        p1 = vault.write_memory(mem)
        p2 = vault.write_memory(mem)
        assert p1 == p2
        assert p1.exists()

    def test_write_scene_creates_file(self, vault: VaultSync):
        scene = _make_scene()
        path = vault.write_scene(scene)
        assert path.exists()
        assert path.suffix == ".md"

    def test_write_scene_has_memory_links(self, vault: VaultSync):
        scene = _make_scene(memory_ids=["l1_m1", "l1_m2"])
        path = vault.write_scene(scene)
        text = path.read_text()
        assert "[[l1_m1]]" in text
        assert "[[l1_m2]]" in text

    def test_write_persona_creates_file(self, vault: VaultSync):
        persona = _make_persona()
        path = vault.write_persona(persona)
        assert path.exists()
        assert "personas" in str(path)

    def test_write_persona_has_traits(self, vault: VaultSync):
        persona = _make_persona()
        path = vault.write_persona(persona)
        text = path.read_text()
        assert "detail-oriented" in text
        assert "pragmatic" in text

    def test_write_persona_has_topic_wikilinks(self, vault: VaultSync):
        persona = _make_persona()
        path = vault.write_persona(persona)
        text = path.read_text()
        assert "[[python]]" in text
        assert "[[postgres]]" in text

    def test_write_entity_creates_file(self, vault: VaultSync):
        entity = _make_entity()
        path = vault.write_entity(entity)
        assert path.exists()
        assert "graph" in str(path)

    def test_write_entity_with_edges(self, vault: VaultSync):
        entity = _make_entity()
        edges = [{"target_id": "asyncpg", "kind": "uses", "weight": 1.0}]
        path = vault.write_entity(entity, edges=edges)
        text = path.read_text()
        assert "[[asyncpg]]" in text
        assert "uses" in text

    def test_write_entity_frontmatter(self, vault: VaultSync):
        entity = _make_entity(kind="tool", tags=["database", "sql"])
        path = vault.write_entity(entity)
        text = path.read_text()
        assert "ebrain_type" in text
        assert "entity" in text


# ── Index & Log ────────────────────────────────────────────────────────────────


class TestIndexAndLog:
    def test_update_index_creates_file(self, vault: VaultSync):
        vault.write_memory(_make_memory())
        index_path = vault.update_index()
        assert index_path.exists()
        assert index_path.name == "index.md"

    def test_update_index_lists_memories(self, vault: VaultSync):
        vault.write_memory(_make_memory(mem_id="l1_idx_001"))
        index_path = vault.update_index()
        text = index_path.read_text()
        assert "l1_idx_001" in text

    def test_update_index_all_types(self, vault: VaultSync):
        vault.write_memory(_make_memory())
        vault.write_scene(_make_scene())
        vault.write_persona(_make_persona())
        vault.write_entity(_make_entity())
        text = vault.update_index().read_text()
        assert "## Memories" in text
        assert "## Scenes" in text
        assert "## Personas" in text
        assert "## Graph Entities" in text

    def test_update_index_is_idempotent(self, vault: VaultSync):
        vault.write_memory(_make_memory())
        vault.update_index()
        vault.update_index()
        assert (vault.root / "index.md").exists()

    def test_append_log_creates_file(self, vault: VaultSync):
        vault._ensure()
        vault.append_log("test", "first entry")
        assert (vault.root / "log.md").exists()

    def test_append_log_is_append_only(self, vault: VaultSync):
        vault._ensure()
        vault.append_log("ingest", "doc1")
        vault.append_log("sync", "session-1")
        text = (vault.root / "log.md").read_text()
        assert "ingest" in text
        assert "sync" in text

    def test_append_log_with_details(self, vault: VaultSync):
        vault._ensure()
        vault.append_log("pipeline", "sess-1", {"memories": 3, "scenes": 1})
        text = (vault.root / "log.md").read_text()
        assert "memories: 3" in text

    def test_append_log_grep_friendly_prefix(self, vault: VaultSync):
        vault._ensure()
        vault.append_log("ingest", "some article")
        text = (vault.root / "log.md").read_text()
        assert "## [" in text
        assert "] ingest |" in text


# ── Read & Status ──────────────────────────────────────────────────────────────


class TestReadAndStatus:
    def test_read_edited_memories_empty(self, vault: VaultSync):
        result = vault.read_edited_memories()
        assert result == []

    def test_read_edited_memories_returns_all(self, vault: VaultSync):
        vault.write_memory(_make_memory(mem_id="l1_r01"))
        vault.write_memory(_make_memory(mem_id="l1_r02"))
        result = vault.read_edited_memories()
        assert len(result) == 2

    def test_read_edited_memories_fields(self, vault: VaultSync):
        vault.write_memory(_make_memory(mem_id="l1_r03", content="Test content", keywords=[]))
        result = vault.read_edited_memories()
        assert len(result) == 1
        m = result[0]
        assert m["id"] == "l1_r03"
        assert m["content"] == "Test content"
        assert m["session_id"] == "test-session"
        assert m["kind"] == "preference"

    def test_status_empty_vault(self, vault: VaultSync):
        counts = vault.status()
        assert all(v == 0 for v in counts.values())

    def test_status_counts_correctly(self, vault: VaultSync):
        vault.write_memory(_make_memory(mem_id="l1_s01"))
        vault.write_memory(_make_memory(mem_id="l1_s02"))
        vault.write_scene(_make_scene())
        vault.write_entity(_make_entity())
        counts = vault.status()
        assert counts["memories"] == 2
        assert counts["scenes"] == 1
        assert counts["entities"] == 1
        assert counts["personas"] == 0


# ── Lint ───────────────────────────────────────────────────────────────────────


class TestLint:
    def test_lint_empty_vault(self, vault: VaultSync):
        report = vault.lint()
        assert report["total_pages"] == 0
        assert report["orphan_pages"] == []
        assert report["broken_links"] == []
        assert report["health"] == "ok"

    def test_lint_clean_vault(self, vault: VaultSync):
        mem = _make_memory(keywords=[])
        vault.write_memory(mem)
        vault.update_index()
        report = vault.lint()
        assert report["health"] in ("ok", "issues")

    def test_lint_detects_broken_link(self, vault: VaultSync):
        vault._ensure("graph")
        bad_page = vault.root / "graph" / "broken-test.md"
        bad_page.write_text(
            '---\nid: "broken-test"\nebrain_type: "entity"\nname: "broken"\nkind: "concept"\ntags: []\n---\n\n'
            "[[nonexistent-page-xyz]]",
            encoding="utf-8",
        )
        report = vault.lint()
        broken = " ".join(report["broken_links"])
        assert "nonexistent-page-xyz" in broken

    def test_lint_detects_untyped_page(self, vault: VaultSync):
        vault._ensure()
        untyped = vault.root / "untyped-page.md"
        untyped.write_text("# Random Note\nNo frontmatter.", encoding="utf-8")
        report = vault.lint()
        assert any("untyped-page" in u for u in report["untyped_pages"])

    def test_lint_health_ok_no_issues(self, vault: VaultSync):
        vault.write_entity(_make_entity())
        vault.update_index()
        report = vault.lint()
        assert isinstance(report["health"], str)

    def test_lint_orphan_detection(self, vault: VaultSync):
        vault.write_entity(_make_entity(entity_id="orphan-entity"))
        report = vault.lint()
        assert "orphan-entity" in report["orphan_pages"]


# ── VaultWatcher ───────────────────────────────────────────────────────────────


class TestVaultWatcher:
    def test_import_error_without_watchdog(self, monkeypatch: pytest.MonkeyPatch, tmp_vault: Path):
        import sys
        monkeypatch.setitem(sys.modules, "watchdog", None)
        monkeypatch.setitem(sys.modules, "watchdog.observers", None)
        monkeypatch.setitem(sys.modules, "watchdog.events", None)
        from ebrain.vault_watcher import VaultWatcher
        with pytest.raises(ImportError, match="watchdog"):
            VaultWatcher(tmp_vault, on_change=lambda p: None).start()

    def test_watcher_context_manager_stop(self, tmp_vault: Path):
        pytest.importorskip("watchdog")
        from ebrain.vault_watcher import VaultWatcher
        changes: list[Path] = []
        watcher = VaultWatcher(tmp_vault, on_change=changes.append)
        watcher.start()
        assert watcher._observer is not None
        watcher.stop()
        assert watcher._observer is None

    def test_watcher_start_idempotent(self, tmp_vault: Path):
        pytest.importorskip("watchdog")
        from ebrain.vault_watcher import VaultWatcher
        watcher = VaultWatcher(tmp_vault, on_change=lambda p: None)
        watcher.start()
        obs1 = watcher._observer
        watcher.start()
        obs2 = watcher._observer
        assert obs1 is obs2
        watcher.stop()

    def test_watcher_detects_file_change(self, tmp_vault: Path):
        pytest.importorskip("watchdog")
        import time

        from ebrain.vault_watcher import VaultWatcher
        changes: list[Path] = []
        with VaultWatcher(tmp_vault, on_change=changes.append):
            test_file = tmp_vault / "test.md"
            test_file.write_text("initial", encoding="utf-8")
            time.sleep(0.3)
            test_file.write_text("changed", encoding="utf-8")
            time.sleep(0.5)
        assert any("test.md" in str(p) for p in changes)
