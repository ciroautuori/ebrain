"""EBrain Vault — Karpathy-pattern wiki sync to Obsidian vault.

Converts EBrain memory layers → Obsidian-compatible markdown files.
Bidirectional: write_* pushes to vault; read_edited_memories pulls human edits back.
Maintains index.md (catalog) and log.md (append-only event log).

Vault structure:
    {vault}/ebrain/
    ├── index.md                         — full catalog, rebuilt on update_index()
    ├── log.md                           — append-only ingest/sync events
    ├── memories/{session}/{mem_id}.md   — L1 memories (facts, preferences, …)
    ├── scenes/{session}/{scene_id}.md   — L2 scenes (thematic clusters)
    ├── personas/{session}.md            — L3 persona (long-term identity)
    └── graph/{entity_id}.md             — knowledge graph entities with wikilinks

Obsidian graph view shows connections via [[wikilinks]] between entities,
keywords, and sessions — making the knowledge graph visually explorable.

Usage:
    vault = VaultSync("/path/to/ObsidianVault")
    vault.write_memory(memory)   # after L1 extraction
    vault.write_entity(entity)   # after KG upsert
    vault.update_index()         # after any batch write
    report = vault.lint()        # periodic health check
"""

from __future__ import annotations

import re
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ebrain.db import Entity
    from ebrain.memory.types import L1Memory
    from ebrain.memory.types import Persona
    from ebrain.memory.types import Scene


# ── Helpers ────────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(text: str) -> str:
    """Filesystem-safe slug, max 64 chars."""
    return re.sub(r"[^a-z0-9_-]", "-", text.lower())[:64].strip("-")


def _frontmatter(data: dict) -> str:
    """Serialize dict to YAML frontmatter block."""
    lines = ["---"]
    for k, v in data.items():
        if isinstance(v, list):
            if v:
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{k}: []")
        elif isinstance(v, float):
            lines.append(f"{k}: {v:.3f}")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        else:
            val = str(v).replace('"', '\\"')
            lines.append(f'{k}: "{val}"')
    lines.append("---")
    return "\n".join(lines)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata_dict, body). Empty dict if no frontmatter."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_block = text[4:end]
    body = text[end + 4:].lstrip("\n")
    meta: dict = {}
    for line in fm_block.splitlines():
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            stripped = v.strip().strip('"')
            meta[k.strip()] = stripped
    return meta, body


# ── VaultSync ──────────────────────────────────────────────────────────────────


class VaultSync:
    """Sync EBrain memory layers to an Obsidian vault (Karpathy LLM Wiki pattern)."""

    def __init__(self, vault_path: str | Path) -> None:
        self.vault_path = Path(vault_path)
        self.root = self.vault_path / "ebrain"

    # ── Directory helpers ──────────────────────────────────────────────────────

    def _ensure(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts) if parts else self.root
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _mem_dir(self, session_id: str) -> Path:
        return self._ensure("memories", _slug(session_id))

    def _scene_dir(self, session_id: str) -> Path:
        return self._ensure("scenes", _slug(session_id))

    def _persona_dir(self) -> Path:
        return self._ensure("personas")

    def _graph_dir(self) -> Path:
        return self._ensure("graph")

    # ── Writers ────────────────────────────────────────────────────────────────

    def write_memory(self, memory: "L1Memory") -> Path:
        """Write L1 memory as Obsidian markdown. Overwrites if exists (idempotent)."""
        created = memory.created_at[:10] if memory.created_at else ""
        fm = _frontmatter({
            "id": memory.id,
            "session": memory.session_id,
            "kind": memory.kind,
            "confidence": memory.confidence,
            "keywords": memory.keywords,
            "created": created,
            "ebrain_type": "memory",
        })
        kw_links = " ".join(f"[[{kw}]]" for kw in memory.keywords) if memory.keywords else ""
        body = f"{fm}\n\n{memory.content}\n"
        if kw_links:
            body += f"\n{kw_links}\n"
        p = self._mem_dir(memory.session_id) / f"{memory.id}.md"
        p.write_text(body, encoding="utf-8")
        return p

    def write_scene(self, scene: "Scene") -> Path:
        """Write L2 scene as Obsidian markdown."""
        created = scene.created_at[:10] if scene.created_at else ""
        fm = _frontmatter({
            "id": scene.id,
            "session": scene.session_id,
            "title": scene.title,
            "tags": scene.tags,
            "memory_count": len(scene.memory_ids),
            "created": created,
            "ebrain_type": "scene",
        })
        mem_links = "\n".join(f"- [[{mid}]]" for mid in scene.memory_ids) or "_none_"
        body = f"{fm}\n\n# {scene.title}\n\n{scene.summary}\n\n## Memories\n{mem_links}\n"
        p = self._scene_dir(scene.session_id) / f"{scene.id}.md"
        p.write_text(body, encoding="utf-8")
        return p

    def write_persona(self, persona: "Persona") -> Path:
        """Write L3 persona as Obsidian markdown."""
        updated = persona.updated_at[:10] if persona.updated_at else ""
        fm = _frontmatter({
            "session": persona.session_id,
            "name": persona.name,
            "role": persona.role,
            "total_memories": persona.total_memories,
            "total_conversations": persona.total_conversations,
            "updated": updated,
            "ebrain_type": "persona",
        })
        traits = "\n".join(f"- {t}" for t in persona.traits) if persona.traits else "_none_"
        prefs = "\n".join(f"- {pr}" for pr in persona.preferences) if persona.preferences else "_none_"
        topics = "\n".join(f"- [[{t}]]" for t in persona.recurring_topics) if persona.recurring_topics else "_none_"
        tools = "\n".join(f"- [[{t}]]" for t in persona.tools_used) if persona.tools_used else "_none_"
        label = persona.name or persona.session_id
        body = (
            f"{fm}\n\n# {label}\n\n"
            f"**Role:** {persona.role}\n\n"
            f"{persona.summary}\n\n"
            f"## Traits\n{traits}\n\n"
            f"## Preferences\n{prefs}\n\n"
            f"## Recurring Topics\n{topics}\n\n"
            f"## Tools Used\n{tools}\n"
        )
        p = self._persona_dir() / f"{_slug(persona.session_id)}.md"
        p.write_text(body, encoding="utf-8")
        return p

    def write_entity(self, entity: "Entity", edges: list[dict] | None = None) -> Path:
        """Write KG entity as Obsidian markdown with wikilinked connections."""
        fm = _frontmatter({
            "id": entity.id,
            "name": entity.name,
            "kind": entity.kind,
            "tags": list(entity.tags),
            "ebrain_type": "entity",
        })
        edge_section = ""
        if edges:
            edge_section = "\n## Connections\n"
            for e in edges:
                target = e.get("target_id", "")
                kind = e.get("kind", "relates_to")
                weight = float(e.get("weight", 1.0))
                edge_section += f"- [[{target}]] `{kind}` (w={weight:.2f})\n"
        body = f"{fm}\n\n# {entity.name}\n\n**Kind:** `{entity.kind}`\n{edge_section}"
        p = self._graph_dir() / f"{entity.id}.md"
        p.write_text(body, encoding="utf-8")
        return p

    # ── Index & Log ────────────────────────────────────────────────────────────

    def update_index(self) -> Path:
        """Rebuild index.md catalog from all pages. Call after any batch write."""
        self._ensure()

        memories: list[tuple[str, str, str]] = []
        scenes: list[tuple[str, str, str]] = []
        personas: list[tuple[str, str]] = []
        entities: list[tuple[str, str, str]] = []

        for f in sorted(self.root.rglob("*.md")):
            if f.name in ("index.md", "log.md"):
                continue
            rel = str(f.relative_to(self.root))
            parts = Path(rel).parts
            if not parts:
                continue
            text = f.read_text(encoding="utf-8")
            meta, _ = _parse_frontmatter(text)
            etype = meta.get("ebrain_type", "")
            if etype == "memory":
                memories.append((rel, meta.get("id", f.stem), meta.get("session", "")))
            elif etype == "scene":
                scenes.append((rel, meta.get("id", f.stem), meta.get("session", "")))
            elif etype == "persona":
                personas.append((rel, meta.get("name", f.stem)))
            elif etype == "entity":
                entities.append((rel, meta.get("id", f.stem), meta.get("name", f.stem)))

        lines = [
            "# EBrain Wiki — Index",
            "",
            (
                f"_Updated: {_now_iso()} · {len(memories)} memories"
                f" · {len(scenes)} scenes · {len(personas)} personas"
                f" · {len(entities)} entities_"
            ),
            "",
        ]

        if memories:
            lines += ["## Memories", "", "| File | ID | Session |", "|---|---|---|"]
            for path, mid, sid in memories:
                lines.append(f"| [[{path}\\|{mid[:20]}]] | `{mid}` | `{sid}` |")
            lines.append("")

        if scenes:
            lines += ["## Scenes", "", "| File | ID | Session |", "|---|---|---|"]
            for path, sid_, ssid in scenes:
                lines.append(f"| [[{path}\\|{sid_[:20]}]] | `{sid_}` | `{ssid}` |")
            lines.append("")

        if personas:
            lines += ["## Personas", "", "| File | Name |", "|---|---|"]
            for path, name in personas:
                lines.append(f"| [[{path}\\|{name}]] | {name} |")
            lines.append("")

        if entities:
            lines += ["## Graph Entities", "", "| File | ID | Name |", "|---|---|---|"]
            for path, eid, ename in entities:
                lines.append(f"| [[{path}\\|{ename}]] | `{eid}` | {ename} |")
            lines.append("")

        index_path = self.root / "index.md"
        index_path.write_text("\n".join(lines), encoding="utf-8")
        return index_path

    def append_log(self, event_type: str, description: str, details: dict | None = None) -> None:
        """Append entry to log.md (append-only, grep-friendly prefix)."""
        log_path = self.root / "log.md"
        ts = _now_iso()
        entry = f"\n## [{ts}] {event_type} | {description}\n"
        if details:
            for k, v in details.items():
                entry += f"- {k}: {v}\n"
        if not log_path.exists():
            self._ensure()
            log_path.write_text("# EBrain Wiki — Log\n", encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(entry)

    # ── Read & Lint ────────────────────────────────────────────────────────────

    def read_edited_memories(self) -> list[dict]:
        """Return all memory pages as dicts (for re-ingestion after human edits)."""
        results = []
        mem_root = self.root / "memories"
        if not mem_root.exists():
            return results
        for f in sorted(mem_root.rglob("*.md")):
            text = f.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(text)
            if meta.get("ebrain_type") == "memory":
                results.append({
                    "id": meta.get("id", f.stem),
                    "session_id": meta.get("session", ""),
                    "kind": meta.get("kind", "fact"),
                    "content": body.strip(),
                    "path": str(f),
                })
        return results

    def lint(self) -> dict:
        """Health check: find orphan pages, broken wikilinks, pages missing ebrain_type."""
        if not self.root.exists():
            return {"total_pages": 0, "orphan_pages": [], "broken_links": [], "untyped_pages": [], "health": "ok"}

        all_stems: set[str] = set()
        all_ids: set[str] = set()
        untyped: list[str] = []

        for f in self.root.rglob("*.md"):
            if f.name in ("index.md", "log.md"):
                continue
            all_stems.add(f.stem)
            text = f.read_text(encoding="utf-8")
            meta, _ = _parse_frontmatter(text)
            if meta.get("id"):
                all_ids.add(meta["id"])
            if not meta.get("ebrain_type"):
                untyped.append(str(f.relative_to(self.root)))

        reachable: set[str] = set()
        broken: list[str] = []
        wikilink_re = re.compile(r"\[\[([^\]|#]+)")

        for f in self.root.rglob("*.md"):
            text = f.read_text(encoding="utf-8")
            for match in wikilink_re.finditer(text):
                target_raw = match.group(1).strip()
                target_stem = Path(target_raw).stem
                if target_stem in all_stems or target_raw in all_ids:
                    reachable.add(target_stem)
                else:
                    broken.append(f"{f.relative_to(self.root)} → [[{target_raw}]]")

        skip = {"index", "log"}
        orphans = sorted((all_stems - reachable) - skip)

        health = "ok" if not orphans and not broken and not untyped else "issues"
        return {
            "total_pages": len(all_stems),
            "orphan_pages": orphans,
            "broken_links": broken,
            "untyped_pages": untyped,
            "health": health,
        }

    def status(self) -> dict:
        """Return page counts per content type."""
        counts: dict[str, int] = {"memories": 0, "scenes": 0, "personas": 0, "entities": 0}
        if not self.root.exists():
            return counts
        subdirs = {"memories": "memories", "scenes": "scenes", "personas": "personas", "entities": "graph"}
        for key, subdir in subdirs.items():
            d = self.root / subdir
            if d.exists():
                counts[key] = sum(1 for _ in d.rglob("*.md"))
        return counts
