"""EBrain Entity Extractor — extract typed entities from text.

Lightweight rule-based + LLM extraction. Creates entities and edges
in the knowledge graph automatically from conversations and documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field

from ebrain.db import Entity


@dataclass
class EntityExtractor:
    """Extracts entities from text and adds them to a KnowledgeGraph."""

    min_confidence: float = 0.6

    # Simple known entity patterns (extendable)
    _patterns: dict[str, list[str]] = field(default_factory=lambda: {
        "tool": ["claude", "opencode", "docker", "git", "nginx", "postgres", "qdrant", "fastapi"],
        "framework": ["react", "next.js", "tailwind", "typescript", "python", "rust"],
        "platform": ["github", "vercel", "stripe", "tiktok", "youtube", "instagram"],
    })

    def extract(
        self, text: str, *, existing_ids: set[str] | None = None,
    ) -> list[Entity]:
        """Extract entities from text using pattern matching.

        For production use, inject an LLM-based extractor via set_ask_json().
        """
        found: list[Entity] = []
        seen = existing_ids or set()
        text_lower = text.lower()

        for kind, terms in self._patterns.items():
            for term in terms:
                if term in text_lower and term not in seen:
                    eid = self._to_slug(term)
                    found.append(Entity(id=eid, name=term.title(), kind=kind, tags=[kind]))
                    seen.add(eid)

        return found

    async def extract_and_store(
        self, graph, text: str,  # graph: KnowledgeGraph
    ) -> list[Entity]:
        """Extract entities and add them to the graph in one step."""
        existing = {e.id for e in await graph.list_entities(limit=500)}
        entities = self.extract(text, existing_ids=existing)
        for e in entities:
            await graph.add_entity(e.id, e.name, kind=e.kind, tags=e.tags)
        return entities

    @staticmethod
    def _to_slug(name: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9\s-]", "", name.lower())
        return re.sub(r"\s+", "-", s).strip("-")[:80]
