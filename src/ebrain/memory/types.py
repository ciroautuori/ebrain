"""Data types for the 4-layer memory system."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass
class L1Memory:
    """A single extracted memory fact (L1).

    Stored in PG with vector embedding in Qdrant.
    Similarity-based dedup happens at extraction time.
    """

    id: str
    session_id: str
    content: str  # the extracted fact/observation
    kind: str = "fact"  # fact | preference | decision | pattern | question
    keywords: list[str] = field(default_factory=list)
    source_turn: int = 0  # conversation turn that generated this
    confidence: float = 0.8  # LLM self-assessed confidence 0-1
    created_at: str = ""  # ISO 8601
    embedding: list[float] | None = None  # vector for Qdrant (set after storage)

    def to_injection(self, max_chars: int = 0) -> str:
        """Format memory for injection into agent context."""
        text = f"[{self.kind}] {self.content}"
        if self.keywords:
            text += f" (#{' #'.join(self.keywords)})"
        if max_chars and len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        return text


@dataclass
class Scene:
    """A scene/profile block (L2) — groups related L1 memories into a coherent context."""

    id: str
    session_id: str
    title: str  # short descriptive title
    summary: str  # 2-3 sentence summary
    memory_ids: list[str] = field(default_factory=list)  # L1 memories in this scene
    tags: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_injection(self) -> str:
        """Format scene for LLM context injection."""
        tag_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"## {self.title}{tag_str}\n{self.summary}"


@dataclass
class Persona:
    """Agent persona (L3) — long-term identity built from profiles and memories."""

    session_id: str
    name: str = ""
    role: str = ""  # e.g. "content creator", "dev ops", "commerce manager"
    traits: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    recurring_topics: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    summary: str = ""  # 3-5 sentence narrative
    total_memories: int = 0
    total_conversations: int = 0
    updated_at: str = ""

    def to_injection(self) -> str:
        """Format persona for LLM context injection (compact)."""
        parts = []
        if self.name:
            parts.append(f"# Persona: {self.name}")
        if self.role:
            parts.append(f"Role: {self.role}")
        if self.traits:
            parts.append(f"Traits: {', '.join(self.traits)}")
        if self.preferences:
            parts.append(f"Preferences: {', '.join(self.preferences[:5])}")
        if self.recurring_topics:
            parts.append(f"Topics: {', '.join(self.recurring_topics[:5])}")
        if self.summary:
            parts.append(self.summary)
        return "\n".join(parts)


@dataclass
class RecallResult:
    """Result from memory recall (vector search)."""

    memories: list[L1Memory] = field(default_factory=list)
    persona: Persona | None = None
    scenes: list[Scene] = field(default_factory=list)
    elapsed_ms: float = 0
    strategy: str = "vector"  # vector | hybrid | keyword

    def format_context(self, max_total_chars: int = 2000) -> str:
        """Format recall results as a single context string for prompt injection."""
        parts: list[str] = []

        if self.persona and self.persona.summary:
            parts.append(self.persona.to_injection())
            parts.append("")

        if self.scenes:
            parts.append("## Relevant Context (from past sessions)")
            for scene in self.scenes[:3]:
                parts.append(scene.to_injection())
            parts.append("")

        if self.memories:
            parts.append("## Past Observations")
            for mem in self.memories[: self._estimate_slots(max_total_chars, parts)]:
                parts.append(f"- {mem.to_injection(500)}")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _estimate_slots(max_chars: int, parts: list[str]) -> int:
        used = sum(len(p) for p in parts)
        remaining = max_chars - used
        if remaining <= 0:
            return 0
        return max(1, remaining // 500)
