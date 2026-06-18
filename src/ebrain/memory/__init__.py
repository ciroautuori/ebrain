"""EROS Memory — 4-layer agent memory system (TencentDB-inspired).

L0: Conversation recording → raw conversation logs (PG).
L1: Memory extraction → structured facts via LLM + vector dedup (Qdrant).
L2: Scene profiling → grouped knowledge from L1 memories.
L3: Persona generation → long-term agent identity from profiles.

Usage:
    from ebrain.memory import MemoryPipeline

    pipeline = MemoryPipeline()
    await pipeline.record_conversation(session_id, messages)
    await pipeline.extract_memories(session_id)  # L1
    await pipeline.build_profile(session_id)     # L2
    persona = await pipeline.get_persona(session_id)  # L3

Auto-recall at session start:
    memories = await pipeline.recall("user query about tool X")
"""

from .config import MemoryConfig
from .pipeline import MemoryPipeline
from .types import L1Memory
from .types import Persona
from .types import RecallResult
from .types import Scene

__all__ = [
    "MemoryPipeline",
    "MemoryConfig",
    "L1Memory",
    "Scene",
    "Persona",
    "RecallResult",
]
