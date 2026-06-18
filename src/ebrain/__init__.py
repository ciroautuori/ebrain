"""EBrain — standalone agent memory + knowledge graph system.

Portable across agent frameworks. Zero cloud lock-in.
PostgreSQL + Qdrant, Python 3.12+.

Main components:
- KnowledgeGraph: typed entities/edges with BFS traversal
- MemoryPipeline: 4-layer memory (L0→L1→L2→L3)
- recall: auto-recall relevant context at session start
- EntityExtractor: extract entities from text
- Dream cycle: gap analysis + web enrichment
"""

from ebrain.db import Edge
from ebrain.db import Entity
from ebrain.entities import EntityExtractor
from ebrain.graph_store import KnowledgeGraph
from ebrain.llm import ask_json
from ebrain.llm import set_ask_json
from ebrain.llm import set_default_model
from ebrain.memory.config import MemoryConfig
from ebrain.memory.pipeline import MemoryPipeline
from ebrain.memory.pipeline import get_pipeline
from ebrain.memory.recall import recall
from ebrain.memory.types import L1Memory
from ebrain.memory.types import Persona
from ebrain.memory.types import RecallResult
from ebrain.memory.types import Scene

__version__ = "0.1.0"

__all__ = [
    "KnowledgeGraph",
    "MemoryPipeline",
    "MemoryConfig",
    "get_pipeline",
    "recall",
    "Entity",
    "Edge",
    "L1Memory",
    "Scene",
    "Persona",
    "RecallResult",
    "EntityExtractor",
    "set_ask_json",
    "set_default_model",
    "ask_json",
]
