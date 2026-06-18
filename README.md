# EBrain — Agent Memory & Knowledge Graph

**Standalone, portable, zero cloud lock-in.** PostgreSQL + Qdrant, Python 3.12+.

## What is EBrain?

A 4-layer memory system for AI agents inspired by TencentDB Agent Memory, plus a typed knowledge graph. Designed to be embedded in any agent runtime — not just EROS.

```
L0: Conversation Recording  →  PostgreSQL (asyncpg)
L1: Memory Extraction       →  LLM + vector dedup (Qdrant)
L2: Scene Profiling         →  Thematic clusters from L1
L3: Persona Generation      →  Long-term agent identity
```

Plus:
- **KnowledgeGraph** — typed entities (Person, Company, Tool, Concept) + edges with BFS traversal
- **Dream Cycle** — gap analysis + automatic web enrichment
- **GBrain Bridge** — MCP stdio sidecar for external brain access
- **Vector Search** — Qdrant + fastembed, $0 local ONNX

## Quick Start

```bash
pip install ebrain

# Or with dev deps
pip install "ebrain[brain,dev]"
```

```python
from ebrain import MemoryPipeline, KnowledgeGraph

# Memory pipeline
pipeline = MemoryPipeline()
await pipeline.record("my-session", "user", "I prefer dark mode")
await pipeline.maybe_extract("my-session")

# Knowledge graph
graph = KnowledgeGraph()
await graph.add_entity("claude", "Claude Code", kind="tool", tags=["ai", "cli"])
await graph.add_entity("vps1", "Server VPS 1", kind="infra")
await graph.add_edge("vps1", "claude", "runs")

# Recall at session start
from ebrain import recall
result = await recall("deploy the chatbot", "my-session")
print(result.format_context())
```

## Architecture

```
ebrain/
├── db.py              # asyncpg connection pool (PG 17+)
├── graph_store.py     # KnowledgeGraph — Entity/Edge/BFS
├── entities.py        # Entity extraction from text
├── dream.py           # Gap analysis + web enrichment
├── synthesize.py      # Knowledge synthesis from fragments
├── memory/            # 4-layer memory pipeline
│   ├── pipeline.py    # L0→L1→L2→L3 orchestrator
│   ├── l0_recorder.py # Conversation recording
│   ├── l1_extractor.py# LLM-powered memory extraction
│   ├── l2l3.py        # Scene + persona builders
│   ├── recall.py      # Qdrant vector search
│   ├── offload.py     # Symbolic tool output compression
│   ├── config.py      # Configuration
│   └── types.py       # Data types
└── cli.py             # ebrain CLI
```

## Environment

| Variable | Default | Description |
|---|---|---|
| `EBRAIN_DATABASE_URL` | `postgresql://eros:eros_dev_2026@127.0.0.1:5433/eros` | PostgreSQL connection |
| `EBRAIN_QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant server |
| `EBRAIN_MEMORY_L1_ENABLED` | `true` | Enable L1 extraction |

## License

MIT — use it anywhere, for any agent.
