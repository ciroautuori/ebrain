<div align="center">

<img src="https://img.shields.io/badge/python-3.12+-blue" alt="Python 3.12+">
<img src="https://img.shields.io/badge/postgresql-17+-336791" alt="PostgreSQL 17+">
<img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
<img src="https://img.shields.io/badge/tests-61%2F61-brightgreen" alt="Tests">
<img src="https://img.shields.io/badge/ruff-clean-purple" alt="Ruff">

</div>

# EBrain — Agent Memory & Knowledge Graph

**Give your AI agents a memory that actually works.**

EBrain is a 4-layer memory system for AI agents, plus a typed knowledge graph.
It records conversations, extracts structured knowledge, builds profiles,
and recalls relevant context — all locally, zero cloud lock-in.

Inspired by the [TencentDB Agent Memory](https://github.com/TencentCloud/tencentdb-agent-memory)
architecture (L0→L1→L2→L3 pipeline), reimplemented in Python with PostgreSQL + Qdrant.

---

## Why EBrain?

| Without EBrain | With EBrain |
|---|---|
| Agents forget every session | Persistent memory across sessions |
| Context window overload | Smart recall — only relevant memories injected |
| No user profile | Automatic persona building from conversations |
| Flat vector search | 4-layer structured memory (facts → scenes → persona) |
| Vendor lock-in | PostgreSQL + Qdrant, both open-source, both local |

**Benchmark results** (from TencentDB Agent Memory paper, which EBrain's architecture replicates):
- **-61% token usage** on WideSearch benchmark
- **+51% pass rate** (relative) on agent tasks
- **+59% accuracy** on PersonaMem long-term recall

---

## Quick Start

### 1. Prerequisites

```bash
# PostgreSQL 17+ with pgvector
docker run -d --name ebrain-pg \
  -e POSTGRES_USER=ebrain -e POSTGRES_PASSWORD=ebrain \
  -p 5433:5432 pgvector/pgvector:pg17

# Qdrant vector database
docker run -d --name ebrain-qdrant -p 6333:6333 qdrant/qdrant:v1.12
```

### 2. Install

```bash
pip install git+https://github.com/ciroautuori/ebrain.git
```

### 3. Use

```python
from ebrain import set_ask_json, MemoryPipeline, KnowledgeGraph, recall

# Inject your LLM (Claude, OpenAI, local — any provider)
async def my_llm(prompt: str) -> dict:
    """Your LLM call here. Must return a dict."""
    ...

set_ask_json(my_llm)

# Record conversations
pipeline = MemoryPipeline()
await pipeline.record("my-agent", "user", "I prefer dark mode everywhere")
await pipeline.record("my-agent", "user", "The API should return JSON:API format")

# Extract memories (L1) + index in Qdrant for vector recall
result = await pipeline.run_pipeline("my-agent")
print(f"Extracted {result['l1_extracted']} memories")

# Recall at session start — vector search (Qdrant) with keyword fallback
ctx = await recall("What are the user's preferences?", "my-agent")
print(ctx.format_context())
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    EBRAIN                           │
│                                                     │
│  Conversation  ──→  L0 Recorder  ──→  PostgreSQL    │
│       │                                             │
│       ▼                                             │
│  L1 Extractor  ──→  LLM Extraction  ──→  PG + Qdrant│
│       │              + Vector Dedup                 │
│       ▼                                             │
│  L2 Scene Builder  ──→  Thematic Clusters           │
│       │                                             │
│       ▼                                             │
│  L3 Persona Generator  ──→  Long-term Profile       │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ KnowledgeGraph: Entity/Edge/BFS/AutoLink     │   │
│  │ Dream Cycle: Gap Analysis + Web Enrichment   │   │
│  │ Offload: Symbolic Context Compression        │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Layer-by-layer

| Layer | What | Storage |
|---|---|---|
| **L0** | Raw conversation recording (turns, tool calls, metadata) | PostgreSQL `ebrain_memory_l0_conversations` |
| **L1** | Structured memory extraction (facts, preferences, decisions) with vector dedup via Qdrant | PostgreSQL `ebrain_memory_l1_extractions` + Qdrant `ebrain_memories` |
| **L2** | Scene/profile building — groups related L1 memories into coherent thematic clusters | PostgreSQL `ebrain_memory_l2_scenes` |
| **L3** | Persona generation — synthesizes a long-term agent/user identity from accumulated knowledge | PostgreSQL `ebrain_memory_l3_personas` |

### Knowledge Graph

Typed entities (person, company, tool, client, project, concept, infra, platform, framework)
connected by typed edges (works_at, owns, depends_on, implements, manages, produces, runs).

- **BFS shortest path** between any two entities
- **Auto-linking** from conversation text
- **Neighbor queries** with direction filtering

---

## API Reference

### `ebrain.MemoryPipeline`

```python
pipeline = MemoryPipeline(config=MemoryConfig(...))

# L0: Record a turn
await pipeline.record(session_id, role, content, turn_number=0)

# L1: Trigger extraction if threshold met (extracts + indexes in Qdrant)
memories = await pipeline.maybe_extract(session_id)

# L2: Build scenes from memories
scenes = await pipeline.maybe_build_profile(session_id)

# L3: Generate persona
persona = await pipeline.maybe_generate_persona(session_id)

# Full pipeline
result = await pipeline.run_pipeline(session_id)
# {"l1_extracted": 3, "l2_scenes": 1, "l3_persona": True}
```

### `ebrain.KnowledgeGraph`

```python
graph = KnowledgeGraph()

await graph.add_entity("pg", "PostgreSQL", kind="tool", tags=["database", "sql"])
await graph.add_edge("pg", "myapp", kind="depends_on")
path = await graph.shortest_path("pg", "client-x")  # BFS
neighbors = await graph.get_neighbors("pg")
stats = await graph.stats()
```

### `ebrain.recall()`

```python
result = await recall("What database does the app use?", session_id)
# Returns: RecallResult with .memories, .persona, .scenes
# Primary: Qdrant vector search. Fallback: keyword search.
context = result.format_context(max_total_chars=2000)
```

### `ebrain.set_ask_json()`

```python
# Inject any LLM provider
async def claude_llm(prompt: str) -> dict:
    # Call Claude, OpenAI, Ollama, etc.
    return {"key": "value"}

set_ask_json(claude_llm)
```

### CLI

```bash
ebrain init          # Create database schema
ebrain stats         # Show graph statistics
ebrain search "tool" # Search entities
ebrain dream         # Run gap analysis
ebrain recall "query" session-id  # Recall memories
ebrain memory session-id          # List memories
```

---

## Configuration

All env vars use the `EBRAIN_*` prefix.

| Environment Variable | Default | Description |
|---|---|---|
| `EBRAIN_DATABASE_URL` | `postgresql://ebrain:ebrain@127.0.0.1:5433/ebrain` | PostgreSQL connection |
| `EBRAIN_QDRANT_HOST` | `127.0.0.1` | Qdrant host |
| `EBRAIN_QDRANT_PORT` | `6333` | Qdrant port |
| `EBRAIN_MEMORY_L0_ENABLED` | `true` | Enable L0 recording |
| `EBRAIN_MEMORY_L1_ENABLED` | `true` | Enable L1 extraction |
| `EBRAIN_MEMORY_L1_EVERY_N` | `5` | Trigger L1 after N conversations |
| `EBRAIN_MEMORY_L1_MAX_PER_RUN` | `20` | Max memories per extraction |
| `EBRAIN_MEMORY_L1_DEDUP_THRESHOLD` | `0.85` | Cosine similarity for vector dedup |
| `EBRAIN_MEMORY_L2_ENABLED` | `true` | Enable L2 scene building |
| `EBRAIN_MEMORY_L2_EVERY_N` | `50` | Trigger L2 after N memories |
| `EBRAIN_MEMORY_L3_ENABLED` | `true` | Enable L3 persona generation |
| `EBRAIN_MEMORY_RECALL_ENABLED` | `true` | Enable auto-recall |
| `EBRAIN_MEMORY_RECALL_MAX_RESULTS` | `5` | Max memories per recall |
| `EBRAIN_MEMORY_RECALL_SCORE_THRESHOLD` | `0.3` | Min similarity for recall |

---

## Testing

```bash
pip install -e ".[dev]"

# Unit tests (no services required)
pytest tests/test_ebrain.py -v

# Integration tests (requires PostgreSQL + Qdrant)
pytest tests/test_integration.py -v

# Full suite
pytest tests/ -v
```

Integration tests run against real services — no mocks. 61 tests total.

---

## License

MIT — use it anywhere, for any agent. Fork it, ship it, build on it.

---

<div align="center">

**Agents remember. Humans innovate.**

[GitHub](https://github.com/ciroautuori/ebrain) · [Issues](https://github.com/ciroautuori/ebrain/issues) · [MIT License](./LICENSE)

</div>
