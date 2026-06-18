"""EBrain CLI — command-line interface for brain operations.

Usage:
    ebrain init          Initialize database schema
    ebrain stats         Show knowledge graph statistics
    ebrain search <q>    Search entities
    ebrain dream         Run dream cycle (gap analysis + enrichment)
    ebrain recall <q>    Recall memories for a query
    ebrain memory <sid>  Show memories for a session
"""

from __future__ import annotations

import asyncio
import sys


def main() -> None:
    args = sys.argv[1:]
    if not args:
        _usage()
        return

    cmd = args[0].lower()

    if cmd == "init":
        asyncio.run(_cmd_init())
    elif cmd == "stats":
        asyncio.run(_cmd_stats())
    elif cmd == "search":
        query = " ".join(args[1:]) if len(args) > 1 else ""
        if not query:
            print("Usage: ebrain search <query>")
            sys.exit(1)
        asyncio.run(_cmd_search(query))
    elif cmd == "dream":
        asyncio.run(_cmd_dream())
    elif cmd == "recall":
        query = " ".join(args[1:]) if len(args) > 1 else ""
        sid = args[2] if len(args) > 2 else "default"
        if not query:
            print("Usage: ebrain recall <query> [session_id]")
            sys.exit(1)
        asyncio.run(_cmd_recall(query, sid))
    elif cmd == "memory":
        sid = args[1] if len(args) > 1 else "default"
        asyncio.run(_cmd_memory(sid))
    elif cmd in ("-h", "--help", "help"):
        _usage()
    else:
        print(f"Unknown command: {cmd}")
        _usage()
        sys.exit(1)


def _usage() -> None:
    print(__doc__)


async def _cmd_init() -> None:
    from ebrain.db import ensure_schema
    from ebrain.memory.l0_recorder import ensure_schema as _l0
    from ebrain.memory.l1_extractor import ensure_schema as _l1
    from ebrain.memory.l2l3 import ensure_schema as _l2l3

    print("Initializing ebrain database...")
    await ensure_schema()
    await _l0()
    await _l1()
    await _l2l3()
    print("OK  Schema ready.")


async def _cmd_stats() -> None:
    from ebrain.graph_store import KnowledgeGraph

    graph = KnowledgeGraph()
    stats = await graph.stats()
    print(f"Entities: {stats['total_entities']}")
    print(f"Edges:    {stats['total_edges']}")


async def _cmd_search(query: str) -> None:
    from ebrain.graph_store import KnowledgeGraph

    graph = KnowledgeGraph()
    results = await graph.search_entities(query, limit=20)
    if not results:
        print(f"No entities found for '{query}'")
        return
    for e in results:
        tags = f" [{', '.join(e.tags)}]" if e.tags else ""
        print(f"  [{e.kind}] {e.name}{tags}")


async def _cmd_dream() -> None:
    from ebrain.dream import dream_cycle
    from ebrain.graph_store import KnowledgeGraph

    graph = KnowledgeGraph()
    print("Running dream cycle...")
    result = await dream_cycle(graph)
    print(f"Status: {result.get('status')}")
    if result.get("summary"):
        print(f"Summary: {result['summary']}")
    if result.get("enrichments"):
        for e in result["enrichments"]:
            print(f"  - {e.get('query')}: +{e.get('added_entities', 0)} entities, +{e.get('added_edges', 0)} edges")


async def _cmd_recall(query: str, session_id: str) -> None:
    from ebrain.memory.config import MemoryConfig
    from ebrain.memory.recall import recall

    cfg = MemoryConfig()
    result = await recall(query, session_id, config=cfg)
    print(f"Strategy: {result.strategy} ({result.elapsed_ms:.0f}ms)")
    print(f"Memories: {len(result.memories)}")
    print(f"Persona:  {'yes' if result.persona else 'no'}")
    print()
    ctx = result.format_context()
    if ctx:
        print(ctx)


async def _cmd_memory(session_id: str) -> None:
    from ebrain.memory.l1_extractor import get_memories

    mems = await get_memories(session_id, limit=50)
    if not mems:
        print(f"No memories for session '{session_id}'")
        return
    for m in mems:
        print(f"[{m.kind}] {m.content[:120]}")


if __name__ == "__main__":
    main()
