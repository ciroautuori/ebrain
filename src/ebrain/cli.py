"""EBrain CLI — command-line interface for brain operations.

Usage:
    ebrain init                          Initialize database schema
    ebrain stats                         Show knowledge graph statistics
    ebrain search <q>                    Search entities
    ebrain dream                         Run dream cycle (gap analysis + enrichment)
    ebrain recall <q> [<session>]        Recall memories for a query
    ebrain memory <session>              Show memories for a session
    ebrain vault sync --vault PATH       Sync all memories to Obsidian vault
    ebrain vault watch --vault PATH      Watch vault for human edits (live)
    ebrain vault status --vault PATH     Show vault page counts
    ebrain vault lint --vault PATH       Health check: orphans + broken links
    ebrain vault ingest --vault PATH --source FILE   Ingest source doc into wiki
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
    elif cmd == "vault":
        asyncio.run(_cmd_vault(args[1:]))
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


async def _cmd_vault(args: list[str]) -> None:
    import signal

    from ebrain.vault import VaultSync

    def _get_opt(name: str) -> str:
        try:
            idx = args.index(name)
            return args[idx + 1]
        except (ValueError, IndexError):
            return ""

    subcmd = args[0].lower() if args else ""
    vault_path = _get_opt("--vault")

    if not subcmd or subcmd in ("-h", "--help"):
        print("Usage: ebrain vault <sync|watch|status|lint|ingest> --vault PATH [--session SID] [--source FILE]")
        return

    if not vault_path:
        print("Error: --vault PATH is required")
        sys.exit(1)

    vault = VaultSync(vault_path)

    if subcmd == "status":
        counts = vault.status()
        print(f"Vault: {vault.root}")
        for k, v in counts.items():
            print(f"  {k}: {v}")

    elif subcmd == "lint":
        report = vault.lint()
        print(f"Vault: {vault.root}")
        print(f"Total pages: {report['total_pages']}")
        print(f"Health: {report['health']}")
        if report["orphan_pages"]:
            print(f"Orphans ({len(report['orphan_pages'])}):")
            for p in report["orphan_pages"]:
                print(f"  - {p}")
        if report["broken_links"]:
            print(f"Broken links ({len(report['broken_links'])}):")
            for b in report["broken_links"]:
                print(f"  - {b}")
        if report["untyped_pages"]:
            print(f"Untyped pages ({len(report['untyped_pages'])}):")
            for u in report["untyped_pages"]:
                print(f"  - {u}")

    elif subcmd == "sync":
        session_id = _get_opt("--session") or None
        from ebrain.memory.l1_extractor import get_memories
        from ebrain.memory.l2l3 import get_persona
        from ebrain.memory.l2l3 import get_scenes
        from ebrain.migrations import run_migrations
        await run_migrations()
        sessions_to_sync: list[str] = []
        if session_id:
            sessions_to_sync = [session_id]
        else:
            from ebrain.db import fetch
            rows = await fetch(
                "SELECT DISTINCT session_id FROM ebrain_memory_l1_extractions ORDER BY session_id"
            )
            sessions_to_sync = [r["session_id"] for r in rows]

        total_mem = 0
        total_scenes = 0
        total_personas = 0
        for sid in sessions_to_sync:
            mems = await get_memories(sid, limit=1000)
            scenes = await get_scenes(sid, limit=100)
            persona = await get_persona(sid)
            for m in mems:
                vault.write_memory(m)
            for sc in scenes:
                vault.write_scene(sc)
            if persona:
                vault.write_persona(persona)
            total_mem += len(mems)
            total_scenes += len(scenes)
            total_personas += 1 if persona else 0

        vault.update_index()
        vault.append_log("sync", "full", {
            "sessions": len(sessions_to_sync),
            "memories": total_mem,
            "scenes": total_scenes,
            "personas": total_personas,
        })
        print(f"Synced: {total_mem} memories, {total_scenes} scenes, {total_personas} personas")
        print(f"Vault: {vault.root}")

    elif subcmd == "watch":
        from ebrain.vault_watcher import VaultWatcher

        def _on_change(path: object) -> None:
            print(f"changed: {path}")

        print(f"Watching: {vault.vault_path}  (Ctrl+C to stop)")
        watcher = VaultWatcher(vault.vault_path, _on_change)
        watcher.start()
        stop_event = __import__("threading").Event()

        def _sig(*_: object) -> None:
            stop_event.set()

        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
        stop_event.wait()
        watcher.stop()
        print("Stopped.")

    elif subcmd == "ingest":
        source = _get_opt("--source")
        if not source:
            print("Error: --source FILE is required for ingest")
            sys.exit(1)
        from pathlib import Path

        from ebrain.vault_ingest import ingest_source
        result = await ingest_source(Path(source), vault)
        print(f"Ingested: {source}")
        print(f"  Wiki page: {result['wiki_page']}")
        print(f"  Summary: {result['summary'][:120]}")
        print(f"  Entities added: {result['entities_added']} / {result['entities_found']} found")
        print(f"  Key facts: {result['key_facts_count']}")

    else:
        print(f"Unknown vault subcommand: {subcmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
