"""EBrain VaultIngest — LLM-powered source document ingestion into wiki.

Implements the Karpathy "ingest" operation:
1. Read source document (markdown, text, PDF-converted text)
2. LLM extracts: summary, entities, key facts, related concepts
3. Write wiki page (derived/summary)
4. Upsert entity pages in vault graph/
5. Update index.md and log.md

This is the core Karpathy pattern — the wiki is a persistent, compounding
artifact that accumulates knowledge from every source you add.

Usage:
    from ebrain.vault_ingest import ingest_source
    from ebrain.vault import VaultSync

    vault = VaultSync("/path/to/vault")
    result = await ingest_source(Path("paper.md"), vault)
    print(result["entities_added"], result["wiki_page"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ebrain.vault import VaultSync

_log = logging.getLogger("ebrain.vault_ingest")

_INGEST_SCHEMA = {
    "summary": "2-3 sentence summary of the source document",
    "key_facts": ["list of key facts extracted from this document"],
    "entities": [
        {
            "id": "lowercase-kebab-id",
            "name": "Entity Name",
            "kind": "concept|person|tool|platform|framework|place|event",
            "tags": ["tag1", "tag2"],
        }
    ],
    "related_concepts": ["concept1", "concept2"],
    "contradictions": ["any contradictions with known facts (empty list if none)"],
}

_INGEST_PROMPT_TMPL = """\
You are a knowledge base curator. Analyze the following source document and extract structured knowledge.

SOURCE TITLE: {title}
SOURCE PATH: {path}

SOURCE CONTENT:
{content}

Extract and return JSON matching this exact schema:
{schema}

Rules:
- summary: 2-3 sentences capturing the core insight
- key_facts: concrete facts, not opinions (max 10)
- entities: named things worth tracking (people, tools, concepts, platforms)
- entity id: lowercase kebab-case, globally unique identifier
- entity kind: one of concept|person|tool|platform|framework|place|event
- related_concepts: broader themes this document relates to
- contradictions: note if any claim contradicts common knowledge
"""


async def ingest_source(
    source_path: Path,
    vault: "VaultSync",
    *,
    max_content_chars: int = 8000,
) -> dict:
    """Read source document, extract knowledge via LLM, write to vault wiki.

    Returns dict with: wiki_page, entities_added, key_facts_count, summary
    """
    from ebrain.db import Entity
    from ebrain.llm import ask_json

    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    content = source_path.read_text(encoding="utf-8", errors="replace")
    if len(content) > max_content_chars:
        content = content[:max_content_chars] + "\n\n[... truncated ...]"

    title = source_path.stem.replace("-", " ").replace("_", " ").title()
    import json
    prompt = _INGEST_PROMPT_TMPL.format(
        title=title,
        path=str(source_path),
        content=content,
        schema=json.dumps(_INGEST_SCHEMA, indent=2),
    )

    extracted = await ask_json(prompt)

    summary = str(extracted.get("summary", ""))
    key_facts: list[str] = extracted.get("key_facts", [])
    entities_raw: list[dict] = extracted.get("entities", [])
    related: list[str] = extracted.get("related_concepts", [])
    contradictions: list[str] = extracted.get("contradictions", [])

    # Write wiki summary page for this source
    from ebrain.vault import _frontmatter
    from ebrain.vault import _now_iso
    from ebrain.vault import _slug
    wiki_dir = vault._ensure("wiki")
    fm = _frontmatter({
        "title": title,
        "source": str(source_path),
        "ingested": _now_iso(),
        "entities": [e.get("id", "") for e in entities_raw],
        "related": related,
        "ebrain_type": "wiki_source",
    })
    fact_lines = "\n".join(f"- {f}" for f in key_facts) if key_facts else "_none extracted_"
    contra_lines = "\n".join(f"- {c}" for c in contradictions) if contradictions else "_none_"
    entity_links = " ".join(f"[[{e.get('id', '')}]]" for e in entities_raw if e.get("id"))
    wiki_body = (
        f"{fm}\n\n# {title}\n\n{summary}\n\n"
        f"## Key Facts\n{fact_lines}\n\n"
        f"## Contradictions\n{contra_lines}\n\n"
        f"## Entities\n{entity_links}\n"
    )
    wiki_page = wiki_dir / f"{_slug(title)}.md"
    wiki_page.write_text(wiki_body, encoding="utf-8")

    # Upsert entity pages → vault .md AND PostgreSQL knowledge graph
    from ebrain.graph_store import KnowledgeGraph
    kg = KnowledgeGraph()
    entities_added = 0
    for e_raw in entities_raw:
        eid = e_raw.get("id", "").strip()
        ename = e_raw.get("name", "").strip()
        ekind = e_raw.get("kind", "concept")
        etags = e_raw.get("tags", [])
        if not eid or not ename:
            continue
        entity = Entity(id=eid, name=ename, kind=ekind, tags=etags)

        # Upsert to PG KG (add_entity is idempotent via ON CONFLICT DO NOTHING)
        try:
            await kg.add_entity(eid, ename, kind=ekind, tags=etags)
        except Exception:
            pass

        # Upsert to vault (non-destructive)
        entity_path = vault._graph_dir() / f"{eid}.md"
        source_link = f"[[wiki/{_slug(title)}]]"
        if entity_path.exists():
            existing = entity_path.read_text(encoding="utf-8")
            if source_link not in existing:
                with entity_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"\n## Sources\n- {source_link}\n")
        else:
            vault.write_entity(entity)
            with entity_path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n## Sources\n- {source_link}\n")
            entities_added += 1

    vault.update_index()
    vault.append_log(
        "ingest",
        title,
        {
            "source": str(source_path),
            "entities": len(entities_raw),
            "entities_added": entities_added,
            "key_facts": len(key_facts),
        },
    )

    return {
        "wiki_page": str(wiki_page),
        "summary": summary,
        "entities_added": entities_added,
        "entities_found": len(entities_raw),
        "key_facts_count": len(key_facts),
    }


async def ingest_directory(
    source_dir: Path,
    vault: "VaultSync",
    *,
    glob: str = "**/*.md",
) -> list[dict]:
    """Ingest all matching files in a directory sequentially."""
    results = []
    for path in sorted(source_dir.glob(glob)):
        _log.info("ingesting: %s", path)
        try:
            result = await ingest_source(path, vault)
            results.append(result)
        except Exception as exc:
            _log.error("ingest failed for %s: %s", path, exc)
            results.append({"source": str(path), "error": str(exc)})
    return results
