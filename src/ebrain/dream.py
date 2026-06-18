"""EBrain Dream Cycle — gap analysis and knowledge enrichment.

Analyzes the knowledge graph for gaps (missing entities, weak edges,
unconnected clusters) and enriches via web search or LLM synthesis.

Run periodically to keep the brain healthy and growing.
"""

from __future__ import annotations

import logging
from typing import Any

from ebrain.db import fetch
from ebrain.graph_store import KnowledgeGraph
from ebrain.llm import ask_json
from ebrain.llm import is_configured

_log = logging.getLogger("ebrain.dream")

GAP_ANALYSIS_PROMPT = """Analyze this knowledge graph snapshot and identify gaps.

Entities ({total_entities}):
{entities_sample}

Edges ({total_edges}):
{edges_sample}

Find:
1. Isolated entities (no connections) — these need linking
2. Thin areas (kinds with few entities)
3. Missing connections between related entities
4. Suggested new entities based on existing patterns

Return JSON:
{{
  "isolated": [{{"id": "...", "name": "...", "suggestion": "..."}}],
  "thin_areas": [{{"kind": "...", "count": 0, "suggestion": "..."}}],
  "missing_edges": [{{"from": "...", "to": "...", "kind": "relates_to", "reason": "..."}}],
  "suggested_entities": [{{"name": "...", "kind": "tool", "reason": "..."}}],
  "summary": "2-3 sentence gap analysis summary"
}}"""

WEB_ENRICH_PROMPT = """Research the following topic and extract key facts for the knowledge graph.

Topic: {query}

Extract:
- Key entities (people, companies, tools, concepts)
- Relationships between them
- Important facts or data points

Return JSON:
{{
  "entities": [{{"name": "...", "kind": "tool", "facts": "..."}}],
  "edges": [{{"from": "...", "to": "...", "kind": "relates_to", "reason": "..."}}],
  "summary": "Brief research summary"
}}"""


async def analyze_gaps(graph: KnowledgeGraph) -> dict[str, Any]:
    """Run gap analysis on the knowledge graph."""
    if not is_configured():
        return {"error": "LLM not configured. Call ebrain.llm.set_ask_json() first."}

    stats = await graph.stats()
    if stats["total_entities"] < 5:
        return {"status": "skip", "reason": "not enough entities for gap analysis", "stats": stats}

    # Sample entities and edges for analysis
    entities = await graph.list_entities(limit=20)
    entities_sample = "\n".join(
        f"- [{e.kind}] {e.name} (id: {e.id})" for e in entities
    )

    # Get sample edges
    edges_rows = await fetch(
        """SELECT e.kind, src.name as source, tgt.name as target
           FROM edges e
           JOIN entities src ON e.source_id = src.id
           JOIN entities tgt ON e.target_id = tgt.id
           LIMIT 30"""
    )
    edges_sample = "\n".join(
        f"- {r['source']} --[{r['kind']}]--> {r['target']}"
        for r in edges_rows
    )

    try:
        result = await ask_json(
            GAP_ANALYSIS_PROMPT.format(
                total_entities=stats["total_entities"],
                entities_sample=entities_sample,
                total_edges=stats["total_edges"],
                edges_sample=edges_sample or "(no edges yet)",
            )
        )
    except Exception as exc:
        _log.warning("gap analysis failed: %s", exc)
        return {"status": "error", "error": str(exc), "stats": stats}

    return {"status": "ok", "stats": stats, **result}


async def enrich_topic(graph: KnowledgeGraph, query: str) -> dict[str, Any]:
    """Research a topic and enrich the knowledge graph."""
    if not is_configured():
        return {"error": "LLM not configured."}

    try:
        result = await ask_json(WEB_ENRICH_PROMPT.format(query=query))
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    if not isinstance(result, dict):
        return {"status": "error", "error": "unexpected LLM response format"}

    added_entities = 0
    added_edges = 0

    for e in result.get("entities", []):
        if not isinstance(e, dict):
            continue
        eid = e.get("name", "").lower().replace(" ", "-")
        if eid:
            await graph.add_entity(eid, e["name"], kind=e.get("kind", "concept"))
            added_entities += 1

    for edge in result.get("edges", []):
        if not isinstance(edge, dict):
            continue
        try:
            await graph.add_edge(edge["from"], edge["to"], kind=edge.get("kind", "relates_to"))
            added_edges += 1
        except Exception:
            pass

    return {
        "status": "ok",
        "query": query,
        "added_entities": added_entities,
        "added_edges": added_edges,
        "summary": result.get("summary", ""),
    }


async def dream_cycle(graph: KnowledgeGraph) -> dict[str, Any]:
    """Run a full dream cycle: gap analysis + enrichment for top gaps."""
    gaps = await analyze_gaps(graph)

    if gaps.get("status") != "ok":
        return gaps

    enrichments = []
    for suggestion in gaps.get("suggested_entities", [])[:3]:
        if isinstance(suggestion, dict) and suggestion.get("name"):
            result = await enrich_topic(graph, suggestion["name"])
            enrichments.append(result)

    gaps["enrichments"] = enrichments
    return gaps
