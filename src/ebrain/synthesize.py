"""EBrain Synthesize — combine knowledge fragments into coherent summaries.

Takes entity data + edges + L1 memories and produces a knowledge synthesis
that can be used as context for agents or stored as derived knowledge.
"""

from __future__ import annotations

import logging
from typing import Any

from ebrain.graph_store import KnowledgeGraph
from ebrain.llm import ask_json
from ebrain.llm import is_configured

_log = logging.getLogger("ebrain.synthesize")

SYNTHESIZE_PROMPT = """Synthesize the following knowledge fragments into a coherent summary.

Topic: {topic}

Entities:
{entities}

Relationships:
{edges}

Additional context:
{context}

Produce a concise 2-4 paragraph synthesis. Include:
1. Core concepts and their relationships
2. Key facts or data points
3. Any patterns or themes that emerge

Return JSON:
{{
  "title": "Synthesis title",
  "summary": "2-4 paragraph synthesis",
  "key_points": ["point 1", "point 2", "point 3"],
  "confidence": 0.0-1.0
}}
"""


async def synthesize(
    graph: KnowledgeGraph,
    topic: str,
    *,
    context: str = "",
    entity_limit: int = 20,
) -> dict[str, Any]:
    """Synthesize knowledge about a topic from the graph.

    Gathers related entities, their edges, and any provided context,
    then produces a coherent summary via LLM.
    """
    if not is_configured():
        return {"error": "LLM not configured. Call ebrain.llm.set_ask_json() first."}

    # Find relevant entities
    search_results = await graph.search_entities(topic, limit=entity_limit)
    if not search_results:
        return {"status": "empty", "message": f"No entities found for '{topic}'"}

    entities_str = "\n".join(
        f"- [{e.kind}] {e.name} (id: {e.id})" for e in search_results
    )

    # Get edges between found entities
    edge_lines: list[str] = []
    for e in search_results:
        neighbors = await graph.get_neighbors(e.id, direction="both")
        for n in neighbors[:3]:
            edge_lines.append(
                f"- {n['source']['name']} --[{n['kind']}]--> {n['target']['name']}"
            )

    try:
        result = await ask_json(
            SYNTHESIZE_PROMPT.format(
                topic=topic,
                entities=entities_str,
                edges="\n".join(edge_lines) or "(no edges)",
                context=context or "(no additional context)",
            )
        )
    except Exception as exc:
        _log.warning("synthesis failed: %s", exc)
        return {"status": "error", "error": str(exc)}

    if not isinstance(result, dict):
        return {"status": "error", "error": "unexpected LLM response format"}

    return {
        "status": "ok",
        "topic": topic,
        "entity_count": len(search_results),
        **result,
    }
