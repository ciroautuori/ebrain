"""EBrain KnowledgeGraph — typed entities/edges on PostgreSQL.

Entity kinds: person, company, project, concept, document, tool, brand, client, product, infra.
Edge kinds: works_at, owns, relates_to, depends_on, implements, deprecated_by, manages, produces, runs.
Features: BFS shortest path, neighbor queries, auto-linking, tag-based search.
"""

from __future__ import annotations

import json
from typing import Any

from ebrain.db import Entity
from ebrain.db import execute
from ebrain.db import fetch
from ebrain.db import fetchone

_ENTITY_KINDS = frozenset({
    "person", "company", "project", "concept", "document",
    "tool", "brand", "client", "product", "infra",
})
_EDGE_KINDS = frozenset({
    "works_at", "owns", "relates_to", "depends_on",
    "implements", "deprecated_by", "manages", "produces", "runs",
})


class KnowledgeGraph:
    """PostgreSQL-backed knowledge graph with typed entities and edges."""

    async def add_entity(  # noqa: E501
        self, eid: str, name: str, *, kind: str = "concept", tags: list[str] | None = None, metadata: dict | None = None,
    ) -> Entity:
        if kind not in _ENTITY_KINDS:
            raise ValueError(f"unknown entity kind: {kind} (allowed: {sorted(_ENTITY_KINDS)})")
        await execute(
            """INSERT INTO entities (id, name, kind, tags, metadata)
               VALUES ($1, $2, $3, $4, $5)
               ON CONFLICT(id) DO UPDATE SET name=$2, kind=$3, tags=$4, metadata=$5""",
            eid, name, kind, json.dumps(tags or []), json.dumps(metadata or {}),
        )
        return Entity(id=eid, name=name, kind=kind, tags=tags or [])

    async def get_entity(self, eid: str) -> Entity | None:
        row = await fetchone("SELECT * FROM entities WHERE id = $1", eid)
        if not row:
            return None
        return Entity(
            id=row["id"], name=row["name"], kind=row["kind"],
            tags=json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"] or [],
            created_at=str(row["created_at"]),
        )

    async def list_entities(self, kind: str | None = None, limit: int = 100) -> list[Entity]:
        if kind and kind not in _ENTITY_KINDS:
            raise ValueError(f"unknown entity kind: {kind}")
        if kind:
            rows = await fetch("SELECT * FROM entities WHERE kind = $1 ORDER BY name LIMIT $2", kind, limit)
        else:
            rows = await fetch("SELECT * FROM entities ORDER BY kind, name LIMIT $1", limit)
        return [
            Entity(
                id=r["id"], name=r["name"], kind=r["kind"],
                tags=json.loads(r["tags"]) if isinstance(r["tags"], str) else r["tags"] or [],
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]

    async def search_entities(self, query: str, limit: int = 10) -> list[Entity]:
        pattern = f"%{query}%"
        rows = await fetch(  # noqa: E501
            """SELECT * FROM entities
               WHERE name ILIKE $1 OR id ILIKE $1
               ORDER BY name LIMIT $2""",
            pattern, limit,
        )
        return [
            Entity(
                id=r["id"], name=r["name"], kind=r["kind"],
                tags=json.loads(r["tags"]) if isinstance(r["tags"], str) else r["tags"] or [],
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]

    async def count_entities(self, kind: str | None = None) -> int:
        if kind:
            row = await fetchone("SELECT COUNT(*) as cnt FROM entities WHERE kind = $1", kind)
        else:
            row = await fetchone("SELECT COUNT(*) as cnt FROM entities")
        return row["cnt"] if row else 0

    # ── Edges ──────────────────────────────────────────────────────────

    async def add_edge(
        self, source_id: str, target_id: str, *, kind: str = "relates_to", weight: float = 1.0,
    ) -> None:
        if kind not in _EDGE_KINDS:
            raise ValueError(f"unknown edge kind: {kind} (allowed: {sorted(_EDGE_KINDS)})")
        await execute(
            """INSERT INTO edges (source_id, target_id, kind, weight)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT DO NOTHING""",
            source_id, target_id, kind, weight,
        )

    async def get_neighbors(
        self, entity_id: str, *, edge_kind: str | None = None, direction: str = "both",
    ) -> list[dict[str, Any]]:
        params: list[Any] = [entity_id]
        conditions = ["(e.source_id = $1 OR e.target_id = $1)"]
        if edge_kind:
            params.append(edge_kind)
            conditions.append(f"e.kind = ${len(params)}")

        where = " AND ".join(conditions)
        rows = await fetch(  # noqa: E501
            f"""SELECT e.*,  # noqa: W291
                       src.name as source_name, src.kind as source_kind,
                       tgt.name as target_name, tgt.kind as target_kind
                FROM edges e
                JOIN entities src ON e.source_id = src.id
                JOIN entities tgt ON e.target_id = tgt.id
                WHERE {where}
                ORDER BY e.weight DESC
                LIMIT 50""",
            *params,
        )
        return [
            {
                "edge_id": r["id"], "kind": r["kind"], "weight": float(r["weight"]),
                "source": {"id": r["source_id"], "name": r["source_name"], "kind": r["source_kind"]},
                "target": {"id": r["target_id"], "name": r["target_name"], "kind": r["target_kind"]},
            }
            for r in rows
        ]

    async def shortest_path(self, from_id: str, to_id: str, max_depth: int = 5) -> list[str] | None:
        """BFS shortest path between two entities. Returns list of entity IDs or None."""
        if from_id == to_id:
            return [from_id]

        visited: set[str] = {from_id}
        queue: list[tuple[str, list[str]]] = [(from_id, [from_id])]

        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                continue

            rows = await fetch(
                """SELECT source_id, target_id FROM edges
                   WHERE source_id = $1 OR target_id = $1""",
                current,
            )
            for row in rows:
                neighbor = row["target_id"] if row["source_id"] == current else row["source_id"]
                if neighbor == to_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    async def stats(self) -> dict[str, Any]:
        entities = await fetchone("SELECT COUNT(*) as cnt FROM entities")
        edges = await fetchone("SELECT COUNT(*) as cnt FROM edges")
        return {
            "total_entities": entities["cnt"] if entities else 0,
            "total_edges": edges["cnt"] if edges else 0,
        }

    async def auto_link(self, entity_id: str, text: str) -> list[str]:
        """Find other entities mentioned in text and create edges automatically."""
        all_entities = await self.list_entities(limit=500)
        linked: list[str] = []
        text_lower = text.lower()

        for e in all_entities:
            if e.id == entity_id:
                continue
            if e.name.lower() in text_lower or e.id.lower() in text_lower:
                await self.add_edge(entity_id, e.id, kind="relates_to")
                linked.append(e.id)

        return linked
