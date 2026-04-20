"""GraphStore: SQLite persistence layer using a nodes+edges graph schema.

All tables use the graph model (nodes + edges) as the core structure.
TASK-M1-001 adds 8 concept-first tables alongside the legacy nodes/edges tables.
No ORM — plain sqlite3 from the stdlib.
"""

import json
import logging
import re
import sqlite3
from pathlib import Path

from src.models.concepts import (
    CitationSnapshot,
    Concept,
    ConceptQuery,
    ConceptRelationship,
    ContentPublication,
    PaperConceptLink,
    PaperRecord,
    ResurrectionCandidate,
)
from src.models.graph import Edge, Node, WeeklyRun

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_NODES = """
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    node_type   TEXT NOT NULL,
    label       TEXT NOT NULL,
    properties  TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_EDGES = """
CREATE TABLE IF NOT EXISTS edges (
    source_id         TEXT NOT NULL REFERENCES nodes(id),
    target_id         TEXT NOT NULL REFERENCES nodes(id),
    relationship_type TEXT NOT NULL,
    weight            REAL DEFAULT 1.0,
    properties        TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (source_id, target_id, relationship_type)
);
"""

_CREATE_WEEKLY_RUNS = """
CREATE TABLE IF NOT EXISTS weekly_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date          TEXT NOT NULL,
    started_at        TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at      TEXT,
    papers_fetched    INTEGER DEFAULT 0,
    papers_classified INTEGER DEFAULT 0,
    papers_failed     INTEGER DEFAULT 0,
    digest_path       TEXT,
    status            TEXT NOT NULL DEFAULT 'running',
    error_message     TEXT
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_nodes_type   ON nodes(node_type);",
    "CREATE INDEX IF NOT EXISTS idx_nodes_label  ON nodes(label);",
    "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);",
    "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);",
    "CREATE INDEX IF NOT EXISTS idx_edges_type   ON edges(relationship_type);",
    "CREATE INDEX IF NOT EXISTS idx_runs_date    ON weekly_runs(run_date);",
]

# ---------------------------------------------------------------------------
# DDL — concept-first schema (TASK-M1-001, TDD §2.4.2)
# ---------------------------------------------------------------------------

_CREATE_CONCEPTS = """
CREATE TABLE IF NOT EXISTS concepts (
    id                      INTEGER PRIMARY KEY,
    user_id                 TEXT NOT NULL DEFAULT 'default',
    name                    TEXT NOT NULL,
    slug                    TEXT NOT NULL,
    concept_type            TEXT NOT NULL CHECK(concept_type IN
                            ('Algorithm','Technique','Framework','Concept','Mechanism','Problem','Category')),
    what_it_is              TEXT,
    what_problem_it_solves  TEXT,
    innovation_chain        TEXT,
    limitations             TEXT,
    introduced_year         INTEGER,
    domain_tags             TEXT,
    source                  TEXT,
    source_refs             TEXT,
    content_angles          TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, slug)
);
"""

_CREATE_CONCEPT_RELATIONSHIPS = """
CREATE TABLE IF NOT EXISTS concept_relationships (
    id                      INTEGER PRIMARY KEY,
    user_id                 TEXT NOT NULL DEFAULT 'default',
    source_concept_id       INTEGER NOT NULL REFERENCES concepts(id),
    target_concept_id       INTEGER NOT NULL REFERENCES concepts(id),
    relationship_type       TEXT NOT NULL CHECK(relationship_type IN (
                              'BUILDS_ON',
                              'ADDRESSES',
                              'ALTERNATIVE_TO',
                              'BASELINE_OF',
                              'PREREQUISITE_OF',
                              'INTRODUCES',
                              'BELONGS_TO'
                            )),
    label                   TEXT,
    -- confidence is intentionally unused here; link-level confidence lives on
    -- paper_concept_links (PaperConceptLink.confidence). Reserved for future
    -- use when relationship provenance is tracked at the edge level.
    confidence              TEXT CHECK(confidence IN ('high','medium','low')),
    source                  TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, source_concept_id, target_concept_id, relationship_type),
    CHECK (source_concept_id != target_concept_id)
);
"""

_CREATE_PAPERS = """
CREATE TABLE IF NOT EXISTS papers (
    id                      INTEGER PRIMARY KEY,
    user_id                 TEXT NOT NULL DEFAULT 'default',
    arxiv_id                TEXT NOT NULL,
    title                   TEXT NOT NULL,
    abstract                TEXT,
    authors                 TEXT,
    primary_category        TEXT,
    all_categories          TEXT,
    published_date          TEXT,
    tier                    INTEGER CHECK(tier BETWEEN 1 AND 5),
    tier_confidence         TEXT,
    summary                 TEXT,
    key_contributions       TEXT,
    hf_upvotes              INTEGER DEFAULT 0,
    importance_score        REAL,
    openreview_accepted     INTEGER DEFAULT 0,
    classification_failed   INTEGER DEFAULT 0,
    run_date                TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, arxiv_id)
);
"""

_CREATE_PAPER_CONCEPT_LINKS = """
CREATE TABLE IF NOT EXISTS paper_concept_links (
    paper_id                INTEGER NOT NULL REFERENCES papers(id),
    concept_id              INTEGER NOT NULL REFERENCES concepts(id),
    link_type               TEXT NOT NULL CHECK(link_type IN
                              ('INTRODUCES','USES','SURVEYS','EVALUATES')),
    user_id                 TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (paper_id, concept_id, link_type)
);
"""

_CREATE_CITATION_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS citation_snapshots (
    paper_id                INTEGER NOT NULL REFERENCES papers(id),
    check_date              TEXT NOT NULL,
    citation_count          INTEGER NOT NULL,
    user_id                 TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (paper_id, check_date)
);
"""

_CREATE_HF_MODEL_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS hf_model_snapshots (
    paper_id                INTEGER NOT NULL REFERENCES papers(id),
    check_date              TEXT NOT NULL,
    model_ids               TEXT NOT NULL,
    user_id                 TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (paper_id, check_date)
);
"""

_CREATE_CONCEPT_QUERIES = """
CREATE TABLE IF NOT EXISTS concept_queries (
    id                      INTEGER PRIMARY KEY,
    user_id                 TEXT NOT NULL DEFAULT 'default',
    concept_id              INTEGER NOT NULL REFERENCES concepts(id),
    query_text              TEXT,
    queried_at              TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_CONTENT_PUBLICATIONS = """
CREATE TABLE IF NOT EXISTS content_publications (
    id                      INTEGER PRIMARY KEY,
    user_id                 TEXT NOT NULL DEFAULT 'default',
    channel                 TEXT NOT NULL,
    url                     TEXT,
    concept_ids             TEXT NOT NULL,
    published_at            TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_CONCEPT_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_concepts_name     ON concepts(name);",
    "CREATE INDEX IF NOT EXISTS idx_concepts_source   ON concepts(source);",
    "CREATE INDEX IF NOT EXISTS idx_cr_source         ON concept_relationships(source_concept_id);",
    "CREATE INDEX IF NOT EXISTS idx_cr_target         ON concept_relationships(target_concept_id);",
    "CREATE INDEX IF NOT EXISTS idx_cr_type           ON concept_relationships(relationship_type);",
    "CREATE INDEX IF NOT EXISTS idx_papers_published  ON papers(published_date);",
    "CREATE INDEX IF NOT EXISTS idx_papers_tier       ON papers(tier);",
    "CREATE INDEX IF NOT EXISTS idx_pcl_concept       ON paper_concept_links(concept_id);",
    "CREATE INDEX IF NOT EXISTS idx_cq_concept        ON concept_queries(concept_id);",
    "CREATE INDEX IF NOT EXISTS idx_cq_queried        ON concept_queries(queried_at);",
    "CREATE INDEX IF NOT EXISTS idx_cp_published      ON content_publications(published_at);",
]


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------


class GraphStore:
    """All SQLite read/write operations for the paper-monitoring graph."""

    def __init__(self, db_path: Path | str) -> None:
        """Open or create the SQLite database and run schema migrations."""
        if str(db_path) == ":memory:":
            self._conn = sqlite3.connect(":memory:")
        else:
            db_path = Path(db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path))

        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._create_schema()
        logger.debug("GraphStore opened at %s", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_schema(self) -> None:
        with self._conn:
            # Legacy tables (nodes/edges pipeline)
            self._conn.execute(_CREATE_NODES)
            self._conn.execute(_CREATE_EDGES)
            self._conn.execute(_CREATE_WEEKLY_RUNS)
            for stmt in _CREATE_INDEXES:
                self._conn.execute(stmt)
            # Concept-first tables (TASK-M1-001)
            self._conn.execute(_CREATE_CONCEPTS)
            self._conn.execute(_CREATE_CONCEPT_RELATIONSHIPS)
            self._conn.execute(_CREATE_PAPERS)
            self._conn.execute(_CREATE_PAPER_CONCEPT_LINKS)
            self._conn.execute(_CREATE_CITATION_SNAPSHOTS)
            self._conn.execute(_CREATE_HF_MODEL_SNAPSHOTS)
            self._conn.execute(_CREATE_CONCEPT_QUERIES)
            self._conn.execute(_CREATE_CONTENT_PUBLICATIONS)
            for stmt in _CREATE_CONCEPT_INDEXES:
                self._conn.execute(stmt)

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def upsert_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        properties: dict,
    ) -> None:
        """Insert a new node or update properties + updated_at on conflict."""
        props_json = json.dumps(properties)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO nodes (id, node_type, label, properties)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    properties = excluded.properties,
                    updated_at = datetime('now')
                """,
                (node_id, node_type, label, props_json),
            )

    def get_node(self, node_id: str) -> Node | None:
        """Return a single Node by ID, or None if not found."""
        row = self._conn.execute(
            "SELECT id, node_type, label, properties FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def get_nodes_by_type(self, node_type: str) -> list[Node]:
        """Return all nodes of the given type."""
        rows = self._conn.execute(
            "SELECT id, node_type, label, properties FROM nodes WHERE node_type = ?",
            (node_type,),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_concept_index(self) -> list[str]:
        """Return all concept node labels — injected into the classifier prompt."""
        rows = self._conn.execute(
            "SELECT label FROM nodes WHERE node_type = 'concept' ORDER BY label",
        ).fetchall()
        return [r["label"] for r in rows]

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        properties: dict | None = None,
    ) -> None:
        """Insert or replace an edge on the composite primary key."""
        props_json = json.dumps(properties or {})
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO edges
                    (source_id, target_id, relationship_type, weight, properties)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_id, target_id, relationship_type, weight, props_json),
            )

    def get_edges_from(
        self,
        node_id: str,
        relationship_type: str | None = None,
    ) -> list[Edge]:
        """Return all edges originating from node_id, optionally filtered by type."""
        if relationship_type:
            rows = self._conn.execute(
                """
                SELECT source_id, target_id, relationship_type, weight, properties
                FROM edges
                WHERE source_id = ? AND relationship_type = ?
                """,
                (node_id, relationship_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT source_id, target_id, relationship_type, weight, properties
                FROM edges WHERE source_id = ?
                """,
                (node_id,),
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_edges_to(
        self,
        node_id: str,
        relationship_type: str | None = None,
    ) -> list[Edge]:
        """Return all edges pointing to node_id, optionally filtered by type."""
        if relationship_type:
            rows = self._conn.execute(
                """
                SELECT source_id, target_id, relationship_type, weight, properties
                FROM edges
                WHERE target_id = ? AND relationship_type = ?
                """,
                (node_id, relationship_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT source_id, target_id, relationship_type, weight, properties
                FROM edges WHERE target_id = ?
                """,
                (node_id,),
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    # ------------------------------------------------------------------
    # Weekly run tracking
    # ------------------------------------------------------------------

    def create_run(self, run_date: str) -> int:
        """Create a new weekly run record with status='running'. Returns the run ID.

        Call this at the start of the pipeline. Follow up with update_run() when
        the pipeline completes or fails.
        """
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO weekly_runs (run_date, status) VALUES (?, 'running')",
                (run_date,),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def update_run(
        self,
        run_id: int,
        *,
        status: str,
        papers_fetched: int = 0,
        papers_classified: int = 0,
        papers_failed: int = 0,
        digest_path: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update a weekly run record with final stats and completion status.

        Call this at the end of the pipeline (or on failure) to record outcomes.
        """
        with self._conn:
            self._conn.execute(
                """
                UPDATE weekly_runs
                SET status          = ?,
                    completed_at    = datetime('now'),
                    papers_fetched  = ?,
                    papers_classified = ?,
                    papers_failed   = ?,
                    digest_path     = ?,
                    error_message   = ?
                WHERE id = ?
                """,
                (
                    status,
                    papers_fetched,
                    papers_classified,
                    papers_failed,
                    digest_path,
                    error_message,
                    run_id,
                ),
            )

    def log_run(
        self,
        run_date: str,
        papers_fetched: int,
        papers_classified: int,
        digest_path: str,
        papers_failed: int = 0,
        status: str = "completed",
        error_message: str | None = None,
    ) -> int:
        """Insert a weekly run record and return its auto-generated ID."""
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO weekly_runs
                    (run_date, completed_at, papers_fetched, papers_classified,
                     papers_failed, digest_path, status, error_message)
                VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_date,
                    papers_fetched,
                    papers_classified,
                    papers_failed,
                    digest_path,
                    status,
                    error_message,
                ),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def get_latest_run(self) -> WeeklyRun | None:
        """Return the most recent weekly run record, or None."""
        row = self._conn.execute(
            """
            SELECT id, run_date, started_at, completed_at,
                   papers_fetched, papers_classified, papers_failed,
                   digest_path, status, error_message
            FROM weekly_runs
            ORDER BY started_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return WeeklyRun(**dict(row))

    # ------------------------------------------------------------------
    # Mutation methods for manual graph editing (MVP)
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        properties: dict,
    ) -> None:
        """Create a new node. Raises ValueError if node already exists."""
        existing = self.get_node(node_id)
        if existing is not None:
            raise ValueError(f"Node already exists: {node_id}")
        props_json = json.dumps(properties)
        with self._conn:
            self._conn.execute(
                "INSERT INTO nodes (id, node_type, label, properties) VALUES (?, ?, ?, ?)",
                (node_id, node_type, label, props_json),
            )

    def delete_node(self, node_id: str) -> int:
        """Delete a node and all connected edges (incoming and outgoing).

        Returns the number of edges removed.
        Raises ValueError if the node does not exist.
        """
        existing = self.get_node(node_id)
        if existing is None:
            raise ValueError(f"Node does not exist: {node_id}")
        with self._conn:
            # Count edges before deletion
            row = self._conn.execute(
                "SELECT COUNT(*) FROM edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id),
            ).fetchone()
            edge_count = row[0]
            # Delete edges first (both directions)
            self._conn.execute(
                "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id),
            )
            # Delete the node
            self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        logger.info("Deleted node %s and %d connected edges", node_id, edge_count)
        return edge_count

    def update_node_properties(
        self,
        node_id: str,
        properties_patch: dict,
    ) -> None:
        """Merge a partial dict into the existing node properties JSON.

        Does not overwrite keys not present in the patch.
        Sets updated_at to the current timestamp.
        Raises ValueError if node does not exist.
        """
        row = self._conn.execute(
            "SELECT properties FROM nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Node does not exist: {node_id}")
        existing_props = json.loads(row["properties"]) if row["properties"] else {}
        existing_props.update(properties_patch)
        props_json = json.dumps(existing_props)
        with self._conn:
            self._conn.execute(
                "UPDATE nodes SET properties = ?, updated_at = datetime('now') WHERE id = ?",
                (props_json, node_id),
            )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        properties: dict | None = None,
    ) -> None:
        """Create a new edge. Raises ValueError if edge already exists."""
        existing = self._conn.execute(
            """
            SELECT 1 FROM edges
            WHERE source_id = ? AND target_id = ? AND relationship_type = ?
            """,
            (source_id, target_id, relationship_type),
        ).fetchone()
        if existing is not None:
            raise ValueError(
                f"Edge already exists: {source_id} -[{relationship_type}]-> {target_id}"
            )
        props_json = json.dumps(properties or {})
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO edges (source_id, target_id, relationship_type, weight, properties)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_id, target_id, relationship_type, weight, props_json),
            )

    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
    ) -> None:
        """Delete a specific edge. Raises ValueError if edge does not exist."""
        with self._conn:
            cur = self._conn.execute(
                """
                DELETE FROM edges
                WHERE source_id = ? AND target_id = ? AND relationship_type = ?
                """,
                (source_id, target_id, relationship_type),
            )
            if cur.rowcount == 0:
                raise ValueError(
                    f"Edge does not exist: {source_id} -[{relationship_type}]-> {target_id}"
                )
        logger.info(
            "Deleted edge %s -[%s]-> %s", source_id, relationship_type, target_id
        )

    # ------------------------------------------------------------------
    # Index methods for MVP (technique, problem)
    # ------------------------------------------------------------------

    def get_technique_index(self) -> list[str]:
        """Return all technique node labels — injected into the analyzer prompt."""
        rows = self._conn.execute(
            "SELECT label FROM nodes WHERE node_type = 'technique' ORDER BY label",
        ).fetchall()
        return [r["label"] for r in rows]

    def get_problem_index(self) -> list[str]:
        """Return all problem node labels — injected into the analyzer prompt."""
        rows = self._conn.execute(
            "SELECT label FROM nodes WHERE node_type = 'problem' ORDER BY label",
        ).fetchall()
        return [r["label"] for r in rows]

    # ------------------------------------------------------------------
    # Temporal query methods (MVP — for ChangeDetector)
    # ------------------------------------------------------------------

    def get_nodes_created_since(
        self,
        since_date: str,
        node_type: str | None = None,
    ) -> list[Node]:
        """Return nodes created on or after since_date, optionally filtered by type."""
        if node_type:
            rows = self._conn.execute(
                """
                SELECT id, node_type, label, properties
                FROM nodes
                WHERE created_at >= ? AND node_type = ?
                ORDER BY created_at DESC
                """,
                (since_date, node_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, node_type, label, properties
                FROM nodes
                WHERE created_at >= ?
                ORDER BY created_at DESC
                """,
                (since_date,),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_edges_created_since(
        self,
        since_date: str,
        relationship_type: str | None = None,
    ) -> list[Edge]:
        """Return edges created on or after since_date, optionally filtered by type."""
        if relationship_type:
            rows = self._conn.execute(
                """
                SELECT source_id, target_id, relationship_type, weight, properties
                FROM edges
                WHERE created_at >= ? AND relationship_type = ?
                ORDER BY created_at DESC
                """,
                (since_date, relationship_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT source_id, target_id, relationship_type, weight, properties
                FROM edges
                WHERE created_at >= ?
                ORDER BY created_at DESC
                """,
                (since_date,),
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    # ------------------------------------------------------------------
    # Graph neighborhood (MVP — for visualization)
    # ------------------------------------------------------------------

    def get_node_neighborhood(
        self,
        node_id: str,
        depth: int = 1,
    ) -> dict:
        """Return a node and all nodes/edges within N hops via BFS.

        Returns ``{"nodes": [dict], "edges": [dict]}`` where each node dict
        has keys ``id``, ``node_type``, ``label``, ``properties`` and each
        edge dict has keys ``source_id``, ``target_id``, ``relationship_type``,
        ``weight``, ``properties``.

        Returns empty lists if the starting node does not exist.
        """
        start = self.get_node(node_id)
        if start is None:
            return {"nodes": [], "edges": []}

        visited_nodes: dict[str, dict] = {}
        collected_edges: list[dict] = []
        edge_seen: set[tuple[str, str, str]] = set()

        # BFS frontier
        frontier = {node_id}
        for _ in range(depth + 1):  # depth 0 = just the start node
            for nid in frontier:
                if nid in visited_nodes:
                    continue
                node = self.get_node(nid)
                if node is None:
                    continue
                visited_nodes[nid] = {
                    "id": node.id,
                    "node_type": node.node_type,
                    "label": node.label,
                    "properties": node.properties,
                }
            if _ == depth:
                break
            # Expand frontier via edges (both directions)
            next_frontier: set[str] = set()
            for nid in frontier:
                for edge in self.get_edges_from(nid):
                    key = (edge.source_id, edge.target_id, edge.relationship_type)
                    if key not in edge_seen:
                        edge_seen.add(key)
                        collected_edges.append({
                            "source_id": edge.source_id,
                            "target_id": edge.target_id,
                            "relationship_type": edge.relationship_type,
                            "weight": edge.weight,
                            "properties": edge.properties,
                        })
                    next_frontier.add(edge.target_id)
                for edge in self.get_edges_to(nid):
                    key = (edge.source_id, edge.target_id, edge.relationship_type)
                    if key not in edge_seen:
                        edge_seen.add(key)
                        collected_edges.append({
                            "source_id": edge.source_id,
                            "target_id": edge.target_id,
                            "relationship_type": edge.relationship_type,
                            "weight": edge.weight,
                            "properties": edge.properties,
                        })
                    next_frontier.add(edge.source_id)
            frontier = next_frontier - set(visited_nodes.keys())

        return {
            "nodes": list(visited_nodes.values()),
            "edges": collected_edges,
        }

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def paper_exists(self, arxiv_id: str) -> bool:
        """Return True if a paper node with id 'paper:{arxiv_id}' exists."""
        node_id = f"paper:{arxiv_id}"
        row = self._conn.execute(
            "SELECT 1 FROM nodes WHERE id = ? AND node_type = 'paper'",
            (node_id,),
        ).fetchone()
        return row is not None

    def get_papers_for_digest(self, run_date: str) -> list[dict]:
        """Return paper node data for run_date, each entry including linked concept nodes.

        A paper is included if its ``properties`` JSON contains a ``run_date``
        field equal to the requested date.  Each entry has the shape::

            {
                "id": "paper:...",
                "label": "Paper Title",
                "properties": {...},
                "concepts": [{"id": ..., "label": ..., "properties": {...}}, ...]
            }
        """
        rows = self._conn.execute(
            """
            SELECT id, label, properties
            FROM nodes
            WHERE node_type = 'paper'
              AND json_extract(properties, '$.run_date') = ?
            """,
            (run_date,),
        ).fetchall()

        results = []
        for row in rows:
            paper_id = row["id"]
            props = json.loads(row["properties"]) if row["properties"] else {}

            concept_rows = self._conn.execute(
                """
                SELECT cn.id, cn.label, cn.properties
                FROM edges e
                JOIN nodes cn ON e.target_id = cn.id
                WHERE e.source_id = ?
                  AND e.relationship_type = 'BUILDS_ON'
                  AND cn.node_type = 'concept'
                """,
                (paper_id,),
            ).fetchall()

            concepts = [
                {
                    "id": cr["id"],
                    "label": cr["label"],
                    "properties": json.loads(cr["properties"]) if cr["properties"] else {},
                }
                for cr in concept_rows
            ]

            results.append(
                {
                    "id": paper_id,
                    "label": row["label"],
                    "properties": props,
                    "concepts": concepts,
                }
            )

        return results

    def get_all_papers(self, limit: int = 500) -> list[dict]:
        """Return all paper nodes with parsed properties and linked concepts.

        Sorted by run_date DESC, tier ASC, hf_upvotes DESC.
        Each returned dict has keys: id, label, properties, linked_concepts.
        Returns [] if no paper nodes exist.
        """
        rows = self._conn.execute(
            """
            SELECT id, label, properties
            FROM nodes
            WHERE node_type = 'paper'
            ORDER BY
                json_extract(properties, '$.run_date') DESC,
                json_extract(properties, '$.tier') ASC,
                json_extract(properties, '$.hf_upvotes') DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        results = []
        for r in rows:
            paper_id = r["id"]
            props = json.loads(r["properties"]) if r["properties"] else {}

            # Fetch linked concepts via BUILDS_ON edges
            concept_rows = self._conn.execute(
                """
                SELECT cn.label FROM edges e
                JOIN nodes cn ON e.target_id = cn.id
                WHERE e.source_id = ?
                  AND e.relationship_type = 'BUILDS_ON'
                  AND cn.node_type = 'concept'
                """,
                (paper_id,),
            ).fetchall()

            results.append({
                "id": paper_id,
                "label": r["label"],
                "properties": props,
                "linked_concepts": [cr["label"] for cr in concept_rows],
            })

        return results

    def get_all_concepts(self) -> list[dict]:
        """Return all concept nodes with source papers and prerequisites.

        Each returned dict has keys:
          - id, label, properties (description, domain_tags, seeded_from)
          - source_papers: list of paper labels that INTRODUCES this concept
          - prerequisites: list of concept labels this concept depends on
        """
        rows = self._conn.execute(
            """
            SELECT id, label, properties
            FROM nodes
            WHERE node_type = 'concept'
            ORDER BY label
            """
        ).fetchall()

        results = []
        for row in rows:
            concept_id = row["id"]
            props = json.loads(row["properties"]) if row["properties"] else {}

            # Papers that INTRODUCES this concept (paper → concept)
            source_rows = self._conn.execute(
                """
                SELECT n.label FROM edges e
                JOIN nodes n ON e.source_id = n.id
                WHERE e.target_id = ? AND e.relationship_type = 'INTRODUCES'
                """,
                (concept_id,),
            ).fetchall()

            # Concepts this one depends on (this → prerequisite via PREREQUISITE_OF)
            prereq_rows = self._conn.execute(
                """
                SELECT n.label FROM edges e
                JOIN nodes n ON e.target_id = n.id
                WHERE e.source_id = ? AND e.relationship_type = 'PREREQUISITE_OF'
                """,
                (concept_id,),
            ).fetchall()

            results.append({
                "id": concept_id,
                "label": row["label"],
                "properties": props,
                "source_papers": [r["label"] for r in source_rows],
                "prerequisites": [r["label"] for r in prereq_rows],
            })

        return results

    # ------------------------------------------------------------------
    # Concept CRUD (TASK-M1-001)
    # ------------------------------------------------------------------

    @staticmethod
    def _slugify(name: str) -> str:
        """Normalize a concept name to a URL-safe slug for the UNIQUE constraint."""
        slug = name.lower().strip()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        return slug.strip("-")

    def upsert_concept(self, concept: Concept) -> None:
        """Insert or update a concept row.  On conflict (user_id, slug) all
        mutable fields are updated and updated_at is refreshed."""
        slug = self._slugify(concept.name)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO concepts
                    (user_id, name, slug, concept_type, what_it_is,
                     what_problem_it_solves, innovation_chain, limitations,
                     introduced_year, domain_tags, source, source_refs,
                     content_angles)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, slug) DO UPDATE SET
                    name                   = excluded.name,
                    concept_type           = excluded.concept_type,
                    what_it_is             = excluded.what_it_is,
                    what_problem_it_solves = excluded.what_problem_it_solves,
                    innovation_chain       = excluded.innovation_chain,
                    limitations            = excluded.limitations,
                    introduced_year        = excluded.introduced_year,
                    domain_tags            = excluded.domain_tags,
                    source                 = excluded.source,
                    source_refs            = excluded.source_refs,
                    content_angles         = excluded.content_angles,
                    updated_at             = datetime('now')
                """,
                (
                    concept.user_id,
                    concept.name,
                    slug,
                    concept.concept_type,
                    concept.what_it_is,
                    concept.what_problem_it_solves,
                    json.dumps(concept.innovation_chain),
                    json.dumps(concept.limitations),
                    concept.introduced_year,
                    json.dumps(concept.domain_tags),
                    concept.source,
                    json.dumps(concept.source_refs),
                    json.dumps(concept.content_angles),
                ),
            )

    def get_concept_by_name(self, name: str, user_id: str) -> Concept | None:
        """Return a Concept by exact name (case-insensitive slug lookup) and user_id,
        or None if not found."""
        slug = self._slugify(name)
        row = self._conn.execute(
            """
            SELECT name, concept_type, what_it_is, what_problem_it_solves,
                   innovation_chain, limitations, introduced_year, domain_tags,
                   source, source_refs, content_angles, user_id
            FROM concepts
            WHERE slug = ? AND user_id = ?
            """,
            (slug, user_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_concept(row)

    def list_concepts(
        self, user_id: str, domain_tag: str | None = None
    ) -> list[Concept]:
        """Return all concepts for user_id, optionally filtered by a domain_tag."""
        rows = self._conn.execute(
            """
            SELECT name, concept_type, what_it_is, what_problem_it_solves,
                   innovation_chain, limitations, introduced_year, domain_tags,
                   source, source_refs, content_angles, user_id
            FROM concepts
            WHERE user_id = ?
            ORDER BY name
            """,
            (user_id,),
        ).fetchall()
        concepts = [self._row_to_concept(r) for r in rows]
        if domain_tag is not None:
            concepts = [c for c in concepts if domain_tag in c.domain_tags]
        return concepts

    # ------------------------------------------------------------------
    # Typed relationships (TASK-M1-001)
    # ------------------------------------------------------------------

    def _get_concept_id(self, name: str, user_id: str) -> int:
        """Return the integer PK for a concept, raising ValueError if absent."""
        slug = self._slugify(name)
        row = self._conn.execute(
            "SELECT id FROM concepts WHERE slug = ? AND user_id = ?",
            (slug, user_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"Concept not found: {name!r} (user_id={user_id!r})")
        return row["id"]

    def upsert_concept_relationship(self, rel: ConceptRelationship) -> None:
        """Insert or update a typed relationship between two concepts."""
        src_id = self._get_concept_id(rel.from_concept, rel.user_id)
        tgt_id = self._get_concept_id(rel.to_concept, rel.user_id)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO concept_relationships
                    (user_id, source_concept_id, target_concept_id,
                     relationship_type, label, source)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, source_concept_id, target_concept_id,
                            relationship_type) DO UPDATE SET
                    label  = excluded.label,
                    source = excluded.source
                """,
                (
                    rel.user_id,
                    src_id,
                    tgt_id,
                    rel.relationship_type,
                    rel.label,
                    rel.source_ref,
                ),
            )

    def get_relationships(
        self, concept_name: str, user_id: str
    ) -> list[ConceptRelationship]:
        """Return all outgoing relationships from concept_name for user_id."""
        src_id = self._get_concept_id(concept_name, user_id)
        rows = self._conn.execute(
            """
            SELECT c_src.name  AS from_concept,
                   c_tgt.name  AS to_concept,
                   cr.relationship_type,
                   cr.label,
                   cr.source   AS source_ref,
                   cr.user_id
            FROM concept_relationships cr
            JOIN concepts c_src ON cr.source_concept_id = c_src.id
            JOIN concepts c_tgt ON cr.target_concept_id = c_tgt.id
            WHERE cr.source_concept_id = ? AND cr.user_id = ?
            """,
            (src_id, user_id),
        ).fetchall()
        return [
            ConceptRelationship(
                from_concept=r["from_concept"],
                to_concept=r["to_concept"],
                relationship_type=r["relationship_type"],
                label=r["label"] or "",
                source_ref=r["source_ref"],
                user_id=r["user_id"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Papers (TASK-M1-001)
    # ------------------------------------------------------------------

    def upsert_paper(self, paper: PaperRecord) -> None:
        """Insert or update a paper in the concept-first papers table."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO papers (user_id, arxiv_id, title, abstract, authors,
                                    published_date)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, arxiv_id) DO UPDATE SET
                    title          = excluded.title,
                    abstract       = excluded.abstract,
                    authors        = excluded.authors,
                    published_date = excluded.published_date
                """,
                (
                    paper.user_id,
                    paper.arxiv_id,
                    paper.title,
                    paper.abstract,
                    json.dumps(paper.authors),
                    paper.published_date,
                ),
            )

    def _get_paper_id(self, arxiv_id: str, user_id: str) -> int:
        """Return the integer PK for a paper row, raising ValueError if absent."""
        row = self._conn.execute(
            "SELECT id FROM papers WHERE arxiv_id = ? AND user_id = ?",
            (arxiv_id, user_id),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Paper not found: {arxiv_id!r} (user_id={user_id!r})"
            )
        return row["id"]

    def link_paper_to_concept(self, link: PaperConceptLink) -> None:
        """Insert a paper → concept link (idempotent — silently ignores duplicates)."""
        paper_id = self._get_paper_id(link.arxiv_id, link.user_id)
        concept_id = self._get_concept_id(link.concept_name, link.user_id)
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO paper_concept_links
                    (paper_id, concept_id, link_type, user_id)
                VALUES (?, ?, ?, ?)
                """,
                (paper_id, concept_id, link.link_type, link.user_id),
            )

    # ------------------------------------------------------------------
    # Citation snapshots / trend resurrection (TASK-M1-001)
    # ------------------------------------------------------------------

    def write_citation_snapshot(self, snapshot: CitationSnapshot) -> None:
        """Insert or replace a citation snapshot for a paper on a given date."""
        paper_id = self._get_paper_id(snapshot.arxiv_id, snapshot.user_id)
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO citation_snapshots
                    (paper_id, check_date, citation_count, user_id)
                VALUES (?, ?, ?, ?)
                """,
                (
                    paper_id,
                    snapshot.snapshot_date,
                    snapshot.citation_count,
                    snapshot.user_id,
                ),
            )

    def get_citation_delta(self, arxiv_id: str, weeks: int, user_id: str) -> int:
        """Return the change in citation_count over the last `weeks` weeks.

        Compares the most-recent snapshot against the oldest snapshot that is
        at least `weeks` * 7 days before the most recent one.
        Returns 0 if fewer than 2 snapshots exist for that window.
        """
        paper_id = self._get_paper_id(arxiv_id, user_id)
        latest_row = self._conn.execute(
            """
            SELECT citation_count, check_date
            FROM citation_snapshots
            WHERE paper_id = ? AND user_id = ?
            ORDER BY check_date DESC
            LIMIT 1
            """,
            (paper_id, user_id),
        ).fetchone()
        if latest_row is None:
            return 0
        cutoff = self._conn.execute(
            """
            SELECT check_date
            FROM citation_snapshots
            WHERE paper_id = ? AND user_id = ?
              AND check_date <= date(?, ?)
            ORDER BY check_date DESC
            LIMIT 1
            """,
            (paper_id, user_id, latest_row["check_date"], f"-{weeks * 7} days"),
        ).fetchone()
        if cutoff is None:
            return 0
        earlier_row = self._conn.execute(
            """
            SELECT citation_count
            FROM citation_snapshots
            WHERE paper_id = ? AND user_id = ?
              AND check_date = ?
            """,
            (paper_id, user_id, cutoff["check_date"]),
        ).fetchone()
        if earlier_row is None:
            return 0
        return latest_row["citation_count"] - earlier_row["citation_count"]

    def get_resurrection_cohort(
        self, min_delta: int, max_total: int, user_id: str
    ) -> list[ResurrectionCandidate]:
        """Return papers whose citation delta (4-week window) is >= min_delta
        and whose most-recent total is <= max_total.

        Each result is a ResurrectionCandidate with the best matching concept name
        (first paper_concept_link found, or empty string if none).
        """
        paper_rows = self._conn.execute(
            """
            SELECT p.arxiv_id, p.user_id,
                   MAX(cs.citation_count) AS latest_count
            FROM papers p
            JOIN citation_snapshots cs ON cs.paper_id = p.id
            WHERE p.user_id = ?
            GROUP BY p.id
            HAVING latest_count <= ?
            """,
            (user_id, max_total),
        ).fetchall()

        candidates: list[ResurrectionCandidate] = []
        for r in paper_rows:
            delta = self.get_citation_delta(r["arxiv_id"], 4, user_id)
            if delta < min_delta:
                continue
            # Find first linked concept name
            paper_id = self._get_paper_id(r["arxiv_id"], user_id)
            concept_row = self._conn.execute(
                """
                SELECT c.name
                FROM paper_concept_links pcl
                JOIN concepts c ON pcl.concept_id = c.id
                WHERE pcl.paper_id = ? AND pcl.user_id = ?
                LIMIT 1
                """,
                (paper_id, user_id),
            ).fetchone()
            concept_name = concept_row["name"] if concept_row else ""
            candidates.append(
                ResurrectionCandidate(
                    arxiv_id=r["arxiv_id"],
                    concept_name=concept_name,
                    delta_citations=delta,
                    weeks_observed=4,
                    user_id=user_id,
                )
            )
        return candidates

    # ------------------------------------------------------------------
    # Commercial signal instrument (TASK-M1-001)
    # ------------------------------------------------------------------

    def log_concept_query(self, query: ConceptQuery) -> None:
        """Record an explore query for the commercial signal instrument."""
        concept_id = self._get_concept_id(query.concept_name, query.user_id)
        queried_at = query.queried_at or None  # None → SQLite default (datetime('now'))
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO concept_queries (user_id, concept_id, query_text, queried_at)
                VALUES (?, ?, ?, COALESCE(?, datetime('now')))
                """,
                (query.user_id, concept_id, query.export_format, queried_at),
            )

    def log_content_publication(self, pub: ContentPublication) -> None:
        """Record a content publication for the commercial signal instrument."""
        concept_id = self._get_concept_id(pub.concept_name, pub.user_id)
        published_at = pub.published_at or None
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO content_publications
                    (user_id, channel, url, concept_ids, published_at)
                VALUES (?, ?, ?, ?, COALESCE(?, datetime('now')))
                """,
                (
                    pub.user_id,
                    pub.channel,
                    pub.url,
                    json.dumps([concept_id]),
                    published_at,
                ),
            )

    def loop_report(self, user_id: str) -> dict:
        """Return the 30-day usage-to-publication loop metrics.

        Returns:
            {
                "queries_last_30d": int,
                "publications_last_30d": int,
                "conversion_rate": float   # publications / queries, or 0.0 if no queries
            }
        """
        queries_row = self._conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM concept_queries
            WHERE user_id = ?
              AND queried_at >= datetime('now', '-30 days')
            """,
            (user_id,),
        ).fetchone()
        pubs_row = self._conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM content_publications
            WHERE user_id = ?
              AND published_at >= datetime('now', '-30 days')
            """,
            (user_id,),
        ).fetchone()
        queries = queries_row["cnt"] if queries_row else 0
        pubs = pubs_row["cnt"] if pubs_row else 0
        rate = round(pubs / queries, 4) if queries > 0 else 0.0
        return {
            "queries_last_30d": queries,
            "publications_last_30d": pubs,
            "conversion_rate": rate,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_concept(row: sqlite3.Row) -> Concept:
        """Convert a concepts table row to a Concept model."""
        return Concept(
            name=row["name"],
            concept_type=row["concept_type"],
            what_it_is=row["what_it_is"] or "",
            what_problem_it_solves=row["what_problem_it_solves"] or "",
            innovation_chain=json.loads(row["innovation_chain"] or "[]"),
            limitations=json.loads(row["limitations"] or "[]"),
            introduced_year=row["introduced_year"],
            domain_tags=json.loads(row["domain_tags"] or "[]"),
            source=row["source"] or "manual",
            source_refs=json.loads(row["source_refs"] or "[]"),
            content_angles=json.loads(row["content_angles"] or "[]"),
            user_id=row["user_id"],
        )

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> Node:
        props = json.loads(row["properties"]) if row["properties"] else {}
        return Node(
            id=row["id"],
            node_type=row["node_type"],
            label=row["label"],
            properties=props,
        )

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> Edge:
        props = json.loads(row["properties"]) if row["properties"] else {}
        return Edge(
            source_id=row["source_id"],
            target_id=row["target_id"],
            relationship_type=row["relationship_type"],
            weight=row["weight"],
            properties=props,
        )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
