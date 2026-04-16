"""GraphStore: SQLite persistence layer using a nodes+edges graph schema.

All tables use the graph model (nodes + edges) as the core structure.
No ORM — plain sqlite3 from the stdlib.
"""

import json
import logging
import sqlite3
from pathlib import Path

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
            self._conn.execute(_CREATE_NODES)
            self._conn.execute(_CREATE_EDGES)
            self._conn.execute(_CREATE_WEEKLY_RUNS)
            for stmt in _CREATE_INDEXES:
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
        """Return all paper nodes with parsed properties, ordered by published_date DESC.

        Each returned dict has keys: id, label, properties (parsed from JSON).
        Returns [] if no paper nodes exist.
        """
        rows = self._conn.execute(
            """
            SELECT id, label, properties
            FROM nodes
            WHERE node_type = 'paper'
            ORDER BY json_extract(properties, '$.published_date') DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "label": r["label"],
                "properties": json.loads(r["properties"]) if r["properties"] else {},
            }
            for r in rows
        ]

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
    # Internal helpers
    # ------------------------------------------------------------------

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
