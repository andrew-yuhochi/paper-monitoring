# Commercial-signal instrument: logs concept queries and content publications.
# Called from src/explore.py (every explore call) and src/signal.py (log-publication, report).
# Reads/writes concept_queries and content_publications tables (added in TASK-M1-001).

from __future__ import annotations

import logging
from pathlib import Path

from src.config import settings
from src.models.concepts import ConceptQuery, ContentPublication
from src.store.graph_store import GraphStore

logger = logging.getLogger(__name__)


class SignalLogger:
    """Writes and reads commercial-signal rows (concept_queries, content_publications)."""

    def __init__(self, store: GraphStore | None = None) -> None:
        self._store = store or GraphStore(settings.db_path)
        self._owns_store = store is None

    # ------------------------------------------------------------------
    # Write paths
    # ------------------------------------------------------------------

    def log_query(
        self,
        concept_name: str,
        query_text: str = "markdown",
        user_id: str = "default",
    ) -> None:
        """Record a concept explore call in concept_queries."""
        query = ConceptQuery(
            concept_name=concept_name,
            export_format=query_text,
            user_id=user_id,
        )
        try:
            self._store.log_concept_query(query)
            logger.debug("Logged concept query: %s (user=%s)", concept_name, user_id)
        except Exception:
            logger.warning(
                "Failed to log concept query for %r", concept_name, exc_info=True
            )

    def log_publication(
        self,
        concept_name: str,
        channel: str,
        url: str | None = None,
        user_id: str = "default",
    ) -> None:
        """Record a content publication in content_publications."""
        pub = ContentPublication(
            concept_name=concept_name,
            channel=channel,
            url=url,
            user_id=user_id,
        )
        try:
            self._store.log_content_publication(pub)
            logger.debug(
                "Logged publication: %s on %s (user=%s)", concept_name, channel, user_id
            )
        except Exception:
            logger.warning(
                "Failed to log publication for %r", concept_name, exc_info=True
            )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self, days: int = 30, user_id: str = "default") -> dict:
        """Return loop metrics for the given window.

        Extends GraphStore.loop_report with top-queried and published concept names.
        """
        base = self._store.loop_report(user_id)

        top_queried = self._store._conn.execute(
            """
            SELECT c.name, COUNT(*) AS cnt
            FROM concept_queries cq
            JOIN concepts c ON cq.concept_id = c.id
            WHERE cq.user_id = ?
              AND cq.queried_at >= datetime('now', ?)
            GROUP BY cq.concept_id
            ORDER BY cnt DESC
            LIMIT 10
            """,
            (user_id, f"-{days} days"),
        ).fetchall()

        top_published = self._store._conn.execute(
            """
            SELECT c.name, cp.channel, COUNT(*) AS cnt
            FROM content_publications cp,
                 json_each(cp.concept_ids) AS je
            JOIN concepts c ON c.id = CAST(je.value AS INTEGER)
            WHERE cp.user_id = ?
              AND cp.published_at >= datetime('now', ?)
            GROUP BY c.id, cp.channel
            ORDER BY cnt DESC
            LIMIT 10
            """,
            (user_id, f"-{days} days"),
        ).fetchall()

        base["top_queried_concepts"] = [
            {"name": r["name"], "count": r["cnt"]} for r in top_queried
        ]
        base["published_concepts"] = [
            {"name": r["name"], "channel": r["channel"], "count": r["cnt"]}
            for r in top_published
        ]
        base["days_window"] = days
        return base

    def close(self) -> None:
        if self._owns_store:
            self._store.close()

    def __enter__(self) -> "SignalLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
