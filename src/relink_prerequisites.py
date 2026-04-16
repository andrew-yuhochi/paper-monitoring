"""Re-link prerequisite edges from stored prerequisite_concept_names.

Reads all concept nodes that have prerequisite_concept_names in their
properties and creates PREREQUISITE_OF edges via ConceptLinker fuzzy matching.
Does not require Ollama or network — works entirely from stored data.

Usage:
    python -m src.relink_prerequisites              # Apply
    python -m src.relink_prerequisites --dry-run     # Report only
"""
import argparse
import json
import logging

from src.config import Settings, settings as default_settings
from src.services.linker import ConceptLinker
from src.store.graph_store import GraphStore
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def relink_prerequisites(
    cfg: Settings | None = None,
    dry_run: bool = False,
) -> dict:
    cfg = cfg or default_settings
    store = GraphStore(cfg.db_path)
    linker = ConceptLinker(cfg=cfg)

    # Read all concepts with stored prerequisite names
    rows = store._conn.execute(
        """
        SELECT id, label, json_extract(properties, '$.prerequisite_concept_names') as prereqs
        FROM nodes
        WHERE node_type = 'concept'
          AND json_extract(properties, '$.prerequisite_concept_names') IS NOT NULL
          AND json_extract(properties, '$.prerequisite_concept_names') != '[]'
        """
    ).fetchall()

    edges_before = store._conn.execute(
        "SELECT COUNT(*) FROM edges WHERE relationship_type = 'PREREQUISITE_OF'"
    ).fetchone()[0]

    logger.info(
        "Found %d concepts with stored prerequisite names (current edges: %d)",
        len(rows), edges_before,
    )

    total_linked = 0
    for row in rows:
        concept_id = row["id"]
        label = row["label"]
        prereq_names = json.loads(row["prereqs"]) if row["prereqs"] else []

        if not prereq_names:
            continue

        if dry_run:
            logger.info("  [dry-run] %s -> %d prerequisites: %s", label, len(prereq_names), prereq_names)
            total_linked += len(prereq_names)
        else:
            matched = linker.link_concept_prerequisites(concept_id, prereq_names, store)
            total_linked += len(matched)

    edges_after = store._conn.execute(
        "SELECT COUNT(*) FROM edges WHERE relationship_type = 'PREREQUISITE_OF'"
    ).fetchone()[0]

    result = {
        "concepts_with_prereqs": len(rows),
        "edges_before": edges_before,
        "edges_after": edges_after,
        "new_edges": edges_after - edges_before,
    }
    logger.info("Re-link complete: %s", result)
    store.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-link prerequisite edges from stored data.")
    parser.add_argument("--dry-run", action="store_true", help="Report without creating edges.")
    args = parser.parse_args()

    setup_logging(log_dir=default_settings.log_dir, log_level=default_settings.log_level)
    relink_prerequisites(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
