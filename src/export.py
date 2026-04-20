"""CLI entry point for the GraphExporter.

Usage:
    python -m src.export --format obsidian
    python -m src.export --format neo4j
    python -m src.export --format cytoscape

Output directories:
    obsidian  → <project_root>/obsidian_vault/
    neo4j     → <project_root>/cypher_exports/
    cytoscape → <project_root>/exports/
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import settings
from src.services.graph_exporter import GraphExporter
from src.store.graph_store import GraphStore
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

VALID_FORMATS = ("obsidian", "neo4j", "cytoscape")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.export",
        description="Export the concept graph to Obsidian vault, Neo4j Cypher, or Cytoscape JSON.",
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=VALID_FORMATS,
        metavar="FORMAT",
        help=f"Export format: one of {VALID_FORMATS}",
    )
    parser.add_argument(
        "--user-id",
        default="default",
        help="user_id namespace to export (default: 'default')",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to SQLite database (default: from config)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the graph export CLI. Returns exit code (0 = success)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging(settings.log_dir, settings.log_level)
    db_path = args.db or settings.db_path
    project_root = settings.project_root

    logger.info("Opening database: %s", db_path)
    store = GraphStore(db_path)
    exporter = GraphExporter(store, user_id=args.user_id)

    try:
        if args.format == "obsidian":
            vault_dir = project_root / "obsidian_vault"
            count = exporter.to_obsidian_vault(vault_dir)
            print(f"Obsidian vault: {count} concept files written to {vault_dir}/concepts/")

        elif args.format == "neo4j":
            cypher_dir = project_root / "cypher_exports"
            out_path = exporter.to_neo4j_cypher(cypher_dir)
            print(f"Neo4j Cypher export: {out_path}")

        elif args.format == "cytoscape":
            exports_dir = project_root / "exports"
            out_path = exporter.to_cytoscape_json(exports_dir)
            print(f"Cytoscape JSON export: {out_path}")

    except Exception as exc:
        logger.error("Export failed: %s", exc, exc_info=True)
        return 1
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
