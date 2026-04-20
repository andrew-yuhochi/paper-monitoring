"""Concept Explorer CLI — TASK-M1-004.

Usage:
    python -m src.explore "XGBoost"
    python -m src.explore "XGBoost" --format json
    python -m src.explore "XGBoost" --depth 5
    python -m src.explore "XGBoost" --out /tmp/exports/

Resolves concept names with fuzzy matching, traverses BUILDS_ON lineage,
renders the 8-field export, and logs a row to concept_queries.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.config import settings
from src.services.concept_exporter import ConceptExporter, ConceptNotFoundError
from src.services.signal_logger import SignalLogger
from src.store.graph_store import GraphStore
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def _print_markdown_preview(path: Path, concept_name: str, elapsed: float) -> None:
    """Print a brief success summary to the terminal."""
    word_count = len(path.read_text(encoding="utf-8").split())
    print()
    print(f"CONCEPT EXPLORER — {concept_name}")
    print("=" * (len(concept_name) + 20))
    print(f"Export written to: {path}")
    print(f"({word_count} words, retrieved in {elapsed:.1f}s)")


def _print_not_found(query: str, suggestions: list[str]) -> None:
    print(f"\nConcept not found: {query!r}")
    if suggestions:
        print("Did you mean...?")
        for name in suggestions[:3]:
            print(f"  · {name}")
    print()


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns exit code (0 = success, 1 = not found)."""
    setup_logging(log_dir=settings.log_dir, log_level=settings.log_level)
    parser = argparse.ArgumentParser(
        prog="python -m src.explore",
        description="Explore an ML/DS concept — traverse lineage and export content draft.",
    )
    parser.add_argument(
        "concept",
        help='Concept name to explore (fuzzy-matched). Example: "XGBoost"',
    )
    parser.add_argument(
        "--format",
        choices=["md", "json"],
        default="md",
        help="Output format: md (default) or json.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Maximum lineage traversal depth (default: 10).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for the export file (default: exports/ in project root).",
    )
    parser.add_argument(
        "--user",
        default="default",
        help="User ID for multi-tenant isolation (default: 'default').",
    )

    args = parser.parse_args(argv)

    out_dir: Path = args.out or (settings.project_root / "exports")
    start = time.monotonic()

    store = GraphStore(settings.db_path)
    exporter = ConceptExporter(store=store, user_id=args.user)
    signal_logger = SignalLogger(store=store)

    try:
        concept = exporter.resolve_concept(args.concept)
    except ConceptNotFoundError as exc:
        _print_not_found(exc.query, exc.suggestions)
        store.close()
        return 1

    lineage = exporter.traverse_lineage(concept, max_depth=args.depth)

    # Log the query AFTER successful resolution
    signal_logger.log_query(
        concept_name=concept.name,
        query_text=args.format,
        user_id=args.user,
    )

    if args.format == "json":
        out_path = exporter.export_json(concept, out_dir, lineage=lineage)
    else:
        out_path = exporter.export_markdown(concept, out_dir, lineage=lineage)

    elapsed = time.monotonic() - start
    _print_markdown_preview(out_path, concept.name, elapsed)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
