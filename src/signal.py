"""Signal CLI — TASK-M1-004.

Subcommands:
    log-publication  — Record a content publication derived from a concept.
    report           — Print usage-to-publication loop metrics.

Usage:
    python -m src.signal log-publication --concept xgboost --channel linkedin --url "https://..."
    python -m src.signal report --days 30
    python -m src.signal report --days 30 --user default
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import settings
from src.models.concepts import VALID_CHANNELS
from src.services.signal_logger import SignalLogger
from src.store.graph_store import GraphStore
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_log_publication(args: argparse.Namespace) -> int:
    """Write a row to content_publications."""
    store = GraphStore(settings.db_path)
    logger_svc = SignalLogger(store=store)

    try:
        logger_svc.log_publication(
            concept_name=args.concept,
            channel=args.channel,
            url=args.url or None,
            user_id=args.user,
        )
        print(
            f"Logged publication: {args.concept!r} on {args.channel!r}"
            + (f" at {args.url}" if args.url else "")
        )
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()


def _cmd_report(args: argparse.Namespace) -> int:
    """Print loop-ratio and top-concept metrics."""
    store = GraphStore(settings.db_path)
    logger_svc = SignalLogger(store=store)

    try:
        data = logger_svc.report(days=args.days, user_id=args.user)
    finally:
        store.close()

    queries = data["queries_last_30d"]
    pubs = data["publications_last_30d"]
    rate = data["conversion_rate"]
    days = data["days_window"]

    print()
    print(f"Signal Report — last {days} days (user: {args.user})")
    print("─" * 48)
    print(f"Concept queries       : {queries}")
    print(f"Content publications  : {pubs}")
    loop_pct = round(rate * 100, 1)
    print(f"Usage-to-publish ratio: {loop_pct}%  (target ≥ 30% at MVP exit)")
    print()

    if data["top_queried_concepts"]:
        print("Top queried concepts:")
        for item in data["top_queried_concepts"]:
            print(f"  · {item['name']} ({item['count']}x)")
    else:
        print("Top queried concepts: (none yet)")

    print()

    if data["published_concepts"]:
        print("Published concepts:")
        for item in data["published_concepts"]:
            print(f"  · {item['name']} on {item['channel']} ({item['count']}x)")
    else:
        print("Published concepts: (none yet)")

    print()
    return 0


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    setup_logging(log_dir=settings.log_dir, log_level=settings.log_level)

    parser = argparse.ArgumentParser(
        prog="python -m src.signal",
        description="Commercial signal instrument: log publications and view metrics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- log-publication ---
    pub_parser = sub.add_parser(
        "log-publication",
        help="Record a content publication derived from a concept.",
    )
    pub_parser.add_argument(
        "--concept",
        required=True,
        help="Concept name (exact or partial slug, e.g. 'xgboost').",
    )
    pub_parser.add_argument(
        "--channel",
        required=True,
        choices=list(VALID_CHANNELS),
        help="Publication channel: linkedin, youtube, newsletter, other.",
    )
    pub_parser.add_argument(
        "--url",
        default=None,
        help="URL of the published post (optional).",
    )
    pub_parser.add_argument(
        "--user",
        default="default",
        help="User ID (default: 'default').",
    )

    # --- report ---
    rep_parser = sub.add_parser(
        "report",
        help="Print query count, publication count, loop ratio, top concepts.",
    )
    rep_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Report window in days (default: 30).",
    )
    rep_parser.add_argument(
        "--user",
        default="default",
        help="User ID (default: 'default').",
    )

    args = parser.parse_args(argv)

    if args.command == "log-publication":
        return _cmd_log_publication(args)
    elif args.command == "report":
        return _cmd_report(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
