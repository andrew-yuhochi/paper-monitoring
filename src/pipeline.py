"""Weekly paper monitoring pipeline.

Stages (added task by task):
  1. Ingestion    — TASK-011: fetch arXiv + HF, deduplicate, pre-filter
  2. Classification — TASK-012: classify candidates via Ollama
  3. Storage + Rendering — TASK-014: persist to graph, render HTML digest
"""
from __future__ import annotations

import json as _json
import logging
import sys
from datetime import date
from typing import Callable

from src.config import Settings, settings as default_settings
from src.integrations.arxiv_client import ArxivFetcher
from src.integrations.hf_client import HuggingFaceFetcher
from src.models.arxiv import ArxivPaper
from src.models.classification import PaperClassification
from src.models.graph import DigestEntry, Node, ScoredPaper
from src.models.huggingface import HFPaper
from src.services.classifier import OllamaClassifier
from src.services.linker import ConceptLinker
from src.services.prefilter import PreFilter
from src.services.renderer import DigestRenderer
from src.store.graph_store import GraphStore
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: Ingestion helpers
# ---------------------------------------------------------------------------


def _fetch_arxiv(cfg: Settings) -> list[ArxivPaper]:
    """Fetch recent arXiv papers across all configured categories.

    Raises on persistent failure after retries — aborts the pipeline.
    """
    fetcher = ArxivFetcher(cfg=cfg)
    return fetcher.fetch_recent(
        categories=cfg.arxiv_categories,
        max_results=cfg.arxiv_max_results_per_category,
    )


def _fetch_hf(cfg: Settings) -> dict[str, HFPaper]:
    """Fetch HuggingFace daily papers for the past week.

    Returns empty dict on any failure — HuggingFaceFetcher already handles
    errors internally and never raises. An empty return degrades the
    pre-filter to category-priority-only scoring.
    """
    fetcher = HuggingFaceFetcher(cfg=cfg)
    return fetcher.fetch_week(end_date=date.today().isoformat())


def _run_ingestion(
    store: GraphStore,
    cfg: Settings,
) -> tuple[list[ArxivPaper], dict[str, HFPaper], list[ScoredPaper]]:
    """Stage 1: fetch, filter already-known papers, pre-filter.

    Returns:
        Tuple of (arxiv_papers, hf_data, candidates) where:
          - arxiv_papers: all raw papers fetched from arXiv (before known-paper skip)
          - hf_data: HF upvote lookup dict (empty if HF was unavailable)
          - candidates: top-N scored papers ready for classification

    Raises:
        Exception: propagated from ArxivFetcher if all retries fail.
    """
    # 1a. Fetch from arXiv — raises on failure, which aborts the pipeline
    arxiv_papers = _fetch_arxiv(cfg)
    logger.info(
        "arXiv: fetched %d papers across %d categories",
        len(arxiv_papers),
        len(cfg.arxiv_categories),
    )

    # 1b. Fetch from HuggingFace — never raises; returns {} if unavailable
    hf_data = _fetch_hf(cfg)
    if hf_data:
        logger.info("HuggingFace: %d papers with upvote data", len(hf_data))
    else:
        logger.warning(
            "HuggingFace: no data available — pre-filter will use category scores only"
        )

    # 1c. Skip papers already stored in the graph
    new_papers = [p for p in arxiv_papers if not store.paper_exists(p.arxiv_id)]
    known_count = len(arxiv_papers) - len(new_papers)
    logger.info(
        "Deduplication: %d new papers, %d already in database (skipped)",
        len(new_papers),
        known_count,
    )

    # 1d. Pre-filter to top-N candidates by score
    prefilter = PreFilter(cfg=cfg)
    candidates = prefilter.score_and_filter(new_papers, hf_data)
    logger.info(
        "Pre-filter: %d candidates selected (top %d by score)",
        len(candidates),
        cfg.prefilter_top_n,
    )

    return arxiv_papers, hf_data, candidates


# ---------------------------------------------------------------------------
# Stage 2: Classification helpers
# ---------------------------------------------------------------------------


def _run_classification(
    candidates: list[ScoredPaper],
    store: GraphStore,
    cfg: Settings,
    classifier: OllamaClassifier | None = None,
    on_paper_classified: Callable[[int, int], None] | None = None,
) -> list[tuple[ScoredPaper, PaperClassification]]:
    """Stage 2: classify each candidate paper via Ollama.

    Loads the concept index from the graph store, then calls
    ``OllamaClassifier.classify_paper()`` for each candidate in order.
    Papers where Ollama fails all retries are preserved with
    ``classification_failed=True`` rather than being dropped.

    Args:
        candidates: Scored candidates from Stage 1.
        store: GraphStore used to load the concept index.
        cfg: Settings passed to OllamaClassifier if no classifier is injected.
        classifier: Optional pre-built classifier (injected for testing).

    Returns:
        List of (ScoredPaper, PaperClassification) tuples — one per candidate,
        in the same order as ``candidates``.

    Raises:
        RuntimeError: If Ollama is not running or the configured model is not
            found. These errors abort the pipeline immediately.
    """
    if not candidates:
        logger.info("Classification: no candidates to classify")
        return []

    concept_index = store.get_concept_index()
    logger.info("Classification: loaded %d concepts from knowledge bank", len(concept_index))

    _classifier = classifier or OllamaClassifier(cfg=cfg)
    total = len(candidates)
    results: list[tuple[ScoredPaper, PaperClassification]] = []

    for n, scored in enumerate(candidates, start=1):
        logger.info(
            "Classifying paper %d of %d: %s",
            n,
            total,
            scored.paper.title,
        )
        classification = _classifier.classify_paper(scored.paper, concept_index)
        results.append((scored, classification))
        if on_paper_classified:
            on_paper_classified(n, total)

    classified_count = sum(1 for _, c in results if not c.classification_failed)
    failed_count = total - classified_count

    tier_dist: dict[int | None, int] = {}
    for _, c in results:
        tier_dist[c.tier] = tier_dist.get(c.tier, 0) + 1

    tier_summary = ", ".join(
        f"T{t}={count}" for t, count in sorted(tier_dist.items(), key=lambda x: (x[0] is None, x[0]))
        if t is not None
    )
    if failed_count:
        tier_summary = (tier_summary + f", failed={failed_count}").lstrip(", ")

    logger.info(
        "Classification complete: %d classified, %d failed — %s",
        classified_count,
        failed_count,
        tier_summary or "no papers",
    )

    return results


# ---------------------------------------------------------------------------
# Stage 3: Storage and rendering helpers
# ---------------------------------------------------------------------------


def _run_storage_and_rendering(
    classified: list[tuple[ScoredPaper, PaperClassification]],
    run_date: str,
    store: GraphStore,
    cfg: Settings,
    linker: ConceptLinker | None = None,
    renderer: DigestRenderer | None = None,
) -> str:
    """Stage 3: persist paper nodes + edges to GraphStore and render digest HTML.

    For each classified paper:
      1. Builds a properties dict with full classification metadata.
      2. Upserts a paper node into GraphStore.
      3. Creates BUILDS_ON edges to matched concept nodes via ConceptLinker.
      4. Assembles a DigestEntry with resolved concept Node objects.

    Then renders the weekly digest HTML to ``cfg.digest_output_dir``.

    Args:
        classified: (ScoredPaper, PaperClassification) pairs from Stage 2.
        run_date: ISO date string written into each paper node's properties.
        store: GraphStore instance.
        cfg: Settings for digest_output_dir and template_dir.
        linker: Optional pre-built ConceptLinker (injected for testing).
        renderer: Optional pre-built DigestRenderer (injected for testing).

    Returns:
        String path to the rendered HTML digest file.
    """
    _linker = linker or ConceptLinker(cfg=cfg)
    _renderer = renderer or DigestRenderer(template_dir=cfg.template_dir)

    entries: list[DigestEntry] = []

    for scored, classification in classified:
        paper = scored.paper
        hf_upvotes = scored.hf_data.upvotes if scored.hf_data else 0

        properties: dict = {
            "tier": classification.tier,
            "confidence": classification.confidence,
            "reasoning": classification.reasoning,
            "summary": classification.summary,
            "key_contributions": classification.key_contributions,
            "hf_upvotes": hf_upvotes,
            "prefilter_score": scored.score,
            "run_date": run_date,
            "classification_failed": classification.classification_failed,
            "arxiv_url": paper.arxiv_url,
            "pdf_url": paper.pdf_url,
            "authors": paper.authors,
            "primary_category": paper.primary_category,
            "published_date": paper.published_date,
        }

        paper_node_id = f"paper:{paper.arxiv_id}"
        store.upsert_node(paper_node_id, "paper", paper.title, properties)

        # Create BUILDS_ON edges (skipped for failed classifications)
        linked_concepts: list[Node] = []
        if not classification.classification_failed and classification.foundational_concept_names:
            matched_ids = _linker.link_paper_to_concepts(
                paper_node_id,
                classification.foundational_concept_names,
                store,
            )
            linked_concepts = [
                node
                for nid in matched_ids
                if (node := store.get_node(nid)) is not None
            ]

        entries.append(
            DigestEntry(
                paper=paper,
                classification=classification,
                linked_concepts=linked_concepts,
                hf_upvotes=hf_upvotes,
                prefilter_score=scored.score,
            )
        )

    digest_path = _renderer.render(
        entries=entries,
        run_date=run_date,
        output_dir=cfg.digest_output_dir,
    )

    logger.info(
        "Storage complete: %d paper nodes upserted; digest at %s",
        len(classified),
        digest_path,
    )
    return str(digest_path)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def _noop_progress(step: str, message: str) -> None:
    """Default no-op progress callback — zero overhead for cron path."""


def run_pipeline(
    store: GraphStore | None = None,
    cfg: Settings | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> None:
    """Execute the full weekly paper monitoring pipeline.

    Args:
        store: GraphStore instance. Defaults to production DB at cfg.db_path.
               Pass an in-memory store for testing.
        cfg: Settings instance. Defaults to module-level settings loaded from
             environment / .env. Pass a mock for testing.
        progress_callback: Optional callback ``(step, message) -> None`` called
            at pipeline checkpoints. Used by the Streamlit dashboard to report
            live progress. Passing ``None`` (the default) incurs zero overhead.
    """
    _cfg = cfg or default_settings
    _store = store or GraphStore(_cfg.db_path)
    _progress = progress_callback or _noop_progress
    run_date = date.today().isoformat()

    run_id = _store.create_run(run_date)
    logger.info("Pipeline started — run_id=%d, run_date=%s", run_id, run_date)
    _progress("start", f"Pipeline started — run_date={run_date}")

    try:
        # --- Stage 1: Ingestion ---
        _progress("ingestion", "Fetching arXiv papers...")
        arxiv_papers, _hf_data, candidates = _run_ingestion(_store, _cfg)
        _progress(
            "ingestion",
            f"Ingestion complete — {len(arxiv_papers)} fetched, {len(candidates)} candidates after pre-filter",
        )

        # --- Stage 2: Classification ---
        total = len(candidates)
        _progress("classification", f"Classifying {total} papers...")
        classified = _run_classification(
            candidates, _store, _cfg,
            on_paper_classified=lambda n, t: _progress(
                "classification", f"Classifying paper {n} of {total}"
            ),
        )
        papers_classified = sum(1 for _, c in classified if not c.classification_failed)
        papers_failed = len(classified) - papers_classified
        _progress(
            "classification",
            f"Classification complete — {papers_classified} classified, {papers_failed} failed",
        )

        # --- Stage 3: Storage + Rendering ---
        _progress("storage", "Storing results and rendering digest...")
        digest_path = _run_storage_and_rendering(classified, run_date, _store, _cfg)
        _progress("storage", f"Digest rendered at {digest_path}")

        _store.update_run(
            run_id,
            status="completed",
            papers_fetched=len(arxiv_papers),
            papers_classified=papers_classified,
            papers_failed=papers_failed,
            digest_path=digest_path,
        )
        logger.info(
            "Pipeline completed — run_id=%d, papers_fetched=%d, classified=%d, failed=%d, digest=%s",
            run_id,
            len(arxiv_papers),
            papers_classified,
            papers_failed,
            digest_path,
        )
        _progress("done", f"Pipeline complete — {papers_classified} papers classified")

    except Exception as e:
        _store.update_run(run_id, status="failed", error_message=str(e))
        logger.exception("Pipeline failed — run_id=%d", run_id)
        _progress("error", str(e))
        raise


def _stdout_progress(step: str, message: str) -> None:
    """Progress callback that emits JSON lines to stdout for subprocess parsing."""
    line = _json.dumps({"type": step, "message": message})
    print(line, flush=True)


def main() -> None:
    """CLI entry point. Configures logging then runs the pipeline.

    Accepts ``--progress`` flag to emit JSON progress lines to stdout,
    used by the Streamlit dashboard subprocess.
    """
    setup_logging(log_dir=default_settings.log_dir, log_level=default_settings.log_level)
    progress_cb = _stdout_progress if "--progress" in sys.argv else None
    run_pipeline(progress_callback=progress_cb)


if __name__ == "__main__":
    main()
