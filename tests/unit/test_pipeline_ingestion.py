"""Unit tests for the pipeline ingestion stage (TASK-011)."""

import logging
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.models.arxiv import ArxivPaper
from src.models.huggingface import HFPaper
from src.pipeline import _run_ingestion, run_pipeline
from src.store.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_paper(arxiv_id: str, primary_category: str = "cs.LG") -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        abstract="Abstract.",
        authors=["Author"],
        primary_category=primary_category,
        all_categories=[primary_category],
        published_date=date.today().isoformat(),
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def _make_cfg(top_n: int = 100) -> MagicMock:
    cfg = MagicMock()
    cfg.arxiv_categories = ["cs.LG", "cs.AI"]
    cfg.arxiv_max_results_per_category = 50
    cfg.prefilter_top_n = top_n
    cfg.prefilter_upvote_weight = 2.0
    cfg.prefilter_category_priorities = {"cs.LG": 5, "cs.AI": 3}
    cfg.db_path = ":memory:"
    cfg.hf_fetch_delay = 0.0
    return cfg


# ---------------------------------------------------------------------------
# Run tracking: create_run / update_run via pipeline
# ---------------------------------------------------------------------------


def test_run_record_created_and_completed() -> None:
    """Pipeline creates a weekly_runs record and marks it 'completed' on success."""
    store = GraphStore(":memory:")
    cfg = _make_cfg()

    with (
        patch("src.pipeline.ArxivFetcher") as mock_arxiv_cls,
        patch("src.pipeline.HuggingFaceFetcher") as mock_hf_cls,
        patch("src.pipeline._run_storage_and_rendering", return_value="/tmp/digest.html"),
    ):
        mock_arxiv_cls.return_value.fetch_recent.return_value = []
        mock_hf_cls.return_value.fetch_week.return_value = {}
        run_pipeline(store=store, cfg=cfg)

    run = store.get_latest_run()
    assert run is not None
    assert run.status == "completed"
    assert run.run_date == date.today().isoformat()


def test_arxiv_failure_marks_run_as_failed() -> None:
    """ArxivFetcher exception aborts the pipeline and marks the run as 'failed'."""
    store = GraphStore(":memory:")
    cfg = _make_cfg()

    with patch("src.pipeline.ArxivFetcher") as mock_arxiv_cls:
        mock_arxiv_cls.return_value.fetch_recent.side_effect = RuntimeError("arXiv unreachable")

        with pytest.raises(RuntimeError, match="arXiv unreachable"):
            run_pipeline(store=store, cfg=cfg)

    run = store.get_latest_run()
    assert run is not None
    assert run.status == "failed"
    assert "arXiv unreachable" in (run.error_message or "")


def test_papers_fetched_stored_in_run_record() -> None:
    """papers_fetched in the run record equals total raw papers from arXiv."""
    store = GraphStore(":memory:")
    papers = [_make_paper(str(i)) for i in range(5)]
    cfg = _make_cfg()

    with (
        patch("src.pipeline.ArxivFetcher") as mock_arxiv_cls,
        patch("src.pipeline.HuggingFaceFetcher") as mock_hf_cls,
        patch("src.pipeline.OllamaClassifier") as mock_classifier_cls,
        patch("src.pipeline._run_storage_and_rendering", return_value="/tmp/digest.html"),
    ):
        mock_arxiv_cls.return_value.fetch_recent.return_value = papers
        mock_hf_cls.return_value.fetch_week.return_value = {}
        mock_classifier_cls.return_value.classify_paper.return_value = MagicMock(
            classification_failed=False, tier=2
        )
        run_pipeline(store=store, cfg=cfg)

    run = store.get_latest_run()
    assert run.papers_fetched == 5


# ---------------------------------------------------------------------------
# Known-paper deduplication
# ---------------------------------------------------------------------------


def test_already_known_papers_excluded_from_candidates(in_memory_store: GraphStore) -> None:
    """Papers already in the database are skipped in the candidate list."""
    cfg = _make_cfg()
    # Pre-seed two papers as known
    for arxiv_id in ("001", "002"):
        in_memory_store.upsert_node(f"paper:{arxiv_id}", "paper", f"Paper {arxiv_id}", {})

    all_papers = [_make_paper("001"), _make_paper("002"), _make_paper("003")]

    with (
        patch("src.pipeline.ArxivFetcher") as mock_arxiv_cls,
        patch("src.pipeline.HuggingFaceFetcher") as mock_hf_cls,
    ):
        mock_arxiv_cls.return_value.fetch_recent.return_value = all_papers
        mock_hf_cls.return_value.fetch_week.return_value = {}
        _, _, candidates = _run_ingestion(in_memory_store, cfg)

    candidate_ids = {c.paper.arxiv_id for c in candidates}
    assert candidate_ids == {"003"}


def test_no_known_papers_all_become_candidates() -> None:
    """When the database is empty, all fetched papers become candidates."""
    store = GraphStore(":memory:")
    cfg = _make_cfg(top_n=10)
    papers = [_make_paper(str(i)) for i in range(3)]

    with (
        patch("src.pipeline.ArxivFetcher") as mock_arxiv_cls,
        patch("src.pipeline.HuggingFaceFetcher") as mock_hf_cls,
    ):
        mock_arxiv_cls.return_value.fetch_recent.return_value = papers
        mock_hf_cls.return_value.fetch_week.return_value = {}
        _, _, candidates = _run_ingestion(store, cfg)

    assert len(candidates) == 3


# ---------------------------------------------------------------------------
# HuggingFace degradation
# ---------------------------------------------------------------------------


def test_hf_unavailable_pipeline_still_completes() -> None:
    """HuggingFaceFetcher returning empty dict does not abort the pipeline."""
    store = GraphStore(":memory:")
    papers = [_make_paper("001"), _make_paper("002")]
    cfg = _make_cfg()

    with (
        patch("src.pipeline.ArxivFetcher") as mock_arxiv_cls,
        patch("src.pipeline.HuggingFaceFetcher") as mock_hf_cls,
        patch("src.pipeline.OllamaClassifier") as mock_classifier_cls,
        patch("src.pipeline._run_storage_and_rendering", return_value="/tmp/digest.html"),
    ):
        mock_arxiv_cls.return_value.fetch_recent.return_value = papers
        mock_hf_cls.return_value.fetch_week.return_value = {}
        mock_classifier_cls.return_value.classify_paper.return_value = MagicMock(
            classification_failed=False, tier=2
        )
        run_pipeline(store=store, cfg=cfg)

    assert store.get_latest_run().status == "completed"


def test_hf_unavailable_logged_as_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Empty HF response is logged as a WARNING."""
    store = GraphStore(":memory:")
    cfg = _make_cfg()

    with (
        patch("src.pipeline.ArxivFetcher") as mock_arxiv_cls,
        patch("src.pipeline.HuggingFaceFetcher") as mock_hf_cls,
        caplog.at_level(logging.WARNING, logger="src.pipeline"),
    ):
        mock_arxiv_cls.return_value.fetch_recent.return_value = []
        mock_hf_cls.return_value.fetch_week.return_value = {}
        _run_ingestion(store, cfg)

    assert any("HuggingFace" in r.message and r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# Pre-filter integration
# ---------------------------------------------------------------------------


def test_prefilter_top_n_limits_candidates() -> None:
    """Pre-filter respects top_n and returns at most that many candidates."""
    store = GraphStore(":memory:")
    papers = [_make_paper(str(i)) for i in range(10)]
    cfg = _make_cfg(top_n=3)

    with (
        patch("src.pipeline.ArxivFetcher") as mock_arxiv_cls,
        patch("src.pipeline.HuggingFaceFetcher") as mock_hf_cls,
    ):
        mock_arxiv_cls.return_value.fetch_recent.return_value = papers
        mock_hf_cls.return_value.fetch_week.return_value = {}
        _, _, candidates = _run_ingestion(store, cfg)

    assert len(candidates) <= 3


def test_ingestion_summary_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Ingestion logs arXiv fetch count and pre-filter candidate count."""
    store = GraphStore(":memory:")
    papers = [_make_paper(str(i)) for i in range(4)]
    cfg = _make_cfg()

    with (
        patch("src.pipeline.ArxivFetcher") as mock_arxiv_cls,
        patch("src.pipeline.HuggingFaceFetcher") as mock_hf_cls,
        caplog.at_level(logging.INFO, logger="src.pipeline"),
    ):
        mock_arxiv_cls.return_value.fetch_recent.return_value = papers
        mock_hf_cls.return_value.fetch_week.return_value = {}
        _run_ingestion(store, cfg)

    messages = " ".join(r.message for r in caplog.records)
    assert "4" in messages          # fetched count appears in a log message
    assert "arXiv" in messages
    assert "Pre-filter" in messages
