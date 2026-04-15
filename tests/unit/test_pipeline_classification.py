"""Unit tests for the pipeline classification stage (TASK-012)."""

import logging
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.models.arxiv import ArxivPaper
from src.models.classification import PaperClassification
from src.models.graph import ScoredPaper
from src.pipeline import _run_classification, run_pipeline
from src.store.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_paper(arxiv_id: str, title: str | None = None) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=title or f"Paper {arxiv_id}",
        abstract="Abstract.",
        authors=["Author"],
        primary_category="cs.LG",
        all_categories=["cs.LG"],
        published_date=date.today().isoformat(),
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def _make_scored(arxiv_id: str, score: float = 1.0) -> ScoredPaper:
    return ScoredPaper(paper=_make_paper(arxiv_id), score=score)


def _make_classification(tier: int = 2) -> PaperClassification:
    return PaperClassification(
        tier=tier,
        confidence="high",
        reasoning="Solid contribution.",
        summary="A two-sentence summary.",
        classification_failed=False,
    )


def _make_failed_classification() -> PaperClassification:
    return PaperClassification(
        tier=None,
        confidence=None,
        reasoning=None,
        summary=None,
        classification_failed=True,
    )


def _make_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.arxiv_categories = ["cs.LG"]
    cfg.arxiv_max_results_per_category = 50
    cfg.prefilter_top_n = 100
    cfg.prefilter_upvote_weight = 2.0
    cfg.prefilter_category_priorities = {"cs.LG": 5}
    cfg.db_path = ":memory:"
    cfg.hf_fetch_delay = 0.0
    return cfg


# ---------------------------------------------------------------------------
# Concept index loading
# ---------------------------------------------------------------------------


def test_concept_index_loaded_from_store(in_memory_store: GraphStore) -> None:
    """_run_classification loads the concept index from the graph store."""
    in_memory_store.upsert_node("concept:attention_mechanism", "concept", "attention mechanism", {})
    in_memory_store.upsert_node("concept:backpropagation", "concept", "backpropagation", {})

    candidates = [_make_scored("001")]
    mock_classifier = MagicMock()
    mock_classifier.classify_paper.return_value = _make_classification()

    _run_classification(candidates, in_memory_store, MagicMock(), classifier=mock_classifier)

    call_args = mock_classifier.classify_paper.call_args
    concept_index_passed = call_args[0][1]  # second positional arg
    assert "attention mechanism" in concept_index_passed
    assert "backpropagation" in concept_index_passed


def test_empty_concept_index_is_valid(in_memory_store: GraphStore) -> None:
    """Classification proceeds even when the concept index is empty (fresh knowledge bank)."""
    candidates = [_make_scored("001")]
    mock_classifier = MagicMock()
    mock_classifier.classify_paper.return_value = _make_classification()

    results = _run_classification(candidates, in_memory_store, MagicMock(), classifier=mock_classifier)

    assert len(results) == 1
    _, classification = results[0]
    assert not classification.classification_failed


# ---------------------------------------------------------------------------
# Result preservation
# ---------------------------------------------------------------------------


def test_all_candidates_classified() -> None:
    """Every candidate paper appears in the results."""
    store = GraphStore(":memory:")
    candidates = [_make_scored(str(i)) for i in range(5)]
    mock_classifier = MagicMock()
    mock_classifier.classify_paper.return_value = _make_classification()

    results = _run_classification(candidates, store, MagicMock(), classifier=mock_classifier)

    assert len(results) == 5
    assert mock_classifier.classify_paper.call_count == 5


def test_failed_classifications_are_preserved() -> None:
    """Papers where Ollama fails are kept in results with classification_failed=True."""
    store = GraphStore(":memory:")
    candidates = [_make_scored("001"), _make_scored("002"), _make_scored("003")]

    mock_classifier = MagicMock()
    mock_classifier.classify_paper.side_effect = [
        _make_classification(tier=1),
        _make_failed_classification(),
        _make_classification(tier=3),
    ]

    results = _run_classification(candidates, store, MagicMock(), classifier=mock_classifier)

    assert len(results) == 3
    failed = [c for _, c in results if c.classification_failed]
    assert len(failed) == 1


def test_result_order_matches_candidate_order() -> None:
    """Results are returned in the same order as the input candidates."""
    store = GraphStore(":memory:")
    arxiv_ids = ["aaa", "bbb", "ccc"]
    candidates = [_make_scored(aid) for aid in arxiv_ids]
    mock_classifier = MagicMock()
    mock_classifier.classify_paper.return_value = _make_classification()

    results = _run_classification(candidates, store, MagicMock(), classifier=mock_classifier)

    result_ids = [scored.paper.arxiv_id for scored, _ in results]
    assert result_ids == arxiv_ids


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_progress_logged_for_each_paper(
    caplog: pytest.LogCaptureFixture,
    in_memory_store: GraphStore,
) -> None:
    """Progress line 'Classifying paper N of M: {title}' is logged for each paper."""
    candidates = [
        _make_scored("001", score=1.0),
        _make_scored("002", score=0.5),
    ]
    # Override titles for predictable log matching
    candidates[0].paper = _make_paper("001", title="Alpha Paper")
    candidates[1].paper = _make_paper("002", title="Beta Paper")

    mock_classifier = MagicMock()
    mock_classifier.classify_paper.return_value = _make_classification()

    with caplog.at_level(logging.INFO, logger="src.pipeline"):
        _run_classification(candidates, in_memory_store, MagicMock(), classifier=mock_classifier)

    messages = " | ".join(r.message for r in caplog.records)
    assert "1 of 2" in messages
    assert "2 of 2" in messages
    assert "Alpha Paper" in messages
    assert "Beta Paper" in messages


def test_summary_logged_after_classification(
    caplog: pytest.LogCaptureFixture,
    in_memory_store: GraphStore,
) -> None:
    """A summary line is logged after all papers are classified."""
    candidates = [_make_scored("001"), _make_scored("002"), _make_scored("003")]
    mock_classifier = MagicMock()
    mock_classifier.classify_paper.side_effect = [
        _make_classification(tier=1),
        _make_failed_classification(),
        _make_classification(tier=3),
    ]

    with caplog.at_level(logging.INFO, logger="src.pipeline"):
        _run_classification(candidates, in_memory_store, MagicMock(), classifier=mock_classifier)

    messages = " | ".join(r.message for r in caplog.records)
    assert "Classification complete" in messages
    assert "2" in messages   # 2 classified
    assert "1" in messages   # 1 failed


# ---------------------------------------------------------------------------
# Ollama error propagation
# ---------------------------------------------------------------------------


def test_ollama_not_running_aborts_pipeline() -> None:
    """RuntimeError from OllamaClassifier propagates and marks the run as failed."""
    store = GraphStore(":memory:")
    cfg = _make_cfg()

    with (
        patch("src.pipeline.ArxivFetcher") as mock_arxiv_cls,
        patch("src.pipeline.HuggingFaceFetcher") as mock_hf_cls,
        patch("src.pipeline.OllamaClassifier") as mock_classifier_cls,
    ):
        mock_arxiv_cls.return_value.fetch_recent.return_value = [_make_paper("001")]
        mock_hf_cls.return_value.fetch_week.return_value = {}
        mock_classifier_cls.return_value.classify_paper.side_effect = RuntimeError(
            "Ollama is not running. Start with: ollama serve"
        )

        with pytest.raises(RuntimeError, match="Ollama is not running"):
            run_pipeline(store=store, cfg=cfg)

    run = store.get_latest_run()
    assert run is not None
    assert run.status == "failed"
    assert "Ollama is not running" in (run.error_message or "")


def test_model_not_found_aborts_pipeline() -> None:
    """RuntimeError for missing model propagates and marks the run as failed."""
    store = GraphStore(":memory:")
    cfg = _make_cfg()

    with (
        patch("src.pipeline.ArxivFetcher") as mock_arxiv_cls,
        patch("src.pipeline.HuggingFaceFetcher") as mock_hf_cls,
        patch("src.pipeline.OllamaClassifier") as mock_classifier_cls,
    ):
        mock_arxiv_cls.return_value.fetch_recent.return_value = [_make_paper("001")]
        mock_hf_cls.return_value.fetch_week.return_value = {}
        mock_classifier_cls.return_value.classify_paper.side_effect = RuntimeError(
            "Model phi4:14b not found. Pull with: ollama pull phi4:14b"
        )

        with pytest.raises(RuntimeError, match="not found"):
            run_pipeline(store=store, cfg=cfg)

    run = store.get_latest_run()
    assert run.status == "failed"


# ---------------------------------------------------------------------------
# Run record counts
# ---------------------------------------------------------------------------


def test_run_record_counts_classified_and_failed() -> None:
    """papers_classified and papers_failed are written to the weekly_runs record."""
    store = GraphStore(":memory:")
    cfg = _make_cfg()

    with (
        patch("src.pipeline.ArxivFetcher") as mock_arxiv_cls,
        patch("src.pipeline.HuggingFaceFetcher") as mock_hf_cls,
        patch("src.pipeline.OllamaClassifier") as mock_classifier_cls,
        patch("src.pipeline._run_storage_and_rendering", return_value="/tmp/digest.html"),
    ):
        papers = [_make_paper("001"), _make_paper("002"), _make_paper("003")]
        mock_arxiv_cls.return_value.fetch_recent.return_value = papers
        mock_hf_cls.return_value.fetch_week.return_value = {}
        mock_classifier_cls.return_value.classify_paper.side_effect = [
            _make_classification(tier=2),
            _make_failed_classification(),
            _make_classification(tier=4),
        ]

        run_pipeline(store=store, cfg=cfg)

    run = store.get_latest_run()
    assert run.papers_classified == 2
    assert run.papers_failed == 1
