"""Unit tests for the PreFilter scoring service."""

from unittest.mock import MagicMock

import pytest

from src.models.arxiv import ArxivPaper
from src.models.graph import ScoredPaper
from src.models.huggingface import HFPaper
from src.services.prefilter import PreFilter


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
        published_date="2026-04-14",
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def _make_hf(arxiv_id: str, upvotes: int) -> HFPaper:
    return HFPaper(arxiv_id=arxiv_id, title=f"Paper {arxiv_id}", upvotes=upvotes)


def _make_cfg(
    top_n: int = 100,
    upvote_weight: float = 2.0,
    priorities: dict[str, int] | None = None,
) -> MagicMock:
    cfg = MagicMock()
    cfg.prefilter_top_n = top_n
    cfg.prefilter_upvote_weight = upvote_weight
    cfg.prefilter_category_priorities = priorities or {
        "cs.LG": 5,
        "stat.ML": 4,
        "cs.AI": 3,
        "cs.CL": 3,
        "cs.CV": 2,
    }
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_score_formula_with_hf_data() -> None:
    """Paper in cs.LG (priority=5), 10 upvotes, weight=2.0 → score = 25.0."""
    cfg = _make_cfg()
    pf = PreFilter(cfg=cfg)
    paper = _make_paper("0001", primary_category="cs.LG")
    hf = _make_hf("0001", upvotes=10)

    result = pf.score_and_filter([paper], {"0001": hf})

    assert len(result) == 1
    assert result[0].score == 25.0  # 10 * 2.0 + 5


def test_score_formula_no_hf_data() -> None:
    """Paper in cs.LG (priority=5), absent from hf_data → score = 5.0; hf_data is None."""
    cfg = _make_cfg()
    pf = PreFilter(cfg=cfg)
    paper = _make_paper("0002", primary_category="cs.LG")

    result = pf.score_and_filter([paper], {})

    assert len(result) == 1
    assert result[0].score == 5.0   # 0 * 2.0 + 5
    assert result[0].hf_data is None


def test_sort_order_descending() -> None:
    """Three papers; expected scores A=25, C=13, B=5; returned order A, C, B."""
    cfg = _make_cfg()
    pf = PreFilter(cfg=cfg)

    paper_a = _make_paper("A", primary_category="cs.LG")   # 10 upvotes → 10*2+5 = 25
    paper_b = _make_paper("B", primary_category="cs.LG")   # 0 upvotes  → 0*2+5  = 5
    paper_c = _make_paper("C", primary_category="cs.LG")   # 4 upvotes  → 4*2+5  = 13

    hf_data = {
        "A": _make_hf("A", upvotes=10),
        "C": _make_hf("C", upvotes=4),
    }

    result = pf.score_and_filter([paper_a, paper_b, paper_c], hf_data)

    assert [sp.paper.arxiv_id for sp in result] == ["A", "C", "B"]


def test_top_n_limits_output() -> None:
    """5 papers, top_n=3 → only 3 returned, and they are the 3 highest-scoring."""
    cfg = _make_cfg(top_n=5)  # cfg top_n doesn't matter; we pass explicit top_n=3
    pf = PreFilter(cfg=cfg)

    papers = [_make_paper(str(i), primary_category="cs.LG") for i in range(5)]
    # Assign different upvote counts so scores are distinct
    hf_data = {str(i): _make_hf(str(i), upvotes=i * 3) for i in range(5)}
    # Scores: id=0 → 5, id=1 → 11, id=2 → 17, id=3 → 23, id=4 → 29

    result = pf.score_and_filter(papers, hf_data, top_n=3)

    assert len(result) == 3
    assert {sp.paper.arxiv_id for sp in result} == {"4", "3", "2"}


def test_zero_upvote_paper_not_excluded() -> None:
    """Papers with 0 upvotes must not be excluded; category score keeps them alive."""
    cfg = _make_cfg()
    pf = PreFilter(cfg=cfg)

    paper_lg = _make_paper("LG", primary_category="cs.LG")   # score = 0*2+5 = 5
    paper_cv = _make_paper("CV", primary_category="cs.CV")   # score = 0*2+2 = 2

    result = pf.score_and_filter([paper_lg, paper_cv], {}, top_n=10)

    ids = [sp.paper.arxiv_id for sp in result]
    assert "LG" in ids
    assert "CV" in ids
    # cs.LG must rank above cs.CV
    assert ids.index("LG") < ids.index("CV")


def test_category_not_in_priorities_scores_zero() -> None:
    """Paper in cs.RO (not in priorities), no HF data → score = 0.0."""
    cfg = _make_cfg()
    pf = PreFilter(cfg=cfg)
    paper = _make_paper("RO1", primary_category="cs.RO")

    result = pf.score_and_filter([paper], {})

    assert len(result) == 1
    assert result[0].score == 0.0


def test_default_top_n_from_config() -> None:
    """When top_n is not supplied, config.prefilter_top_n is used."""
    cfg = _make_cfg(top_n=3)
    pf = PreFilter(cfg=cfg)

    papers = [_make_paper(str(i)) for i in range(5)]

    result = pf.score_and_filter(papers, {})  # no explicit top_n

    assert len(result) == 3


def test_empty_papers_returns_empty() -> None:
    """Empty input list → empty output list."""
    pf = PreFilter(cfg=_make_cfg())

    result = pf.score_and_filter([], {})

    assert result == []


def test_hf_data_attached_to_scored_paper() -> None:
    """ScoredPaper.hf_data must be the exact HFPaper object passed in hf_data."""
    cfg = _make_cfg()
    pf = PreFilter(cfg=cfg)
    paper = _make_paper("0009", primary_category="cs.LG")
    hf = _make_hf("0009", upvotes=7)

    result = pf.score_and_filter([paper], {"0009": hf})

    assert result[0].hf_data is hf


def test_all_papers_returned_when_fewer_than_top_n() -> None:
    """When there are fewer papers than top_n, all papers are returned."""
    cfg = _make_cfg(top_n=100)
    pf = PreFilter(cfg=cfg)
    papers = [_make_paper(str(i)) for i in range(3)]

    result = pf.score_and_filter(papers, {}, top_n=100)

    assert len(result) == 3
