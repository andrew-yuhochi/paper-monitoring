"""End-to-end integration test for the paper monitoring pipeline (TASK-016).

Exercises the full data flow:
  seed → ingest (mocked) → classify (mocked) → store → render

No live API calls. No live Ollama. All network and LLM calls are mocked.
Target: completes in under 30 seconds.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.integrations.arxiv_client import ArxivFetcher
from src.integrations.pdf_extractor import PdfExtractor
from src.models.arxiv import ArxivPaper
from src.models.classification import ExtractedConcept, PaperClassification
from src.models.huggingface import HFPaper
from src.pipeline import run_pipeline
from src.services.classifier import OllamaClassifier
from src.services.seeder import Seeder
from src.store.graph_store import GraphStore
from src.utils.normalize import normalize_concept_name

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Canned concept definitions (5 foundational concepts)
# ---------------------------------------------------------------------------

_CONCEPTS = [
    ExtractedConcept(
        name="attention mechanism",
        description="Scaled dot-product attention used in Transformers.",
        domain_tags=["deep learning"],
    ),
    ExtractedConcept(
        name="transformer",
        description="Sequence model based entirely on attention.",
        domain_tags=["deep learning", "nlp"],
    ),
    ExtractedConcept(
        name="neural network",
        description="Layered computational graph of neurons.",
        domain_tags=["deep learning"],
    ),
    ExtractedConcept(
        name="backpropagation",
        description="Gradient computation via reverse-mode autodiff.",
        domain_tags=["deep learning", "optimization"],
    ),
    ExtractedConcept(
        name="gradient descent",
        description="First-order iterative optimisation algorithm.",
        domain_tags=["optimization"],
    ),
]

# ---------------------------------------------------------------------------
# Canned arXiv papers (10 total)
#   Papers e2e_001–e2e_005: cs.LG — these will survive the pre-filter
#   Papers e2e_006–e2e_010: cs.AI — these will be filtered out
# ---------------------------------------------------------------------------

def _make_arxiv_papers() -> list[ArxivPaper]:
    papers = []
    for i in range(1, 11):
        arxiv_id = f"e2e_{i:03d}"
        cat = "cs.LG" if i <= 5 else "cs.AI"
        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=f"E2E Paper {i:03d}",
                abstract=f"Abstract of E2E paper {i:03d}.",
                authors=["Alice", "Bob"],
                primary_category=cat,
                all_categories=[cat],
                published_date="2026-04-14",
                arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            )
        )
    return papers


# ---------------------------------------------------------------------------
# Canned HF upvote data (5 papers — only the cs.LG set)
#   Upvotes: 50, 40, 30, 20, 10 → combined with category priority scores
#   these deterministically select papers 001–005 as the top-5 candidates.
# ---------------------------------------------------------------------------

def _make_hf_data() -> dict[str, HFPaper]:
    upvotes = [50, 40, 30, 20, 10]
    return {
        f"e2e_{i:03d}": HFPaper(
            arxiv_id=f"e2e_{i:03d}",
            title=f"E2E Paper {i:03d}",
            upvotes=upvotes[i - 1],
        )
        for i in range(1, 6)
    }


# ---------------------------------------------------------------------------
# Deterministic classification responses (5 — one per top-5 candidate)
#   Call order matches score-descending order: 001, 002, 003, 004, 005
# ---------------------------------------------------------------------------

def _make_classifications() -> list[PaperClassification]:
    return [
        PaperClassification(
            tier=1,
            confidence="high",
            reasoning="Game-changing architecture.",
            summary="Introduces the Transformer architecture.",
            key_contributions=["Self-attention", "Multi-head attention"],
            foundational_concept_names=["attention mechanism", "transformer"],
        ),
        PaperClassification(
            tier=2,
            confidence="high",
            reasoning="Dominant approach for its era.",
            summary="Demonstrates deep neural network training.",
            key_contributions=["Deep nets", "Gradient flow"],
            foundational_concept_names=["neural network", "backpropagation"],
        ),
        PaperClassification(
            tier=3,
            confidence="medium",
            reasoning="Specialized improvement.",
            summary="Applies gradient-based optimisation in a new domain.",
            key_contributions=["Efficient convergence"],
            foundational_concept_names=["gradient descent", "transformer"],
        ),
        PaperClassification(
            tier=4,
            confidence="low",
            reasoning="Analytical commentary.",
            summary="Analysis of existing optimisation methods.",
            key_contributions=["Comparative study"],
            foundational_concept_names=["backpropagation", "gradient descent"],
        ),
        PaperClassification(
            tier=None,
            confidence=None,
            reasoning=None,
            summary=None,
            classification_failed=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Test config helper
# ---------------------------------------------------------------------------

def _make_test_cfg(digest_output_dir: Path) -> MagicMock:
    """MagicMock config with all real values needed by the pipeline stages."""
    cfg = MagicMock()
    cfg.arxiv_categories = ["cs.LG", "cs.AI"]
    cfg.arxiv_max_results_per_category = 50
    cfg.arxiv_lookback_days = 7
    cfg.arxiv_fetch_delay = 0.0
    cfg.hf_fetch_delay = 0.0
    cfg.prefilter_top_n = 5
    cfg.prefilter_upvote_weight = 2.0
    cfg.prefilter_category_priorities = {"cs.LG": 5, "cs.AI": 3}
    cfg.concept_match_threshold = 0.85
    cfg.digest_output_dir = digest_output_dir
    cfg.template_dir = PROJECT_ROOT / "src" / "templates"
    cfg.db_path = ":memory:"
    return cfg


# ---------------------------------------------------------------------------
# Module-scoped fixture — runs the full pipeline once; all tests share it
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def e2e_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the complete seed → pipeline flow and return results for assertions."""
    tmp_path = tmp_path_factory.mktemp("e2e_pipeline")
    store = GraphStore(":memory:")
    cfg = _make_test_cfg(digest_output_dir=tmp_path / "digests")

    # ------------------------------------------------------------------
    # Phase 1: seed knowledge bank with 5 concepts (mocked Ollama extraction)
    # ------------------------------------------------------------------
    seed_paper = ArxivPaper(
        arxiv_id="seed_001",
        title="Foundational ML Methods",
        abstract="Covers attention, transformers, neural networks, backprop, gradient descent.",
        authors=["Seed Author"],
        primary_category="cs.LG",
        all_categories=["cs.LG"],
        published_date="2020-01-01",
        arxiv_url="https://arxiv.org/abs/seed_001",
        pdf_url="https://arxiv.org/pdf/seed_001",
    )

    mock_seed_arxiv = MagicMock(spec=ArxivFetcher)
    mock_seed_arxiv.fetch_batch.side_effect = [
        [seed_paper],   # landmark papers
        [],             # survey papers
    ]
    mock_seed_clf = MagicMock(spec=OllamaClassifier)
    mock_seed_clf.extract_concepts.return_value = _CONCEPTS

    seeder = Seeder(
        store=store,
        arxiv_fetcher=mock_seed_arxiv,
        pdf_extractor=MagicMock(spec=PdfExtractor),
        classifier=mock_seed_clf,
    )
    seeder.seed_all(
        landmark_ids=["seed_001"],
        survey_ids=[],
        textbook_configs=[],
    )

    # ------------------------------------------------------------------
    # Phase 2: run weekly pipeline with mocked fetchers + classifier
    # ------------------------------------------------------------------
    arxiv_papers = _make_arxiv_papers()
    hf_data = _make_hf_data()
    classifications = _make_classifications()

    with (
        patch("src.pipeline.ArxivFetcher") as mock_arxiv,
        patch("src.pipeline.HuggingFaceFetcher") as mock_hf,
        patch("src.pipeline.OllamaClassifier") as mock_clf,
    ):
        mock_arxiv.return_value.fetch_recent.return_value = arxiv_papers
        mock_hf.return_value.fetch_week.return_value = hf_data
        mock_clf.return_value.classify_paper.side_effect = classifications

        run_pipeline(store=store, cfg=cfg)

    run = store.get_latest_run()
    digest_path = Path(run.digest_path)
    html = digest_path.read_text(encoding="utf-8")

    return {
        "store": store,
        "run": run,
        "digest_path": digest_path,
        "html": html,
        "arxiv_papers": arxiv_papers,
    }


# ---------------------------------------------------------------------------
# Test: knowledge bank seeding
# ---------------------------------------------------------------------------


def test_concepts_seeded_before_pipeline(e2e_result: dict) -> None:
    """Knowledge bank contains the 5 seeded concepts."""
    store: GraphStore = e2e_result["store"]
    concept_index = store.get_concept_index()
    expected = {
        "attention mechanism", "transformer", "neural network",
        "backpropagation", "gradient descent",
    }
    assert expected.issubset(set(concept_index)), (
        f"Missing concepts: {expected - set(concept_index)}"
    )


def test_concept_nodes_have_ids(e2e_result: dict) -> None:
    """Each seeded concept has a node with the correct normalised ID."""
    store: GraphStore = e2e_result["store"]
    for concept in _CONCEPTS:
        norm = normalize_concept_name(concept.name)
        node = store.get_node(f"concept:{norm}")
        assert node is not None, f"concept:{norm} not found in store"
        assert node.node_type == "concept"


# ---------------------------------------------------------------------------
# Test: pipeline run record
# ---------------------------------------------------------------------------


def test_run_completed_successfully(e2e_result: dict) -> None:
    """weekly_runs record has status='completed'."""
    assert e2e_result["run"].status == "completed"


def test_papers_fetched_count(e2e_result: dict) -> None:
    """papers_fetched equals the total raw arXiv count (10)."""
    assert e2e_result["run"].papers_fetched == 10


def test_papers_classified_count(e2e_result: dict) -> None:
    """papers_classified equals successful classifications (4 of 5 candidates)."""
    assert e2e_result["run"].papers_classified == 4


def test_papers_failed_count(e2e_result: dict) -> None:
    """papers_failed equals 1 (the deliberately failed classification)."""
    assert e2e_result["run"].papers_failed == 1


def test_digest_path_recorded_in_run(e2e_result: dict) -> None:
    """weekly_runs record stores the path to the rendered digest."""
    assert e2e_result["run"].digest_path is not None
    assert e2e_result["digest_path"].exists()


# ---------------------------------------------------------------------------
# Test: pre-filter selection
# ---------------------------------------------------------------------------


def test_only_top5_candidates_classified(e2e_result: dict) -> None:
    """Pre-filter reduced 10 papers to top 5; only those are stored in the graph."""
    store: GraphStore = e2e_result["store"]
    # The 5 cs.LG papers with HF upvotes should be stored
    for i in range(1, 6):
        assert store.paper_exists(f"e2e_{i:03d}"), f"e2e_{i:03d} not in graph"


def test_low_score_papers_not_stored(e2e_result: dict) -> None:
    """The 5 cs.AI papers (lower score) did not survive the pre-filter."""
    store: GraphStore = e2e_result["store"]
    for i in range(6, 11):
        assert not store.paper_exists(f"e2e_{i:03d}"), f"e2e_{i:03d} should not be in graph"


# ---------------------------------------------------------------------------
# Test: graph storage
# ---------------------------------------------------------------------------


def test_paper_nodes_upserted(e2e_result: dict) -> None:
    """All 5 classified papers have paper nodes in the graph."""
    store: GraphStore = e2e_result["store"]
    for i in range(1, 6):
        node = store.get_node(f"paper:e2e_{i:03d}")
        assert node is not None
        assert node.node_type == "paper"


def test_paper_properties_include_tier(e2e_result: dict) -> None:
    """Paper node properties include the tier from classification."""
    store: GraphStore = e2e_result["store"]
    node = store.get_node("paper:e2e_001")
    assert node is not None
    assert node.properties.get("tier") == 1


def test_paper_properties_include_run_date(e2e_result: dict) -> None:
    """Paper node properties include run_date for digest reconstruction."""
    store: GraphStore = e2e_result["store"]
    node = store.get_node("paper:e2e_002")
    assert node is not None
    assert "run_date" in node.properties


def test_builds_on_edges_for_paper_001(e2e_result: dict) -> None:
    """Paper e2e_001 has BUILDS_ON edges to at least 2 concepts."""
    store: GraphStore = e2e_result["store"]
    edges = store.get_edges_from("paper:e2e_001", "BUILDS_ON")
    assert len(edges) >= 2, f"Expected ≥2 BUILDS_ON edges, got {len(edges)}"


def test_builds_on_edges_for_paper_002(e2e_result: dict) -> None:
    """Paper e2e_002 has BUILDS_ON edges to at least 2 concepts."""
    store: GraphStore = e2e_result["store"]
    edges = store.get_edges_from("paper:e2e_002", "BUILDS_ON")
    assert len(edges) >= 2, f"Expected ≥2 BUILDS_ON edges, got {len(edges)}"


def test_builds_on_targets_are_concept_nodes(e2e_result: dict) -> None:
    """BUILDS_ON edge targets are concept nodes in the knowledge bank."""
    store: GraphStore = e2e_result["store"]
    edges = store.get_edges_from("paper:e2e_001", "BUILDS_ON")
    for edge in edges:
        target = store.get_node(edge.target_id)
        assert target is not None
        assert target.node_type == "concept"


def test_failed_paper_node_still_stored(e2e_result: dict) -> None:
    """A paper whose classification failed is still stored as a paper node."""
    store: GraphStore = e2e_result["store"]
    assert store.paper_exists("e2e_005")
    node = store.get_node("paper:e2e_005")
    assert node is not None
    assert node.properties.get("classification_failed") is True


def test_failed_paper_has_no_builds_on_edges(e2e_result: dict) -> None:
    """Failed classification → no BUILDS_ON edges created."""
    store: GraphStore = e2e_result["store"]
    edges = store.get_edges_from("paper:e2e_005", "BUILDS_ON")
    assert len(edges) == 0


# ---------------------------------------------------------------------------
# Test: HTML digest structure
# ---------------------------------------------------------------------------


def test_digest_file_exists(e2e_result: dict) -> None:
    """Digest HTML file was written to the output directory."""
    assert e2e_result["digest_path"].exists()
    assert e2e_result["digest_path"].suffix == ".html"


def test_digest_contains_all_paper_titles(e2e_result: dict) -> None:
    """All 5 classified paper titles appear in the digest HTML."""
    html = e2e_result["html"]
    for i in range(1, 6):
        assert f"E2E Paper {i:03d}" in html, f"'E2E Paper {i:03d}' not found in digest"


def test_digest_tier1_section_present(e2e_result: dict) -> None:
    """Tier 1 section is present (expanded, in a <section> element)."""
    html = e2e_result["html"]
    assert "Tier 1" in html


def test_digest_tier2_section_present(e2e_result: dict) -> None:
    """Tier 2 section is present."""
    html = e2e_result["html"]
    assert "Tier 2" in html


def test_digest_collapsed_tiers_present(e2e_result: dict) -> None:
    """T3 and T4 collapsed sections are present."""
    html = e2e_result["html"]
    assert "Tier 3" in html
    assert "Tier 4" in html
    assert "<details" in html


def test_digest_failures_section_present(e2e_result: dict) -> None:
    """Classification Failures section present for the 1 failed paper."""
    html = e2e_result["html"]
    assert "Classification Failures" in html


def test_digest_concept_badges_rendered(e2e_result: dict) -> None:
    """Concept badges appear in the digest for linked papers."""
    html = e2e_result["html"]
    assert 'class="concept-badge"' in html


def test_digest_concept_names_in_badges(e2e_result: dict) -> None:
    """At least one seeded concept name appears as a badge label."""
    html = e2e_result["html"]
    # Paper 001 is linked to "attention mechanism" and "transformer"
    assert "attention mechanism" in html or "transformer" in html
