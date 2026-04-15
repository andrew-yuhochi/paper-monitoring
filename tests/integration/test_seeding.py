"""Integration test: seed 2 landmark papers with mocked Ollama, verify graph state."""
import pytest
from unittest.mock import MagicMock
from src.integrations.arxiv_client import ArxivFetcher
from src.integrations.pdf_extractor import PdfExtractor
from src.models.arxiv import ArxivPaper
from src.models.classification import ExtractedConcept
from src.services.classifier import OllamaClassifier
from src.services.seeder import Seeder
from src.store.graph_store import GraphStore


def _make_paper(arxiv_id: str, title: str) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=f"Abstract for {title}.",
        authors=["Author One"],
        primary_category="cs.LG",
        all_categories=["cs.LG"],
        published_date="2017-06-12",
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def _make_concepts(*names: str) -> list[ExtractedConcept]:
    concepts = []
    for i, name in enumerate(names):
        prereqs = [names[i - 1]] if i > 0 else []
        concepts.append(ExtractedConcept(
            name=name,
            description=f"Description of {name}.",
            domain_tags=["deep learning"],
            prerequisite_concept_names=prereqs,
        ))
    return concepts


@pytest.fixture
def store():
    return GraphStore(":memory:")


@pytest.fixture
def seeder(store):
    mock_arxiv = MagicMock(spec=ArxivFetcher)
    mock_pdf = MagicMock(spec=PdfExtractor)
    mock_classifier = MagicMock(spec=OllamaClassifier)
    return Seeder(
        store=store,
        arxiv_fetcher=mock_arxiv,
        pdf_extractor=mock_pdf,
        classifier=mock_classifier,
    ), mock_arxiv, mock_pdf, mock_classifier


class TestSeederIntegration:

    def test_seed_paper_creates_paper_node(self, seeder, store):
        s, _, _, mock_clf = seeder
        mock_clf.extract_concepts.return_value = []
        paper = _make_paper("1706.03762", "Attention Is All You Need")

        s.seed_paper(paper, "landmark_paper")

        node = store.get_node("paper:1706.03762")
        assert node is not None
        assert node.node_type == "paper"
        assert node.label == "Attention Is All You Need"

    def test_seed_paper_creates_concept_nodes(self, seeder, store):
        s, _, _, mock_clf = seeder
        mock_clf.extract_concepts.return_value = _make_concepts("attention mechanism", "transformer")
        paper = _make_paper("1706.03762", "Attention Is All You Need")

        s.seed_paper(paper, "landmark_paper")

        assert store.get_node("concept:attention_mechanism") is not None
        assert store.get_node("concept:transformer") is not None

    def test_seed_paper_creates_introduces_edges(self, seeder, store):
        s, _, _, mock_clf = seeder
        mock_clf.extract_concepts.return_value = _make_concepts("attention mechanism")
        paper = _make_paper("1706.03762", "Attention Is All You Need")

        s.seed_paper(paper, "landmark_paper")

        edges = store.get_edges_from("paper:1706.03762", "INTRODUCES")
        assert len(edges) == 1
        assert edges[0].target_id == "concept:attention_mechanism"

    def test_flush_prerequisites_creates_edges(self, seeder, store):
        s, _, _, mock_clf = seeder
        # "transformer" has prereq "attention mechanism"
        mock_clf.extract_concepts.return_value = _make_concepts("attention mechanism", "transformer")
        paper = _make_paper("1706.03762", "Attention Is All You Need")

        s.seed_paper(paper, "landmark_paper")
        edges_written = s.flush_prerequisites()

        # transformer -> attention_mechanism (PREREQUISITE_OF)
        edges = store.get_edges_from("concept:transformer", "PREREQUISITE_OF")
        assert len(edges) == 1
        assert edges[0].target_id == "concept:attention_mechanism"
        assert edges_written >= 1

    def test_seed_all_two_papers(self, seeder, store):
        s, mock_arxiv, _, mock_clf = seeder
        paper1 = _make_paper("1706.03762", "Attention Is All You Need")
        paper2 = _make_paper("1512.03385", "Deep Residual Learning")
        mock_arxiv.fetch_batch.side_effect = [
            [paper1],   # landmark papers
            [paper2],   # survey papers
        ]
        mock_clf.extract_concepts.side_effect = [
            _make_concepts("attention mechanism"),
            _make_concepts("residual connection"),
        ]

        summary = s.seed_all(
            landmark_ids=["1706.03762"],
            survey_ids=["1512.03385"],
            textbook_configs=[],
        )

        assert summary["papers_seeded"] == 2
        assert store.get_node("paper:1706.03762") is not None
        assert store.get_node("paper:1512.03385") is not None
        assert store.get_node("concept:attention_mechanism") is not None
        assert store.get_node("concept:residual_connection") is not None

    def test_seed_all_missing_textbook_is_skipped(self, seeder, store, tmp_path):
        s, mock_arxiv, _, mock_clf = seeder
        mock_arxiv.fetch_batch.return_value = []

        missing_pdf = tmp_path / "nonexistent.pdf"
        summary = s.seed_all(
            landmark_ids=[],
            survey_ids=[],
            textbook_configs=[(missing_pdf, [(0, 5, "Missing Textbook, Ch. 1")])],
        )
        # Should not raise; textbook just skipped
        assert summary["concepts_from_textbooks"] == 0

    def test_paper_failure_does_not_abort_seeding(self, seeder, store):
        s, mock_arxiv, _, mock_clf = seeder
        paper1 = _make_paper("1706.03762", "Good Paper")
        paper2 = _make_paper("1512.03385", "Another Good Paper")
        mock_arxiv.fetch_batch.side_effect = [
            [paper1, paper2],
            [],
        ]
        # First paper's concept extraction raises; second succeeds
        mock_clf.extract_concepts.side_effect = [
            RuntimeError("Ollama is not running"),
            _make_concepts("residual connection"),
        ]

        summary = s.seed_all(
            landmark_ids=["1706.03762", "1512.03385"],
            survey_ids=[],
            textbook_configs=[],
        )
        # Second paper still seeded despite first failing
        assert store.get_node("concept:residual_connection") is not None
