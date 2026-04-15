"""Unit tests for all Pydantic data models."""
import pytest
from pydantic import ValidationError

from src.models.arxiv import ArxivPaper
from src.models.classification import ExtractedConcept, PaperClassification
from src.models.graph import DigestEntry, Edge, Node, ScoredPaper, WeeklyRun
from src.models.huggingface import HFPaper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def arxiv_paper_data() -> dict:
    return {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models...",
        "authors": ["Vaswani, Ashish", "Shazeer, Noam"],
        "primary_category": "cs.CL",
        "all_categories": ["cs.CL", "cs.LG"],
        "published_date": "2017-06-12",
        "arxiv_url": "https://arxiv.org/abs/1706.03762",
        "pdf_url": "https://arxiv.org/pdf/1706.03762",
    }


@pytest.fixture
def arxiv_paper(arxiv_paper_data) -> ArxivPaper:
    return ArxivPaper(**arxiv_paper_data)


# ---------------------------------------------------------------------------
# ArxivPaper
# ---------------------------------------------------------------------------

class TestArxivPaper:
    def test_construction(self, arxiv_paper_data):
        paper = ArxivPaper(**arxiv_paper_data)
        assert paper.arxiv_id == "1706.03762"
        assert paper.title == "Attention Is All You Need"
        assert paper.authors == ["Vaswani, Ashish", "Shazeer, Noam"]
        assert paper.all_categories == ["cs.CL", "cs.LG"]

    def test_optional_defaults_are_none(self, arxiv_paper_data):
        paper = ArxivPaper(**arxiv_paper_data)
        assert paper.updated_date is None
        assert paper.comment is None

    def test_optional_fields_accepted(self, arxiv_paper_data):
        arxiv_paper_data["updated_date"] = "2017-12-06"
        arxiv_paper_data["comment"] = "12 pages, 5 figures"
        paper = ArxivPaper(**arxiv_paper_data)
        assert paper.updated_date == "2017-12-06"
        assert paper.comment == "12 pages, 5 figures"

    def test_missing_required_field_raises(self, arxiv_paper_data):
        del arxiv_paper_data["arxiv_id"]
        with pytest.raises(ValidationError):
            ArxivPaper(**arxiv_paper_data)

    def test_missing_title_raises(self, arxiv_paper_data):
        del arxiv_paper_data["title"]
        with pytest.raises(ValidationError):
            ArxivPaper(**arxiv_paper_data)

    def test_missing_pdf_url_raises(self, arxiv_paper_data):
        del arxiv_paper_data["pdf_url"]
        with pytest.raises(ValidationError):
            ArxivPaper(**arxiv_paper_data)


# ---------------------------------------------------------------------------
# HFPaper
# ---------------------------------------------------------------------------

class TestHFPaper:
    def test_construction_minimal(self):
        paper = HFPaper(arxiv_id="1706.03762", title="Attention Is All You Need")
        assert paper.arxiv_id == "1706.03762"
        assert paper.title == "Attention Is All You Need"

    def test_defaults(self):
        paper = HFPaper(arxiv_id="1706.03762", title="Some Paper")
        assert paper.upvotes == 0
        assert paper.ai_keywords == []
        assert paper.ai_summary is None
        assert paper.num_comments == 0

    def test_all_fields(self):
        paper = HFPaper(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            upvotes=512,
            ai_keywords=["transformers", "attention"],
            ai_summary="Introduces the transformer architecture.",
            num_comments=34,
        )
        assert paper.upvotes == 512
        assert paper.ai_keywords == ["transformers", "attention"]
        assert paper.num_comments == 34

    def test_missing_arxiv_id_raises(self):
        with pytest.raises(ValidationError):
            HFPaper(title="Some Paper")

    def test_missing_title_raises(self):
        with pytest.raises(ValidationError):
            HFPaper(arxiv_id="1706.03762")

    def test_list_defaults_are_independent(self):
        """Mutable defaults must not be shared across instances."""
        p1 = HFPaper(arxiv_id="111", title="A")
        p2 = HFPaper(arxiv_id="222", title="B")
        p1.ai_keywords.append("nlp")
        assert p2.ai_keywords == []


# ---------------------------------------------------------------------------
# PaperClassification
# ---------------------------------------------------------------------------

class TestPaperClassification:
    def test_successful_classification(self):
        pc = PaperClassification(
            tier=1,
            confidence="high",
            reasoning="Strong foundational contribution.",
            summary="Introduces the transformer.",
            key_contributions=["Multi-head attention", "Positional encoding"],
            foundational_concept_names=["attention mechanism"],
        )
        assert pc.tier == 1
        assert pc.confidence == "high"
        assert pc.classification_failed is False
        assert pc.raw_response is None

    def test_failed_classification_shape(self):
        pc = PaperClassification(
            tier=None,
            confidence=None,
            reasoning=None,
            summary=None,
            classification_failed=True,
            raw_response="<garbage ollama output>",
        )
        assert pc.tier is None
        assert pc.classification_failed is True
        assert pc.raw_response == "<garbage ollama output>"

    def test_defaults(self):
        pc = PaperClassification(tier=3, confidence="medium", reasoning="OK", summary="Fine paper.")
        assert pc.key_contributions == []
        assert pc.foundational_concept_names == []
        assert pc.classification_failed is False
        assert pc.raw_response is None

    def test_tier_none_allowed(self):
        pc = PaperClassification(tier=None, confidence=None, reasoning=None, summary=None)
        assert pc.tier is None


# ---------------------------------------------------------------------------
# ExtractedConcept
# ---------------------------------------------------------------------------

class TestExtractedConcept:
    def test_construction(self):
        concept = ExtractedConcept(
            name="attention mechanism",
            description="Allows the model to focus on relevant parts of the input.",
        )
        assert concept.name == "attention mechanism"
        assert concept.domain_tags == []
        assert concept.prerequisite_concept_names == []

    def test_full_construction(self):
        concept = ExtractedConcept(
            name="multi-head attention",
            description="Runs attention multiple times in parallel.",
            domain_tags=["nlp", "deep learning"],
            prerequisite_concept_names=["attention mechanism", "linear algebra"],
        )
        assert concept.domain_tags == ["nlp", "deep learning"]
        assert len(concept.prerequisite_concept_names) == 2

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            ExtractedConcept(description="Some concept.")

    def test_missing_description_raises(self):
        with pytest.raises(ValidationError):
            ExtractedConcept(name="attention mechanism")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class TestNode:
    def test_construction(self):
        node = Node(id="concept:attention_mechanism", node_type="concept", label="attention mechanism")
        assert node.id == "concept:attention_mechanism"
        assert node.node_type == "concept"
        assert node.properties == {}

    def test_properties_accepted(self):
        node = Node(
            id="paper:1706.03762",
            node_type="paper",
            label="Attention Is All You Need",
            properties={"tier": 1, "run_date": "2026-04-14"},
        )
        assert node.properties["tier"] == 1

    def test_missing_required_raises(self):
        with pytest.raises(ValidationError):
            Node(node_type="concept", label="attention")  # missing id

    def test_properties_default_independent(self):
        n1 = Node(id="a", node_type="concept", label="A")
        n2 = Node(id="b", node_type="concept", label="B")
        n1.properties["x"] = 1
        assert n2.properties == {}


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

class TestEdge:
    def test_construction(self):
        edge = Edge(
            source_id="paper:1706.03762",
            target_id="concept:attention_mechanism",
            relationship_type="BUILDS_ON",
        )
        assert edge.weight == 1.0
        assert edge.properties == {}

    def test_custom_weight(self):
        edge = Edge(
            source_id="concept:transformer",
            target_id="concept:attention_mechanism",
            relationship_type="PREREQUISITE_OF",
            weight=0.9,
        )
        assert edge.weight == 0.9

    def test_missing_source_raises(self):
        with pytest.raises(ValidationError):
            Edge(target_id="concept:x", relationship_type="BUILDS_ON")


# ---------------------------------------------------------------------------
# ScoredPaper
# ---------------------------------------------------------------------------

class TestScoredPaper:
    def test_construction_without_hf(self, arxiv_paper):
        sp = ScoredPaper(paper=arxiv_paper, score=7.5)
        assert sp.score == 7.5
        assert sp.hf_data is None

    def test_construction_with_hf(self, arxiv_paper):
        hf = HFPaper(arxiv_id="1706.03762", title="Attention Is All You Need", upvotes=100)
        sp = ScoredPaper(paper=arxiv_paper, hf_data=hf, score=12.0)
        assert sp.hf_data.upvotes == 100

    def test_missing_score_raises(self, arxiv_paper):
        with pytest.raises(ValidationError):
            ScoredPaper(paper=arxiv_paper)


# ---------------------------------------------------------------------------
# WeeklyRun
# ---------------------------------------------------------------------------

class TestWeeklyRun:
    def test_construction(self):
        run = WeeklyRun(id=1, run_date="2026-04-11", started_at="2026-04-11T18:00:00")
        assert run.status == "running"
        assert run.papers_fetched == 0
        assert run.completed_at is None
        assert run.error_message is None

    def test_completed_run(self):
        run = WeeklyRun(
            id=2,
            run_date="2026-04-11",
            started_at="2026-04-11T18:00:00",
            completed_at="2026-04-11T18:07:32",
            papers_fetched=200,
            papers_classified=50,
            papers_failed=3,
            digest_path="digests/2026-04-11.html",
            status="completed",
        )
        assert run.status == "completed"
        assert run.papers_classified == 50

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            WeeklyRun(run_date="2026-04-11", started_at="2026-04-11T18:00:00")


# ---------------------------------------------------------------------------
# DigestEntry
# ---------------------------------------------------------------------------

class TestDigestEntry:
    def test_construction(self, arxiv_paper):
        pc = PaperClassification(tier=1, confidence="high", reasoning="Top paper.", summary="Summary.")
        entry = DigestEntry(paper=arxiv_paper, classification=pc)
        assert entry.linked_concepts == []
        assert entry.hf_upvotes == 0
        assert entry.prefilter_score == 0.0

    def test_with_linked_concepts(self, arxiv_paper):
        pc = PaperClassification(tier=2, confidence="medium", reasoning="Good.", summary="Fine.")
        node = Node(id="concept:attention_mechanism", node_type="concept", label="attention mechanism")
        entry = DigestEntry(paper=arxiv_paper, classification=pc, linked_concepts=[node], hf_upvotes=42)
        assert len(entry.linked_concepts) == 1
        assert entry.hf_upvotes == 42

    def test_missing_classification_raises(self, arxiv_paper):
        with pytest.raises(ValidationError):
            DigestEntry(paper=arxiv_paper)
