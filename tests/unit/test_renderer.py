"""Unit tests for DigestRenderer (TASK-014)."""

from datetime import date
from pathlib import Path

import pytest

from src.models.arxiv import ArxivPaper
from src.models.classification import PaperClassification
from src.models.graph import DigestEntry, Node
from src.services.renderer import DigestRenderer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture
def renderer() -> DigestRenderer:
    return DigestRenderer(template_dir=PROJECT_ROOT / "src" / "templates")


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    return tmp_path / "digests"


def _make_arxiv_paper(arxiv_id: str = "2604.00001", title: str = "Test Paper") -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=title,
        abstract="An abstract.",
        authors=["Alice", "Bob"],
        primary_category="cs.LG",
        all_categories=["cs.LG"],
        published_date=date.today().isoformat(),
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def _make_classification(
    tier: int = 2,
    confidence: str = "high",
    summary: str = "A concise summary.",
    key_contributions: list[str] | None = None,
    foundational_concept_names: list[str] | None = None,
    classification_failed: bool = False,
) -> PaperClassification:
    return PaperClassification(
        tier=tier if not classification_failed else None,
        confidence=confidence if not classification_failed else None,
        reasoning="Some reasoning.",
        summary=summary if not classification_failed else None,
        key_contributions=key_contributions or ["Contribution A"],
        foundational_concept_names=foundational_concept_names or [],
        classification_failed=classification_failed,
    )


def _make_concept_node(label: str, description: str = "") -> Node:
    return Node(
        id=f"concept:{label.replace(' ', '_')}",
        node_type="concept",
        label=label,
        properties={"description": description} if description else {},
    )


def _make_entry(
    arxiv_id: str = "2604.00001",
    title: str = "Test Paper",
    tier: int = 2,
    concepts: list[Node] | None = None,
    hf_upvotes: int = 0,
    classification_failed: bool = False,
) -> DigestEntry:
    return DigestEntry(
        paper=_make_arxiv_paper(arxiv_id, title),
        classification=_make_classification(tier=tier, classification_failed=classification_failed),
        linked_concepts=concepts or [],
        hf_upvotes=hf_upvotes,
        prefilter_score=1.0,
    )


# ---------------------------------------------------------------------------
# Basic rendering
# ---------------------------------------------------------------------------


def test_render_returns_html_path(renderer: DigestRenderer, tmp_output: Path) -> None:
    """render() returns a Path pointing to the written HTML file."""
    entries = [_make_entry()]
    path = renderer.render(entries, "2026-04-15", tmp_output)

    assert isinstance(path, Path)
    assert path.name == "2026-04-15.html"
    assert path.exists()


def test_output_dir_created_if_missing(renderer: DigestRenderer, tmp_path: Path) -> None:
    """render() creates output_dir if it does not exist."""
    output_dir = tmp_path / "a" / "b" / "digests"
    assert not output_dir.exists()

    renderer.render([], "2026-04-15", output_dir)
    assert output_dir.exists()


def test_run_date_appears_in_output(renderer: DigestRenderer, tmp_output: Path) -> None:
    """The run date appears in the rendered HTML title and heading."""
    html = renderer.render([], "2026-04-15", tmp_output).read_text()
    assert "2026-04-15" in html


def test_paper_count_in_header(renderer: DigestRenderer, tmp_output: Path) -> None:
    """Total paper count is reflected in the header."""
    entries = [_make_entry(arxiv_id=str(i)) for i in range(3)]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()
    assert "3 papers" in html


# ---------------------------------------------------------------------------
# Empty digest
# ---------------------------------------------------------------------------


def test_empty_digest_message(renderer: DigestRenderer, tmp_output: Path) -> None:
    """Empty entry list renders the 'No papers found this week' message."""
    html = renderer.render([], "2026-04-15", tmp_output).read_text()
    assert "No papers found this week" in html


def test_empty_digest_no_tier_sections(renderer: DigestRenderer, tmp_output: Path) -> None:
    """No tier section headers when there are no entries."""
    html = renderer.render([], "2026-04-15", tmp_output).read_text()
    assert "Tier 1" not in html
    assert "Tier 2" not in html


# ---------------------------------------------------------------------------
# Tier grouping
# ---------------------------------------------------------------------------


def test_tier1_and_tier2_in_expanded_sections(renderer: DigestRenderer, tmp_output: Path) -> None:
    """T1 and T2 papers appear in <section> elements (expanded by default)."""
    entries = [
        _make_entry(arxiv_id="001", title="Game Changer Paper", tier=1),
        _make_entry(arxiv_id="002", title="Clear Winner Paper", tier=2),
    ]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()

    # Tier headings present
    assert "Tier 1" in html
    assert "Tier 2" in html
    # Both paper titles present
    assert "Game Changer Paper" in html
    assert "Clear Winner Paper" in html
    # T1/T2 sections use <section>, not <details>
    assert "<section" in html


def test_tier3_tier4_tier5_in_details_elements(renderer: DigestRenderer, tmp_output: Path) -> None:
    """T3, T4, T5 papers appear inside <details> elements (collapsed by default)."""
    entries = [
        _make_entry(arxiv_id="003", title="Specialized Paper", tier=3),
        _make_entry(arxiv_id="004", title="Argumentative Paper", tier=4),
        _make_entry(arxiv_id="005", title="Survey Paper", tier=5),
    ]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()

    assert "Tier 3" in html
    assert "Tier 4" in html
    assert "Tier 5" in html
    assert "<details" in html


def test_failures_at_bottom_in_details(renderer: DigestRenderer, tmp_output: Path) -> None:
    """Classification failures appear in a <details> section at the bottom."""
    entries = [
        _make_entry(arxiv_id="001", tier=2),
        _make_entry(arxiv_id="002", classification_failed=True),
    ]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()

    assert "Classification Failures" in html
    # Failures section uses <details>
    failures_idx = html.index("Classification Failures")
    details_idx = html.rindex("<details", 0, failures_idx)
    assert details_idx >= 0


def test_tier5_seed_note_present(renderer: DigestRenderer, tmp_output: Path) -> None:
    """T5 paper cards include the 'Consider adding to knowledge bank' note."""
    entries = [_make_entry(arxiv_id="survey001", tier=5)]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()

    assert "Consider adding to knowledge bank" in html
    assert "survey001" in html


def test_tier_badges_present(renderer: DigestRenderer, tmp_output: Path) -> None:
    """Tier badges (e.g. 'T1', 'T2') appear in the card HTML."""
    entries = [
        _make_entry(arxiv_id="001", tier=1),
        _make_entry(arxiv_id="002", tier=2),
    ]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()
    assert "T1" in html
    assert "T2" in html


# ---------------------------------------------------------------------------
# Paper card content
# ---------------------------------------------------------------------------


def test_paper_title_linked_to_arxiv(renderer: DigestRenderer, tmp_output: Path) -> None:
    """Paper title is an anchor pointing to the arXiv URL."""
    entries = [_make_entry(arxiv_id="1706.03762", title="Attention Is All You Need")]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()

    assert "Attention Is All You Need" in html
    assert "https://arxiv.org/abs/1706.03762" in html


def test_paper_authors_present(renderer: DigestRenderer, tmp_output: Path) -> None:
    """Author names appear in the rendered card."""
    entries = [_make_entry()]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()
    assert "Alice" in html
    assert "Bob" in html


def test_pdf_link_present(renderer: DigestRenderer, tmp_output: Path) -> None:
    """PDF link appears in the paper card."""
    entries = [_make_entry(arxiv_id="1706.03762")]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()
    assert "https://arxiv.org/pdf/1706.03762" in html


def test_summary_present(renderer: DigestRenderer, tmp_output: Path) -> None:
    """Classification summary is present in the rendered card."""
    entries = [_make_entry()]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()
    assert "A concise summary." in html


def test_key_contributions_present(renderer: DigestRenderer, tmp_output: Path) -> None:
    """Key contributions appear as list items in the card."""
    entries = [_make_entry()]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()
    assert "Contribution A" in html


def test_hf_upvotes_shown_when_nonzero(renderer: DigestRenderer, tmp_output: Path) -> None:
    """HF upvote count is shown when > 0."""
    entries = [_make_entry(hf_upvotes=42)]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()
    assert "42" in html


# ---------------------------------------------------------------------------
# Concept badges
# ---------------------------------------------------------------------------


def test_concept_badges_rendered(renderer: DigestRenderer, tmp_output: Path) -> None:
    """Linked concept labels appear as concept-badge spans."""
    concepts = [
        _make_concept_node("attention mechanism", "Mechanism in transformers."),
        _make_concept_node("backpropagation"),
    ]
    entries = [_make_entry(concepts=concepts)]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()

    assert "attention mechanism" in html
    assert "backpropagation" in html
    assert 'class="concept-badge"' in html


def test_concept_badge_hover_description(renderer: DigestRenderer, tmp_output: Path) -> None:
    """Concept badge title attribute contains the description for hover text."""
    concepts = [_make_concept_node("attention mechanism", "The scaled dot-product attention.")]
    entries = [_make_entry(concepts=concepts)]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()

    assert "The scaled dot-product attention." in html


def test_no_concept_badges_when_empty(renderer: DigestRenderer, tmp_output: Path) -> None:
    """No concept-badge section rendered when linked_concepts is empty."""
    entries = [_make_entry(concepts=[])]
    html = renderer.render(entries, "2026-04-15", tmp_output).read_text()
    assert 'class="concept-badge"' not in html
