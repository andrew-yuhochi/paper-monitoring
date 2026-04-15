"""Live integration test for ArxivFetcher.

Marked @pytest.mark.slow — not run in the default test suite.
Run explicitly with: pytest -m slow

Makes a real network request to the arXiv API to verify the full
parse pipeline against a known, stable paper ID.
"""

import pytest

from src.integrations.arxiv_client import ArxivFetcher
from src.models.arxiv import ArxivPaper


@pytest.mark.slow
def test_fetch_known_paper_live():
    """Fetch 'Attention Is All You Need' (1706.03762) from the live arXiv API
    and verify that the response parses correctly into an ArxivPaper model."""
    fetcher = ArxivFetcher()
    paper = fetcher.fetch_by_id("1706.03762")

    assert isinstance(paper, ArxivPaper)
    assert paper.arxiv_id == "1706.03762"
    assert "attention" in paper.title.lower()
    assert len(paper.authors) >= 1
    assert paper.primary_category in paper.all_categories
    assert paper.published_date.startswith("2017")
    assert paper.arxiv_url == "https://arxiv.org/abs/1706.03762"
    assert paper.pdf_url == "https://arxiv.org/pdf/1706.03762"
    assert len(paper.abstract) > 50
