"""Pydantic model for arXiv paper metadata."""
from pydantic import BaseModel


class ArxivPaper(BaseModel):
    arxiv_id: str                        # e.g. "1706.03762"
    title: str
    abstract: str
    authors: list[str]
    primary_category: str
    all_categories: list[str]
    published_date: str                  # ISO date string
    updated_date: str | None = None
    arxiv_url: str
    pdf_url: str
    comment: str | None = None           # arxiv:comment field
