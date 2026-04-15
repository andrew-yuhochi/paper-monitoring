"""Pydantic model for HuggingFace Daily Papers data."""
from pydantic import BaseModel


class HFPaper(BaseModel):
    arxiv_id: str
    title: str
    upvotes: int = 0
    ai_keywords: list[str] = []
    ai_summary: str | None = None
    num_comments: int = 0
