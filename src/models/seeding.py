"""Pydantic models for the seeding phase (Phase 1) of the pipeline."""
from pydantic import BaseModel


class ChapterText(BaseModel):
    text: str
    source_description: str  # e.g. "Goodfellow et al. Deep Learning, Ch. 3"
