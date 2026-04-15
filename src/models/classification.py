"""Pydantic models for Ollama classification outputs."""
from pydantic import BaseModel


class PaperClassification(BaseModel):
    tier: int | None                         # 1-5, or None if classification failed
    confidence: str | None                   # "high", "medium", "low"
    reasoning: str | None
    summary: str | None
    key_contributions: list[str] = []
    foundational_concept_names: list[str] = []
    classification_failed: bool = False
    raw_response: str | None = None          # preserved for debugging failures


class ExtractedConcept(BaseModel):
    name: str
    description: str
    domain_tags: list[str] = []
    prerequisite_concept_names: list[str] = []
