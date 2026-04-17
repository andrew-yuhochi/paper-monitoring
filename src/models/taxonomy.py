"""Pydantic models for auto-generated concept taxonomy."""
from pydantic import BaseModel


class CategoryProposal(BaseModel):
    """A proposed category grouping concepts and techniques."""
    category_name: str
    category_description: str
    concept_names: list[str]    # concepts that belong to this category
    technique_names: list[str]  # techniques that belong to this category
