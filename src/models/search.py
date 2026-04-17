"""Pydantic models for search results."""
from pydantic import BaseModel


class SearchResult(BaseModel):
    """A single search result with field-priority ranking score."""
    node_id: str
    node_type: str
    label: str
    score: float
    properties: dict = {}
    match_field: str  # which field matched: "label", "approach", "description"
