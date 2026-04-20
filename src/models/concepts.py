# Pydantic models for the concept-first schema (TASK-M1-001).
# These models mirror the 8 new SQLite tables added in graph_store.py.
# Do NOT modify the existing models in graph.py — legacy pipeline uses them.

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Valid enum values
# ---------------------------------------------------------------------------

VALID_RELATIONSHIP_TYPES = (
    "BUILDS_ON",
    "ADDRESSES",
    "ALTERNATIVE_TO",
    "BASELINE_OF",
    "PREREQUISITE_OF",
    "INTRODUCES",
    "BELONGS_TO",
)

VALID_CONCEPT_TYPES = ("Algorithm", "Technique", "Framework", "Concept", "Mechanism", "Problem", "Category")

VALID_LINK_TYPES = ("INTRODUCES", "USES", "SURVEYS", "EVALUATES")

VALID_CHANNELS = ("linkedin", "youtube", "newsletter", "other")

# ---------------------------------------------------------------------------
# Core concept-first models
# ---------------------------------------------------------------------------


class Concept(BaseModel):
    """A first-class ML concept node in the knowledge graph."""

    name: str
    concept_type: Literal["Algorithm", "Technique", "Framework", "Concept", "Mechanism", "Problem", "Category"] = "Concept"
    what_it_is: str = ""
    what_problem_it_solves: str = ""
    innovation_chain: list[dict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    introduced_year: int | None = None
    domain_tags: list[str] = Field(default_factory=list)
    source: str = "manual"  # wikidata | wikipedia | arxiv | manual | textbook
    source_refs: list[str] = Field(default_factory=list)
    content_angles: list[str] = Field(default_factory=list)
    user_id: str = "default"


class ConceptRelationship(BaseModel):
    """A typed, labeled directed edge between two concepts."""

    from_concept: str  # source concept name
    to_concept: str  # target concept name
    relationship_type: Literal[
        "BUILDS_ON",
        "ADDRESSES",
        "ALTERNATIVE_TO",
        "BASELINE_OF",
        "PREREQUISITE_OF",
        "INTRODUCES",
        "BELONGS_TO",
    ]
    label: str = ""  # narrative prose describing the relationship
    source_ref: str | None = None
    user_id: str = "default"


class PaperRecord(BaseModel):
    """A paper record in the concept-first papers table."""

    arxiv_id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    published_date: str = ""  # ISO date
    source: str = "arxiv"
    user_id: str = "default"


class PaperConceptLink(BaseModel):
    """A link between a paper and a concept it introduces, uses, surveys, or evaluates."""

    arxiv_id: str
    concept_name: str
    link_type: Literal["INTRODUCES", "USES", "SURVEYS", "EVALUATES"]
    confidence: float = 1.0
    user_id: str = "default"


class CitationSnapshot(BaseModel):
    """A point-in-time citation count for a paper (for trend resurrection)."""

    arxiv_id: str
    snapshot_date: str  # ISO date
    citation_count: int
    hf_model_count: int = 0
    user_id: str = "default"


class ConceptQuery(BaseModel):
    """A logged concept exploration query (commercial signal instrument)."""

    concept_name: str
    queried_at: Optional[str] = None  # ISO datetime; None = use DB default
    export_format: str = "markdown"
    user_id: str = "default"


class ContentPublication(BaseModel):
    """A logged content publication derived from a concept (commercial signal instrument)."""

    concept_name: str
    channel: str  # linkedin | youtube | newsletter | other
    published_at: str = ""  # ISO datetime; empty = use DB default
    url: str | None = None
    user_id: str = "default"


class ResurrectionCandidate(BaseModel):
    """A paper flagged as a trend-resurrection candidate."""

    arxiv_id: str
    concept_name: str
    delta_citations: int
    weeks_observed: int
    user_id: str = "default"
