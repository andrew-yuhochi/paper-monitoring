"""Pydantic models for graph nodes, edges, and pipeline run records."""
from pydantic import BaseModel

from src.models.arxiv import ArxivPaper
from src.models.classification import PaperClassification
from src.models.huggingface import HFPaper


class Node(BaseModel):
    id: str
    node_type: str
    label: str
    properties: dict = {}


class Edge(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0
    properties: dict = {}


class ScoredPaper(BaseModel):
    paper: ArxivPaper
    hf_data: HFPaper | None = None
    score: float


class WeeklyRun(BaseModel):
    id: int
    run_date: str
    started_at: str
    completed_at: str | None = None
    papers_fetched: int = 0
    papers_classified: int = 0
    papers_failed: int = 0
    digest_path: str | None = None
    status: str = "running"
    error_message: str | None = None


class DigestEntry(BaseModel):
    """Fully resolved paper data ready for template rendering."""
    paper: ArxivPaper
    classification: PaperClassification
    linked_concepts: list[Node] = []     # resolved concept nodes
    hf_upvotes: int = 0
    prefilter_score: float = 0.0
