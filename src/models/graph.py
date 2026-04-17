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
    edited_by: str = "llm"  # "llm" or "user" — provenance tracking


class Edge(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0
    properties: dict = {}
    edited_by: str = "llm"  # "llm" or "user" — provenance tracking


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


class AnalysisLinks(BaseModel):
    """Summary of nodes and edges created from a PaperAnalysis."""
    problem_node_id: str | None = None
    technique_node_id: str | None = None
    paper_node_id: str
    concepts_linked: int = 0
    baselines_linked: int = 0
    alternatives_linked: int = 0
    problem_is_new: bool = False
    technique_is_new: bool = False


class DigestEntry(BaseModel):
    """Fully resolved paper data ready for template rendering."""
    paper: ArxivPaper
    classification: PaperClassification
    linked_concepts: list[Node] = []     # resolved concept nodes
    hf_upvotes: int = 0
    prefilter_score: float = 0.0
