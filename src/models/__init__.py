"""Data contracts between all pipeline components."""
from src.models.arxiv import ArxivPaper
from src.models.classification import ExtractedConcept, PaperClassification
from src.models.graph import DigestEntry, Edge, Node, ScoredPaper, WeeklyRun
from src.models.huggingface import HFPaper
from src.models.seeding import ChapterText

__all__ = [
    "ArxivPaper",
    "HFPaper",
    "PaperClassification",
    "ExtractedConcept",
    "Node",
    "Edge",
    "ScoredPaper",
    "WeeklyRun",
    "DigestEntry",
    "ChapterText",
]
