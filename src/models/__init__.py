"""Data contracts between all pipeline components."""
from src.models.arxiv import ArxivPaper
from src.models.changes import TrendingTopic, WeeklyChanges
from src.models.classification import ExtractedConcept, PaperAnalysis, PaperClassification
from src.models.graph import AnalysisLinks, DigestEntry, Edge, Node, ScoredPaper, WeeklyRun
from src.models.huggingface import HFPaper
from src.models.search import SearchResult
from src.models.seeding import ChapterText
from src.models.taxonomy import CategoryProposal

__all__ = [
    "ArxivPaper",
    "HFPaper",
    "PaperClassification",
    "PaperAnalysis",
    "ExtractedConcept",
    "Node",
    "Edge",
    "ScoredPaper",
    "WeeklyRun",
    "DigestEntry",
    "AnalysisLinks",
    "ChapterText",
    "SearchResult",
    "WeeklyChanges",
    "TrendingTopic",
    "CategoryProposal",
]
