"""Pydantic models for weekly change detection and trend tracking."""
from pydantic import BaseModel


class WeeklyChanges(BaseModel):
    """Summary of graph changes detected in the past week."""
    new_techniques: list[dict] = []     # [{name, problem, paper_title}]
    new_problems: list[dict] = []       # [{name, technique_count}]
    new_edges: dict = {}                # {relationship_type: count}
    summary: str = ""                   # Human-readable summary


class TrendingTopic(BaseModel):
    """A concept or problem gaining traction over a rolling window."""
    node_id: str
    node_type: str
    label: str
    new_connections_this_week: int
    new_connections_4_weeks: int
    trend_score: float
