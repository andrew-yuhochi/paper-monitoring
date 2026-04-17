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


class PaperAnalysis(BaseModel):
    """Concept-first decomposition of a paper. Replaces PaperClassification for new papers."""
    # Problem
    problem_name: str
    problem_description: str
    is_existing_problem: bool = False

    # Technique
    technique_name: str
    technique_approach: str
    innovation_type: str  # architecture|problem_framing|loss_trick|eval_methodology|dataset|training_technique
    practical_relevance: str
    limitations: str
    is_existing_technique: bool = False

    # Classification
    tier: int | None  # 1-5
    confidence: str | None  # high|medium|low
    reasoning: str | None

    # Evidence
    results_vs_baselines: str

    # Linking
    alternative_technique_names: list[str] = []
    baseline_technique_names: list[str] = []
    foundational_concept_names: list[str] = []

    # Failure handling
    classification_failed: bool = False
    raw_response: str | None = None


class ExtractedConcept(BaseModel):
    name: str
    description: str
    domain_tags: list[str] = []
    prerequisite_concept_names: list[str] = []
