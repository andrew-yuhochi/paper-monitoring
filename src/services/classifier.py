# OllamaClassifier service: paper classification and concept extraction.
# Builds prompts, calls OllamaClient, and maps responses to Pydantic models.
# The OllamaClient handles all retry logic; this module handles prompt construction
# and response mapping only.

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.config import Settings, settings as default_settings
from src.integrations.ollama_client import OllamaClient
from src.models.arxiv import ArxivPaper
from src.models.classification import ExtractedConcept, PaperClassification

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal response models — match the exact JSON structure Ollama returns.
# These are NOT part of the public API.
# ---------------------------------------------------------------------------


class _ClassificationResponse(BaseModel):
    """Intermediate model matching the LLM's JSON output for paper classification."""

    tier: int
    confidence: str
    reasoning: str
    summary: str
    key_contributions: list[str] = []
    foundational_concept_names: list[str] = []


class _ConceptsResponse(BaseModel):
    """Intermediate model matching the LLM's JSON output for concept extraction."""

    concepts: list[ExtractedConcept]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CLASSIFICATION_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert ML researcher classifying research papers by their historical importance and contribution to the field. Use this 5-tier taxonomy:

Tier 1 (Game-changers): Introduces a fundamentally new method or paradigm that creates a new branch of knowledge. Examples: Random Forest, CNN, Transformer architecture.

Tier 2 (Clear winners): A clear improvement that becomes the dominant approach in its area, displacing prior methods. Examples: LSTM over vanilla RNN, ResNet over plain deep CNNs, BERT for NLP pre-training.

Tier 3 (Specialized improvements): A meaningful improvement from a specialized angle — domain-specific optimizations, architectural tweaks, or niche enhancements. Examples: applying attention to a specific domain, improved training tricks.

Tier 4 (Argumentative): Analysis, critique, or argument from a specialized angle. Examples: ablation studies, position papers, empirical comparisons, critique of existing methods.

Tier 5 (Surveys): Survey, summary, or taxonomy of existing work. Examples: literature reviews, state-of-the-art roundups, tutorial papers.

Few-shot examples:
- "Attention Is All You Need" (Transformer, 2017) → Tier 1: Introduced the Transformer architecture, creating an entirely new paradigm for sequence modeling.
- "Deep Residual Learning for Image Recognition" (ResNet, 2015) → Tier 2: Residual connections became the dominant approach for training very deep networks.
- "LoRA: Low-Rank Adaptation of Large Language Models" (2021) → Tier 3: A specialized parameter-efficient fine-tuning method, an improvement over full fine-tuning for a specific use case.
- "A Survey of Large Language Models" (2023) → Tier 5: Comprehensive survey of existing LLM work.

The knowledge bank contains these foundational concepts (use exact names when referencing them):
{concept_index}

Return a JSON object with exactly these fields:
{{
  "tier": <integer 1-5>,
  "confidence": "<high|medium|low>",
  "reasoning": "<1-2 sentences explaining the tier assignment>",
  "summary": "<2-3 sentence plain-English summary of what this paper does>",
  "key_contributions": ["<contribution 1>", "<contribution 2>"],
  "foundational_concept_names": ["<concept name from knowledge bank>", ...]
}}\
"""

_CONCEPT_EXTRACTION_SYSTEM_PROMPT = """\
You are an expert ML researcher extracting foundational concepts from academic text. For each distinct concept found in the text, provide:
- name: a concise canonical name in lowercase (e.g., "backpropagation", "attention mechanism", "variational autoencoder")
- description: a ~200 word explanation suitable for a knowledge bank entry
- domain_tags: list of relevant domains (e.g., ["deep learning", "optimization", "natural language processing"])
- prerequisite_concept_names: list of other concept names (from this same extraction) that this concept depends on

Focus on foundational, reusable concepts — not paper-specific results. Extract 5-15 concepts per source.

Return a JSON object with exactly this structure:
{{
  "concepts": [
    {{
      "name": "<concept name>",
      "description": "<~200 word description>",
      "domain_tags": ["<tag>", ...],
      "prerequisite_concept_names": ["<concept name>", ...]
    }},
    ...
  ]
}}\
"""


# ---------------------------------------------------------------------------
# Prompt builder helpers
# ---------------------------------------------------------------------------


def _build_classification_system_prompt(concept_index: list[str]) -> str:
    concept_block = "\n".join(concept_index) if concept_index else "(none yet)"
    return _CLASSIFICATION_SYSTEM_PROMPT_TEMPLATE.format(concept_index=concept_block)


def _build_classification_user_prompt(paper: ArxivPaper) -> str:
    return (
        f"Title: {paper.title}\n"
        f"Abstract: {paper.abstract}\n"
        f"Categories: {', '.join(paper.all_categories)}\n"
        f"Published: {paper.published_date}"
    )


# ---------------------------------------------------------------------------
# OllamaClassifier
# ---------------------------------------------------------------------------


class OllamaClassifier:
    """High-level classifier that builds prompts and maps Ollama responses to models.

    Depends on OllamaClient for all network calls and retry logic.
    """

    def __init__(
        self,
        cfg: Settings | None = None,
        client: OllamaClient | None = None,
    ) -> None:
        self._cfg = cfg or default_settings
        self._client = client or OllamaClient(cfg=self._cfg)

    def classify_paper(
        self,
        paper: ArxivPaper,
        concept_index: list[str],
    ) -> PaperClassification:
        """Classify a paper using the 5-tier taxonomy.

        Builds system + user prompts, calls OllamaClient.chat(), maps the
        response to PaperClassification. Returns classification_failed=True
        if OllamaClient exhausts all retries.
        """
        system_prompt = _build_classification_system_prompt(concept_index)
        user_prompt = _build_classification_user_prompt(paper)

        result = self._client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=_ClassificationResponse,
        )

        if result is None:
            logger.warning("Classification failed for paper %s", paper.arxiv_id)
            return PaperClassification(
                tier=None,
                confidence=None,
                reasoning=None,
                summary=None,
                classification_failed=True,
            )

        return PaperClassification(
            tier=result.tier,
            confidence=result.confidence,
            reasoning=result.reasoning,
            summary=result.summary,
            key_contributions=result.key_contributions,
            foundational_concept_names=result.foundational_concept_names,
            classification_failed=False,
        )

    def extract_concepts(
        self,
        text: str,
        source_id: str,
    ) -> list[ExtractedConcept]:
        """Extract foundational concepts from text (abstract or textbook chapter).

        Used during the seeding phase only. Returns [] if OllamaClient fails.
        """
        system_prompt = _CONCEPT_EXTRACTION_SYSTEM_PROMPT
        user_prompt = f"Source: {source_id}\nText: {text}"

        result = self._client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=_ConceptsResponse,
        )

        if result is None:
            logger.warning("Concept extraction failed for source %s", source_id)
            return []

        logger.info(
            "Extracted %d concepts from source %s",
            len(result.concepts),
            source_id,
        )
        return result.concepts
