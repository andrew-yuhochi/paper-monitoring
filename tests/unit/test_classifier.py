# Unit tests for OllamaClassifier.
# OllamaClient is injected as a MagicMock — no ollama library calls are made.

from unittest.mock import MagicMock

import pytest

from src.models.arxiv import ArxivPaper
from src.models.classification import ExtractedConcept, PaperClassification
from src.services.classifier import (
    OllamaClassifier,
    _ClassificationResponse,
    _ConceptsResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paper(**kwargs) -> ArxivPaper:
    defaults = dict(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models...",
        authors=["Vaswani, Ashish"],
        primary_category="cs.CL",
        all_categories=["cs.CL", "cs.LG"],
        published_date="2017-06-12",
        arxiv_url="https://arxiv.org/abs/1706.03762",
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )
    defaults.update(kwargs)
    return ArxivPaper(**defaults)


def _make_classifier(chat_return_value):
    """Build an OllamaClassifier with a mocked OllamaClient."""
    mock_client = MagicMock()
    mock_client.chat.return_value = chat_return_value
    return OllamaClassifier(client=mock_client), mock_client


def _valid_classification_response() -> _ClassificationResponse:
    return _ClassificationResponse(
        tier=1,
        confidence="high",
        reasoning="Game changer",
        summary="Introduced transformers",
        key_contributions=["self-attention"],
        foundational_concept_names=["attention mechanism"],
    )


# ---------------------------------------------------------------------------
# classify_paper tests
# ---------------------------------------------------------------------------


def test_classify_paper_success() -> None:
    classifier, _ = _make_classifier(_valid_classification_response())
    paper = _make_paper()

    result = classifier.classify_paper(paper, concept_index=[])

    assert isinstance(result, PaperClassification)
    assert result.tier == 1
    assert result.confidence == "high"
    assert result.reasoning == "Game changer"
    assert result.summary == "Introduced transformers"
    assert result.key_contributions == ["self-attention"]
    assert result.foundational_concept_names == ["attention mechanism"]
    assert result.classification_failed is False


def test_classify_paper_failure_returns_classification_failed() -> None:
    classifier, _ = _make_classifier(None)
    paper = _make_paper()

    result = classifier.classify_paper(paper, concept_index=[])

    assert isinstance(result, PaperClassification)
    assert result.classification_failed is True
    assert result.tier is None
    assert result.confidence is None
    assert result.reasoning is None
    assert result.summary is None


def test_classify_paper_concept_index_injected_in_system_prompt() -> None:
    classifier, mock_client = _make_classifier(_valid_classification_response())
    paper = _make_paper()

    classifier.classify_paper(
        paper,
        concept_index=["attention mechanism", "backpropagation"],
    )

    call_kwargs = mock_client.chat.call_args.kwargs
    system_prompt = call_kwargs["system_prompt"]
    assert "attention mechanism" in system_prompt
    assert "backpropagation" in system_prompt


def test_classify_paper_empty_concept_index_uses_placeholder() -> None:
    classifier, mock_client = _make_classifier(_valid_classification_response())
    paper = _make_paper()

    classifier.classify_paper(paper, concept_index=[])

    call_kwargs = mock_client.chat.call_args.kwargs
    assert "(none yet)" in call_kwargs["system_prompt"]


def test_classify_paper_user_prompt_contains_paper_fields() -> None:
    classifier, mock_client = _make_classifier(_valid_classification_response())
    paper = _make_paper()

    classifier.classify_paper(paper, concept_index=[])

    call_kwargs = mock_client.chat.call_args.kwargs
    user_prompt = call_kwargs["user_prompt"]
    assert paper.title in user_prompt
    assert paper.abstract in user_prompt
    assert paper.published_date in user_prompt


def test_classify_paper_uses_classification_response_model() -> None:
    classifier, mock_client = _make_classifier(_valid_classification_response())
    paper = _make_paper()

    classifier.classify_paper(paper, concept_index=[])

    call_kwargs = mock_client.chat.call_args.kwargs
    assert call_kwargs["response_model"] is _ClassificationResponse


# ---------------------------------------------------------------------------
# extract_concepts tests
# ---------------------------------------------------------------------------


def _make_concepts_response() -> _ConceptsResponse:
    return _ConceptsResponse(
        concepts=[
            ExtractedConcept(
                name="attention mechanism",
                description="A mechanism that allows the model to focus on relevant parts of the input.",
                domain_tags=["deep learning", "natural language processing"],
                prerequisite_concept_names=[],
            ),
            ExtractedConcept(
                name="backpropagation",
                description="An algorithm for computing gradients in neural networks.",
                domain_tags=["deep learning", "optimization"],
                prerequisite_concept_names=[],
            ),
        ]
    )


def test_extract_concepts_success() -> None:
    classifier, _ = _make_classifier(_make_concepts_response())

    result = classifier.extract_concepts(
        text="Some academic text about attention and backpropagation.",
        source_id="test_source",
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(c, ExtractedConcept) for c in result)
    assert result[0].name == "attention mechanism"
    assert result[1].name == "backpropagation"


def test_extract_concepts_failure_returns_empty_list() -> None:
    classifier, _ = _make_classifier(None)

    result = classifier.extract_concepts(
        text="Some academic text.",
        source_id="test_source",
    )

    assert result == []


def test_extract_concepts_user_prompt_contains_source_and_text() -> None:
    classifier, mock_client = _make_classifier(_make_concepts_response())
    text = "Some academic text about neural networks."
    source_id = "Goodfellow et al. Deep Learning, Ch. 3"

    classifier.extract_concepts(text=text, source_id=source_id)

    call_kwargs = mock_client.chat.call_args.kwargs
    user_prompt = call_kwargs["user_prompt"]
    assert source_id in user_prompt
    assert text in user_prompt


def test_extract_concepts_uses_concepts_response_model() -> None:
    classifier, mock_client = _make_classifier(_make_concepts_response())

    classifier.extract_concepts(
        text="Some academic text.",
        source_id="test_source",
    )

    call_kwargs = mock_client.chat.call_args.kwargs
    assert call_kwargs["response_model"] is _ConceptsResponse
