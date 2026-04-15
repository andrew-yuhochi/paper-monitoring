"""Unit tests for ConceptLinker."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.services.linker import ConceptLinker
from src.store.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_cfg(threshold: float = 0.85) -> MagicMock:
    cfg = MagicMock()
    cfg.concept_match_threshold = threshold
    return cfg


@pytest.fixture
def store() -> GraphStore:
    """In-memory GraphStore with a paper node and three concept nodes seeded."""
    gs = GraphStore(":memory:")
    # Paper node required as FK source for BUILDS_ON edges
    gs.upsert_node("paper:1234.5678", "paper", "Test Paper", {})
    # Concept nodes
    gs.upsert_node("concept:attention_mechanism", "concept", "attention mechanism", {})
    gs.upsert_node("concept:transformer", "concept", "transformer", {})
    gs.upsert_node("concept:neural_network", "concept", "neural network", {})
    return gs


@pytest.fixture
def linker() -> ConceptLinker:
    return ConceptLinker(cfg=_make_cfg(threshold=0.85))


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------


def test_exact_match_creates_builds_on_edge(store: GraphStore, linker: ConceptLinker) -> None:
    """Exact name match creates a BUILDS_ON edge and returns the matched node ID."""
    matched = linker.link_paper_to_concepts("paper:1234.5678", ["attention mechanism"], store)

    assert matched == ["concept:attention_mechanism"]
    edges = store.get_edges_from("paper:1234.5678", "BUILDS_ON")
    assert len(edges) == 1
    assert edges[0].target_id == "concept:attention_mechanism"


def test_exact_match_is_case_insensitive(store: GraphStore, linker: ConceptLinker) -> None:
    """Matching normalizes to lowercase before comparison."""
    matched = linker.link_paper_to_concepts("paper:1234.5678", ["Attention Mechanism"], store)

    assert matched == ["concept:attention_mechanism"]


def test_exact_match_collapses_whitespace(store: GraphStore, linker: ConceptLinker) -> None:
    """Extra internal whitespace is collapsed before comparison."""
    matched = linker.link_paper_to_concepts("paper:1234.5678", ["attention  mechanism"], store)

    assert matched == ["concept:attention_mechanism"]


# ---------------------------------------------------------------------------
# Fuzzy match boundary
# ---------------------------------------------------------------------------


def test_fuzzy_match_accepted_at_threshold(store: GraphStore) -> None:
    """A fuzzy ratio exactly equal to the threshold is accepted."""
    linker = ConceptLinker(cfg=_make_cfg(threshold=0.85))
    # "attention mechanismX" normalizes to "attention mechanismx" — no exact match
    with patch("src.services.linker.difflib.SequenceMatcher") as mock_sm_class:
        mock_inst = MagicMock()
        mock_inst.ratio.return_value = 0.85
        mock_sm_class.return_value = mock_inst

        matched = linker.link_paper_to_concepts(
            "paper:1234.5678", ["attention mechanismx"], store
        )

    assert len(matched) == 1


def test_fuzzy_match_rejected_below_threshold(
    store: GraphStore, caplog: pytest.LogCaptureFixture
) -> None:
    """A fuzzy ratio strictly below the threshold is rejected and the miss is logged."""
    linker = ConceptLinker(cfg=_make_cfg(threshold=0.85))
    with patch("src.services.linker.difflib.SequenceMatcher") as mock_sm_class:
        mock_inst = MagicMock()
        mock_inst.ratio.return_value = 0.84
        mock_sm_class.return_value = mock_inst

        with caplog.at_level(logging.INFO, logger="src.services.linker"):
            matched = linker.link_paper_to_concepts(
                "paper:1234.5678", ["attention mechanismx"], store
            )

    assert matched == []
    assert "No concept match found" in caplog.text


# ---------------------------------------------------------------------------
# Unmatched name logging
# ---------------------------------------------------------------------------


def test_unmatched_name_logged_at_info(
    store: GraphStore, linker: ConceptLinker, caplog: pytest.LogCaptureFixture
) -> None:
    """Names with no viable match are logged at INFO level."""
    with caplog.at_level(logging.INFO, logger="src.services.linker"):
        matched = linker.link_paper_to_concepts(
            "paper:1234.5678", ["quantum entanglement in distributed systems"], store
        )

    assert matched == []
    assert "No concept match found" in caplog.text


# ---------------------------------------------------------------------------
# link_concept_prerequisites
# ---------------------------------------------------------------------------


def test_link_concept_prerequisites_creates_prerequisite_of_edge(
    store: GraphStore, linker: ConceptLinker
) -> None:
    """Exact match creates a PREREQUISITE_OF edge and returns the matched node ID."""
    matched = linker.link_concept_prerequisites(
        "concept:transformer", ["neural network"], store
    )

    assert matched == ["concept:neural_network"]
    edges = store.get_edges_from("concept:transformer", "PREREQUISITE_OF")
    assert len(edges) == 1
    assert edges[0].target_id == "concept:neural_network"


def test_link_concept_prerequisites_unmatched_returns_empty(
    store: GraphStore, linker: ConceptLinker
) -> None:
    """Unmatched prerequisite name returns empty list with no edges created."""
    matched = linker.link_concept_prerequisites(
        "concept:transformer", ["quantum entanglement"], store
    )

    assert matched == []
    assert store.get_edges_from("concept:transformer", "PREREQUISITE_OF") == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_concept_store_returns_no_matches(linker: ConceptLinker) -> None:
    """With no concept nodes in the store, every name returns unmatched."""
    empty_store = GraphStore(":memory:")
    empty_store.upsert_node("paper:1234.5678", "paper", "Test Paper", {})
    matched = linker.link_paper_to_concepts("paper:1234.5678", ["attention mechanism"], empty_store)

    assert matched == []


def test_empty_names_list_returns_empty(store: GraphStore, linker: ConceptLinker) -> None:
    """Passing an empty names list returns an empty matched list with no edges."""
    matched = linker.link_paper_to_concepts("paper:1234.5678", [], store)

    assert matched == []
    assert store.get_edges_from("paper:1234.5678", "BUILDS_ON") == []


def test_multiple_names_partial_match(store: GraphStore, linker: ConceptLinker) -> None:
    """Multiple names: matched names return IDs, unmatched names are skipped."""
    matched = linker.link_paper_to_concepts(
        "paper:1234.5678",
        ["attention mechanism", "quantum entanglement", "transformer"],
        store,
    )

    assert "concept:attention_mechanism" in matched
    assert "concept:transformer" in matched
    assert len(matched) == 2
    edges = store.get_edges_from("paper:1234.5678", "BUILDS_ON")
    assert len(edges) == 2
