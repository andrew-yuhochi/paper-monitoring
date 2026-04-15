"""ConceptLinker: matches LLM-returned concept names against the knowledge bank.

Matching strategy (per TDD Section 2.3.2):
  1. Normalize both names: lowercase + collapse whitespace
  2. Exact match first
  3. Fuzzy match via difflib.SequenceMatcher if no exact match found
  4. Accept if ratio >= config.concept_match_threshold (default 0.85)
  5. Log unmatched names at INFO for knowledge bank gap review
"""

import difflib
import logging
from src.config import Settings, settings as default_settings
from src.models.graph import Node
from src.store.graph_store import GraphStore

logger = logging.getLogger(__name__)


def _normalize(name: str) -> str:
    """Normalize a concept name for comparison: lowercase + collapse whitespace."""
    return " ".join(name.lower().split())


class ConceptLinker:
    """Matches concept names to knowledge bank nodes and creates graph edges."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or default_settings

    def link_paper_to_concepts(
        self,
        paper_node_id: str,
        concept_names: list[str],
        store: GraphStore,
    ) -> list[str]:
        """Create BUILDS_ON edges from a paper node to matched concept nodes.

        Args:
            paper_node_id: Node ID of the paper (e.g. "paper:2604.11521").
            concept_names: Concept names returned by OllamaClassifier.
            store: GraphStore instance.

        Returns:
            List of matched concept node IDs.
            Unmatched names are logged at INFO level.
        """
        concept_nodes = store.get_nodes_by_type("concept")
        matched_ids: list[str] = []
        for name in concept_names:
            node_id = self._match_and_link(
                name, concept_nodes, store,
                source_id=paper_node_id,
                relationship_type="BUILDS_ON",
            )
            if node_id is not None:
                matched_ids.append(node_id)
        return matched_ids

    def link_concept_prerequisites(
        self,
        concept_node_id: str,
        prerequisite_names: list[str],
        store: GraphStore,
    ) -> list[str]:
        """Create PREREQUISITE_OF edges between concept nodes.

        Used during seeding when wiring prerequisite relationships.

        Args:
            concept_node_id: Node ID of the concept that has prerequisites.
            prerequisite_names: Prerequisite concept names to resolve.
            store: GraphStore instance.

        Returns:
            List of matched prerequisite concept node IDs.
        """
        concept_nodes = store.get_nodes_by_type("concept")
        matched_ids: list[str] = []
        for name in prerequisite_names:
            node_id = self._match_and_link(
                name, concept_nodes, store,
                source_id=concept_node_id,
                relationship_type="PREREQUISITE_OF",
            )
            if node_id is not None:
                matched_ids.append(node_id)
        return matched_ids

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _match_and_link(
        self,
        name: str,
        concept_nodes: list[Node],
        store: GraphStore,
        source_id: str,
        relationship_type: str,
    ) -> str | None:
        """Find the best matching concept node, create the edge, return node ID."""
        matched_id = self._find_match(name, concept_nodes)
        if matched_id is not None:
            store.upsert_edge(source_id, matched_id, relationship_type)
        else:
            logger.info(
                "No concept match found for %r (threshold=%.2f) — "
                "consider adding to knowledge bank",
                name,
                self._cfg.concept_match_threshold,
            )
        return matched_id

    def _find_match(self, name: str, concept_nodes: list[Node]) -> str | None:
        """Return the node ID of the best matching concept, or None."""
        if not concept_nodes:
            return None

        norm_name = _normalize(name)

        # Pass 1: exact match
        for node in concept_nodes:
            if _normalize(node.label) == norm_name:
                return node.id

        # Pass 2: fuzzy match
        best_ratio = 0.0
        best_node_id: str | None = None
        for node in concept_nodes:
            ratio = difflib.SequenceMatcher(
                None, norm_name, _normalize(node.label)
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_node_id = node.id

        if best_ratio >= self._cfg.concept_match_threshold:
            return best_node_id

        return None
