"""Unit tests for GraphExporter (TASK-M1-005).

Fixture: 3 concepts, 4 relationships in an in-memory SQLite database.
Tests cover Obsidian vault, Neo4j Cypher, and Cytoscape JSON formats.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.models.concepts import Concept, ConceptRelationship
from src.services.graph_exporter import GraphExporter, _slug
from src.store.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> GraphStore:
    """Return an in-memory GraphStore with 3 concepts and 4 relationships seeded."""
    store = GraphStore(":memory:")

    concepts = [
        Concept(
            name="Decision Tree",
            concept_type="Algorithm",
            what_it_is="A tree-structured supervised learning algorithm.",
            what_problem_it_solves="Classification and regression via hierarchical splits.",
            introduced_year=1984,
            domain_tags=["supervised-learning", "interpretable-ml"],
            source_refs=["Breiman et al. 1984"],
            content_angles=["Why Decision Trees Still Matter in 2025"],
        ),
        Concept(
            name="Random Forest",
            concept_type="Algorithm",
            what_it_is="An ensemble of decision trees trained on bootstrap samples.",
            what_problem_it_solves="Reduces variance of a single decision tree.",
            introduced_year=2001,
            domain_tags=["ensemble-learning", "supervised-learning"],
            source_refs=["Breiman 2001"],
            content_angles=["Ensemble vs. Single Model: When to Use Random Forest"],
        ),
        Concept(
            name="XGBoost",
            concept_type="Framework",
            what_it_is="An optimized gradient boosting framework.",
            what_problem_it_solves="Fast, scalable gradient boosting for tabular data.",
            introduced_year=2016,
            domain_tags=["ensemble-learning", "gradient-boosting"],
            source_refs=["Chen & Guestrin 2016"],
            content_angles=["XGBoost vs LightGBM: Which Should You Use?"],
        ),
    ]
    for c in concepts:
        store.upsert_concept(c)

    relationships = [
        ConceptRelationship(
            from_concept="Random Forest",
            to_concept="Decision Tree",
            relationship_type="BUILDS_ON",
            label="Uses decision trees as base learners",
        ),
        ConceptRelationship(
            from_concept="XGBoost",
            to_concept="Random Forest",
            relationship_type="ALTERNATIVE_TO",
            label="Competes with Random Forest on tabular data",
        ),
        ConceptRelationship(
            from_concept="XGBoost",
            to_concept="Decision Tree",
            relationship_type="BUILDS_ON",
            label="Gradient boosting uses decision trees as weak learners",
        ),
        ConceptRelationship(
            from_concept="Random Forest",
            to_concept="Decision Tree",
            relationship_type="PREREQUISITE_OF",
            label="Requires understanding of base decision tree",
        ),
    ]
    for rel in relationships:
        store.upsert_concept_relationship(rel)

    return store


# ---------------------------------------------------------------------------
# Obsidian vault tests
# ---------------------------------------------------------------------------


class TestObsidianExport:
    def test_writes_one_file_per_concept(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        count = exporter.to_obsidian_vault(tmp_path)

        assert count == 3
        concepts_dir = tmp_path / "concepts"
        assert concepts_dir.is_dir()
        files = list(concepts_dir.glob("*.md"))
        assert len(files) == 3

    def test_file_names_are_slugified(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        exporter.to_obsidian_vault(tmp_path)

        concepts_dir = tmp_path / "concepts"
        slugs = {f.stem for f in concepts_dir.glob("*.md")}
        assert "decision-tree" in slugs
        assert "random-forest" in slugs
        assert "xgboost" in slugs

    def test_yaml_frontmatter_present(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        exporter.to_obsidian_vault(tmp_path)

        note = (tmp_path / "concepts" / "xgboost.md").read_text()
        assert note.startswith("---")
        assert "name: XGBoost" in note
        assert "introduced_year: 2016" in note
        assert "gradient-boosting" in note
        assert "Chen & Guestrin 2016" in note

    def test_wikilinks_for_builds_on(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        exporter.to_obsidian_vault(tmp_path)

        rf_note = (tmp_path / "concepts" / "random-forest.md").read_text()
        # Random Forest BUILDS_ON Decision Tree
        assert "[[Decision Tree]]" in rf_note
        # Relationship label should appear inline
        assert "Uses decision trees as base learners" in rf_note

    def test_wikilinks_for_alternative_to(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        exporter.to_obsidian_vault(tmp_path)

        xgb_note = (tmp_path / "concepts" / "xgboost.md").read_text()
        assert "[[Random Forest]]" in xgb_note
        assert "Competes with Random Forest on tabular data" in xgb_note

    def test_content_angles_present(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        exporter.to_obsidian_vault(tmp_path)

        dt_note = (tmp_path / "concepts" / "decision-tree.md").read_text()
        assert "Why Decision Trees Still Matter in 2025" in dt_note

    def test_empty_store_returns_zero(self, tmp_path: Path) -> None:
        store = GraphStore(":memory:")
        exporter = GraphExporter(store)
        count = exporter.to_obsidian_vault(tmp_path)
        assert count == 0

    def test_creates_concepts_dir_if_missing(self, tmp_path: Path) -> None:
        store = _make_store()
        vault_dir = tmp_path / "new_vault"
        assert not vault_dir.exists()
        exporter = GraphExporter(store)
        exporter.to_obsidian_vault(vault_dir)
        assert (vault_dir / "concepts").is_dir()


# ---------------------------------------------------------------------------
# Neo4j Cypher tests
# ---------------------------------------------------------------------------


class TestNeo4jCypherExport:
    def test_writes_cypher_file(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        out_path = exporter.to_neo4j_cypher(tmp_path)

        assert out_path.exists()
        assert out_path.suffix == ".cypher"
        assert "graph-" in out_path.name

    def test_cypher_contains_create_concept(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        out_path = exporter.to_neo4j_cypher(tmp_path)

        content = out_path.read_text()
        # All 3 concepts must appear as CREATE statements
        assert "xgboost" in content
        assert "random-forest" in content
        assert "decision-tree" in content
        assert "CREATE" in content

    def test_cypher_contains_relationship_match_create(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        out_path = exporter.to_neo4j_cypher(tmp_path)

        content = out_path.read_text()
        assert "BUILDS_ON" in content
        assert "ALTERNATIVE_TO" in content
        assert "MATCH" in content

    def test_cypher_relationship_label_embedded(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        out_path = exporter.to_neo4j_cypher(tmp_path)

        content = out_path.read_text()
        assert "Uses decision trees as base learners" in content

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        store = _make_store()
        out_dir = tmp_path / "cypher_exports"
        assert not out_dir.exists()
        exporter = GraphExporter(store)
        exporter.to_neo4j_cypher(out_dir)
        assert out_dir.is_dir()


# ---------------------------------------------------------------------------
# Cytoscape JSON tests
# ---------------------------------------------------------------------------


class TestCytoscapeExport:
    def test_writes_json_file(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        out_path = exporter.to_cytoscape_json(tmp_path)

        assert out_path.exists()
        assert out_path.suffix == ".json"

    def test_json_has_nodes_and_edges_keys(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        out_path = exporter.to_cytoscape_json(tmp_path)

        payload = json.loads(out_path.read_text())
        assert "nodes" in payload
        assert "edges" in payload

    def test_node_count(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        out_path = exporter.to_cytoscape_json(tmp_path)

        payload = json.loads(out_path.read_text())
        assert len(payload["nodes"]) == 3

    def test_edge_count(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        out_path = exporter.to_cytoscape_json(tmp_path)

        payload = json.loads(out_path.read_text())
        # 4 relationships seeded
        assert len(payload["edges"]) == 4

    def test_node_data_fields(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        out_path = exporter.to_cytoscape_json(tmp_path)

        payload = json.loads(out_path.read_text())
        node_ids = {n["data"]["id"] for n in payload["nodes"]}
        assert "xgboost" in node_ids
        assert "random-forest" in node_ids
        assert "decision-tree" in node_ids

    def test_edge_data_fields(self, tmp_path: Path) -> None:
        store = _make_store()
        exporter = GraphExporter(store)
        out_path = exporter.to_cytoscape_json(tmp_path)

        payload = json.loads(out_path.read_text())
        edge_types = {e["data"]["relationship_type"] for e in payload["edges"]}
        assert "BUILDS_ON" in edge_types
        assert "ALTERNATIVE_TO" in edge_types
        assert "PREREQUISITE_OF" in edge_types

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        store = _make_store()
        out_dir = tmp_path / "exports_new"
        assert not out_dir.exists()
        exporter = GraphExporter(store)
        exporter.to_cytoscape_json(out_dir)
        assert out_dir.is_dir()


# ---------------------------------------------------------------------------
# Slug helper
# ---------------------------------------------------------------------------


class TestSlugHelper:
    def test_lowercase_hyphenated(self) -> None:
        assert _slug("Decision Tree") == "decision-tree"
        assert _slug("XGBoost") == "xgboost"
        assert _slug("Random Forest") == "random-forest"

    def test_special_characters_stripped(self) -> None:
        assert _slug("Self-Attention (QKV)") == "self-attention-qkv"
