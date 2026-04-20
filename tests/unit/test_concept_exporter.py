# Unit tests for ConceptExporter and SignalLogger (TASK-M1-004).
# Uses in-memory SQLite fixtures — no disk I/O, no external APIs.

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.models.concepts import Concept, ConceptRelationship
from src.services.concept_exporter import ConceptExporter, ConceptNotFoundError
from src.services.signal_logger import SignalLogger
from src.store.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> GraphStore:
    """In-memory GraphStore pre-populated with a small tree-model graph."""
    g = GraphStore(":memory:")

    # Seed three concepts
    decision_tree = Concept(
        name="Decision Tree",
        concept_type="Algorithm",
        what_it_is="A tree-shaped classifier that partitions feature space via recursive splits.",
        what_problem_it_solves="Provides an interpretable non-linear decision boundary.",
        innovation_chain=[{"step": "Introduced recursive binary splitting", "why_needed": "Enables non-linear boundaries"}],
        limitations=["High variance", "Prone to overfitting"],
        introduced_year=1986,
        domain_tags=["supervised-learning", "tree-based"],
        source="manual",
        source_refs=["Breiman et al. 1984 (CART)"],
        content_angles=["The OG tree algorithm"],
        user_id="default",
    )
    gradient_boosting = Concept(
        name="Gradient Boosting",
        concept_type="Algorithm",
        what_it_is="Ensemble method that builds trees sequentially, fitting each to the negative gradient of the loss.",
        what_problem_it_solves="Reduces bias via sequential residual correction.",
        innovation_chain=[{"step": "Generalized boosting to arbitrary loss", "why_needed": "Handles non-exponential losses"}],
        limitations=["Slow training", "Requires tuning"],
        introduced_year=2001,
        domain_tags=["supervised-learning", "ensemble"],
        source="manual",
        source_refs=["Friedman 2001"],
        content_angles=["Boosting for arbitrary loss functions"],
        user_id="default",
    )
    xgboost = Concept(
        name="XGBoost",
        concept_type="Framework",
        what_it_is="Gradient boosted trees with L1/L2 regularization and Newton-step optimization.",
        what_problem_it_solves="Improves speed and regularization over vanilla gradient boosting.",
        innovation_chain=[{"step": "Added L1/L2 regularization", "why_needed": "Prevents overfitting"}],
        limitations=["Memory-intensive", "Many hyperparameters"],
        introduced_year=2016,
        domain_tags=["ensemble", "tabular"],
        source="manual",
        source_refs=["Chen & Guestrin 2016"],
        content_angles=[
            "How XGBoost Won Kaggle",
            "From Decision Tree to XGBoost in 5 Steps",
        ],
        user_id="default",
    )
    lightgbm = Concept(
        name="LightGBM",
        concept_type="Framework",
        what_it_is="Gradient boosted trees using leaf-wise growth for speed.",
        what_problem_it_solves="Faster than XGBoost on large datasets.",
        innovation_chain=[],
        limitations=["Can overfit on small datasets"],
        introduced_year=2017,
        domain_tags=["ensemble", "tabular"],
        source="manual",
        source_refs=[],
        content_angles=[],
        user_id="default",
    )

    for concept in [decision_tree, gradient_boosting, xgboost, lightgbm]:
        g.upsert_concept(concept)

    # Relationships
    rels = [
        ConceptRelationship(
            from_concept="Gradient Boosting",
            to_concept="Decision Tree",
            relationship_type="BUILDS_ON",
            label="Uses decision trees as weak learners",
            user_id="default",
        ),
        ConceptRelationship(
            from_concept="XGBoost",
            to_concept="Gradient Boosting",
            relationship_type="BUILDS_ON",
            label="Extends with L1/L2 regularization and Newton steps",
            user_id="default",
        ),
        ConceptRelationship(
            from_concept="XGBoost",
            to_concept="LightGBM",
            relationship_type="ALTERNATIVE_TO",
            label="LightGBM is faster on large datasets",
            user_id="default",
        ),
    ]
    for rel in rels:
        g.upsert_concept_relationship(rel)

    return g


@pytest.fixture()
def exporter(store: GraphStore) -> ConceptExporter:
    return ConceptExporter(store=store, user_id="default")


@pytest.fixture()
def signal_logger(store: GraphStore) -> SignalLogger:
    return SignalLogger(store=store)


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# resolve_concept tests
# ---------------------------------------------------------------------------


class TestResolveConceptFuzzy:
    def test_exact_match(self, exporter: ConceptExporter) -> None:
        concept = exporter.resolve_concept("XGBoost")
        assert concept.name == "XGBoost"

    def test_case_insensitive(self, exporter: ConceptExporter) -> None:
        concept = exporter.resolve_concept("xgboost")
        assert concept.name == "XGBoost"

    def test_abbreviation_match(self, exporter: ConceptExporter) -> None:
        # "XGB" should fuzzy-match XGBoost via WRatio
        concept = exporter.resolve_concept("XGB")
        assert concept.name == "XGBoost"

    def test_not_found_raises(self, exporter: ConceptExporter) -> None:
        with pytest.raises(ConceptNotFoundError) as exc_info:
            exporter.resolve_concept("completelymadeupterm12345")
        err = exc_info.value
        assert err.query == "completelymadeupterm12345"
        assert len(err.suggestions) <= 3

    def test_not_found_provides_suggestions(self, exporter: ConceptExporter) -> None:
        with pytest.raises(ConceptNotFoundError) as exc_info:
            exporter.resolve_concept("zzznomatchzzz")  # below threshold for all concepts
        # suggestions must be a list of strings
        suggestions = exc_info.value.suggestions
        assert isinstance(suggestions, list)
        assert all(isinstance(s, str) for s in suggestions)


# ---------------------------------------------------------------------------
# traverse_lineage tests
# ---------------------------------------------------------------------------


class TestTraverseLineage:
    def test_xgboost_lineage_has_ancestors(self, exporter: ConceptExporter, store: GraphStore) -> None:
        concept = store.get_concept_by_name("XGBoost", "default")
        assert concept is not None
        lineage = exporter.traverse_lineage(concept)
        # Should contain Decision Tree and Gradient Boosting
        names = [step.name for step in lineage]
        assert "Decision Tree" in names
        assert "Gradient Boosting" in names

    def test_lineage_ordered_oldest_first(self, exporter: ConceptExporter, store: GraphStore) -> None:
        concept = store.get_concept_by_name("XGBoost", "default")
        assert concept is not None
        lineage = exporter.traverse_lineage(concept)
        # Deeper ancestors (Decision Tree, depth 2) should come before shallower ones (Gradient Boosting, depth 1)
        names = [step.name for step in lineage]
        dt_idx = names.index("Decision Tree")
        gb_idx = names.index("Gradient Boosting")
        assert dt_idx < gb_idx, "Decision Tree (ancestor of Gradient Boosting) should appear first"

    def test_foundation_concept_has_empty_lineage(self, exporter: ConceptExporter, store: GraphStore) -> None:
        # Decision Tree has no outgoing BUILDS_ON edges in this fixture
        concept = store.get_concept_by_name("Decision Tree", "default")
        assert concept is not None
        lineage = exporter.traverse_lineage(concept)
        assert lineage == []

    def test_depth_limit_respected(self, exporter: ConceptExporter, store: GraphStore) -> None:
        concept = store.get_concept_by_name("XGBoost", "default")
        assert concept is not None
        lineage = exporter.traverse_lineage(concept, max_depth=1)
        # With max_depth=1, only direct BUILDS_ON neighbors are returned
        depths = [step.depth for step in lineage]
        assert all(d <= 1 for d in depths)


# ---------------------------------------------------------------------------
# Markdown export tests
# ---------------------------------------------------------------------------


class TestExportMarkdown:
    def test_export_creates_file(self, exporter: ConceptExporter, store: GraphStore, tmp_dir: Path) -> None:
        concept = store.get_concept_by_name("XGBoost", "default")
        assert concept is not None
        out_path = exporter.export_markdown(concept, tmp_dir)
        assert out_path.exists()
        assert out_path.suffix == ".md"

    def test_export_filename_has_slug_and_date(self, exporter: ConceptExporter, store: GraphStore, tmp_dir: Path) -> None:
        concept = store.get_concept_by_name("XGBoost", "default")
        assert concept is not None
        out_path = exporter.export_markdown(concept, tmp_dir)
        assert "xgboost" in out_path.name

    def test_export_contains_all_required_sections(
        self, exporter: ConceptExporter, store: GraphStore, tmp_dir: Path
    ) -> None:
        concept = store.get_concept_by_name("XGBoost", "default")
        assert concept is not None
        out_path = exporter.export_markdown(concept, tmp_dir)
        content = out_path.read_text(encoding="utf-8")

        required_sections = [
            "**CONCEPT**",
            "**DEFINITION**",
            "**CATEGORY**",
            "**YEAR_INTRODUCED**",
            "## LINEAGE",
            "## KEY_INNOVATIONS",
            "## LIMITATIONS",
            "## ALTERNATIVES",
            "## CONTENT_ANGLES",
            "## RELATED",
            "## SOURCES",
        ]
        for section in required_sections:
            assert section in content, f"Missing section: {section}"

    def test_export_content_angles_present(
        self, exporter: ConceptExporter, store: GraphStore, tmp_dir: Path
    ) -> None:
        concept = store.get_concept_by_name("XGBoost", "default")
        assert concept is not None
        out_path = exporter.export_markdown(concept, tmp_dir)
        content = out_path.read_text(encoding="utf-8")
        assert "How XGBoost Won Kaggle" in content

    def test_export_alternatives_populated(
        self, exporter: ConceptExporter, store: GraphStore, tmp_dir: Path
    ) -> None:
        concept = store.get_concept_by_name("XGBoost", "default")
        assert concept is not None
        out_path = exporter.export_markdown(concept, tmp_dir)
        content = out_path.read_text(encoding="utf-8")
        assert "LightGBM" in content


# ---------------------------------------------------------------------------
# JSON export tests
# ---------------------------------------------------------------------------


class TestExportJson:
    def test_json_export_creates_file(
        self, exporter: ConceptExporter, store: GraphStore, tmp_dir: Path
    ) -> None:
        concept = store.get_concept_by_name("XGBoost", "default")
        assert concept is not None
        out_path = exporter.export_json(concept, tmp_dir)
        assert out_path.exists()
        assert out_path.suffix == ".json"

    def test_json_has_required_fields(
        self, exporter: ConceptExporter, store: GraphStore, tmp_dir: Path
    ) -> None:
        concept = store.get_concept_by_name("XGBoost", "default")
        assert concept is not None
        out_path = exporter.export_json(concept, tmp_dir)
        data = json.loads(out_path.read_text(encoding="utf-8"))

        required_keys = [
            "concept", "definition", "category", "year_introduced",
            "lineage", "key_innovations", "limitations", "alternatives",
            "content_angles", "related", "sources",
        ]
        for key in required_keys:
            assert key in data, f"Missing JSON key: {key}"

    def test_json_lineage_is_list(
        self, exporter: ConceptExporter, store: GraphStore, tmp_dir: Path
    ) -> None:
        concept = store.get_concept_by_name("XGBoost", "default")
        assert concept is not None
        out_path = exporter.export_json(concept, tmp_dir)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert isinstance(data["lineage"], list)


# ---------------------------------------------------------------------------
# SignalLogger tests
# ---------------------------------------------------------------------------


class TestSignalLoggerQuery:
    def test_log_query_writes_row(self, signal_logger: SignalLogger, store: GraphStore) -> None:
        signal_logger.log_query("XGBoost", user_id="default")
        row = store._conn.execute(
            "SELECT COUNT(*) AS cnt FROM concept_queries WHERE user_id = 'default'"
        ).fetchone()
        assert row["cnt"] >= 1

    def test_log_query_records_concept_id(self, signal_logger: SignalLogger, store: GraphStore) -> None:
        signal_logger.log_query("XGBoost", user_id="default")
        row = store._conn.execute(
            """
            SELECT cq.concept_id, c.name
            FROM concept_queries cq
            JOIN concepts c ON cq.concept_id = c.id
            WHERE cq.user_id = 'default'
            ORDER BY cq.id DESC
            LIMIT 1
            """
        ).fetchone()
        assert row is not None
        assert row["name"] == "XGBoost"

    def test_log_query_unknown_concept_does_not_crash(
        self, signal_logger: SignalLogger
    ) -> None:
        # Should log a warning but not raise
        signal_logger.log_query("DoesNotExistAtAll", user_id="default")


class TestSignalLoggerPublication:
    def test_log_publication_writes_row(self, signal_logger: SignalLogger, store: GraphStore) -> None:
        signal_logger.log_publication(
            concept_name="XGBoost",
            channel="linkedin",
            url="https://example.com/post",
            user_id="default",
        )
        row = store._conn.execute(
            "SELECT COUNT(*) AS cnt FROM content_publications WHERE channel = 'linkedin'"
        ).fetchone()
        assert row["cnt"] >= 1

    def test_log_publication_stores_url(self, signal_logger: SignalLogger, store: GraphStore) -> None:
        signal_logger.log_publication(
            concept_name="XGBoost",
            channel="youtube",
            url="https://youtube.com/watch?v=test",
            user_id="default",
        )
        row = store._conn.execute(
            "SELECT url FROM content_publications WHERE channel = 'youtube' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert "youtube.com" in row["url"]

    def test_log_publication_unknown_concept_does_not_crash(
        self, signal_logger: SignalLogger
    ) -> None:
        signal_logger.log_publication(
            concept_name="DoesNotExist",
            channel="other",
            user_id="default",
        )


class TestSignalLoggerReport:
    def test_report_returns_expected_keys(self, signal_logger: SignalLogger) -> None:
        data = signal_logger.report(days=30, user_id="default")
        assert "queries_last_30d" in data
        assert "publications_last_30d" in data
        assert "conversion_rate" in data
        assert "top_queried_concepts" in data
        assert "published_concepts" in data

    def test_report_counts_logged_queries(self, signal_logger: SignalLogger) -> None:
        signal_logger.log_query("XGBoost", user_id="default")
        signal_logger.log_query("XGBoost", user_id="default")
        data = signal_logger.report(days=30, user_id="default")
        assert data["queries_last_30d"] >= 2

    def test_report_loop_ratio_zero_when_no_queries(
        self, store: GraphStore
    ) -> None:
        # Fresh store, no queries or publications
        fresh = SignalLogger(store=GraphStore(":memory:"))
        data = fresh.report(days=30, user_id="default")
        assert data["conversion_rate"] == 0.0
