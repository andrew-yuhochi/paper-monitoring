# Unit tests for the concept-first GraphStore methods added in TASK-M1-001.
# All tests use an in-memory SQLite database (:memory:) — no file I/O required.

import sqlite3

import pytest

from src.models.concepts import (
    CitationSnapshot,
    Concept,
    ConceptQuery,
    ConceptRelationship,
    ContentPublication,
    PaperConceptLink,
    PaperRecord,
)
from src.store.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> GraphStore:
    """Fresh in-memory GraphStore for each test."""
    return GraphStore(":memory:")


@pytest.fixture
def xgboost_concept() -> Concept:
    return Concept(
        name="XGBoost",
        concept_type="Technique",
        what_it_is="Gradient boosted trees with L1/L2 regularization.",
        what_problem_it_solves="Overfitting in standard gradient boosting.",
        innovation_chain=[
            {"step": "Gradient Boosting", "why": "base framework"},
            {"step": "XGBoost", "why": "adds regularization"},
        ],
        limitations=["Memory intensive for large datasets"],
        introduced_year=2016,
        domain_tags=["ensemble", "tree-based"],
        source_refs=["Chen & Guestrin 2016"],
        content_angles=["From gradient boosting to XGBoost: the regularization insight"],
    )


@pytest.fixture
def gradient_boosting_concept() -> Concept:
    return Concept(
        name="Gradient Boosting",
        concept_type="Technique",
        what_it_is="Ensemble method that builds trees sequentially.",
        what_problem_it_solves="Weak learner combination for better prediction.",
        domain_tags=["ensemble", "tree-based"],
    )


# ---------------------------------------------------------------------------
# upsert + get_concept_by_name round-trip
# ---------------------------------------------------------------------------


def test_upsert_and_get_concept_round_trip(store: GraphStore, xgboost_concept: Concept) -> None:
    store.upsert_concept(xgboost_concept)
    result = store.get_concept_by_name("XGBoost", "default")

    assert result is not None
    assert result.name == "XGBoost"
    assert result.concept_type == "Technique"
    assert result.introduced_year == 2016
    assert "ensemble" in result.domain_tags
    assert "tree-based" in result.domain_tags
    assert len(result.innovation_chain) == 2
    assert result.limitations == ["Memory intensive for large datasets"]
    assert result.source_refs == ["Chen & Guestrin 2016"]
    assert len(result.content_angles) == 1
    assert result.user_id == "default"


# ---------------------------------------------------------------------------
# Idempotent upsert — same concept twice → one row, updated fields
# ---------------------------------------------------------------------------


def test_idempotent_upsert_updates_fields(store: GraphStore, xgboost_concept: Concept) -> None:
    store.upsert_concept(xgboost_concept)

    updated = xgboost_concept.model_copy(
        update={"what_it_is": "Updated definition.", "introduced_year": 2015}
    )
    store.upsert_concept(updated)

    # Should still be only one row
    count = store._conn.execute(
        "SELECT COUNT(*) FROM concepts WHERE slug = 'xgboost'"
    ).fetchone()[0]
    assert count == 1

    result = store.get_concept_by_name("XGBoost", "default")
    assert result is not None
    assert result.what_it_is == "Updated definition."
    assert result.introduced_year == 2015


# ---------------------------------------------------------------------------
# get_concept_by_name on missing concept returns None
# ---------------------------------------------------------------------------


def test_get_concept_by_name_missing_returns_none(store: GraphStore) -> None:
    result = store.get_concept_by_name("NonExistentConcept", "default")
    assert result is None


# ---------------------------------------------------------------------------
# user_id defaulting — omitting user_id uses 'default'
# ---------------------------------------------------------------------------


def test_user_id_defaults_to_default(store: GraphStore) -> None:
    c = Concept(name="BERT", concept_type="Technique")
    assert c.user_id == "default"

    store.upsert_concept(c)
    result = store.get_concept_by_name("BERT", "default")
    assert result is not None
    assert result.user_id == "default"


def test_user_id_isolation(store: GraphStore) -> None:
    """Concepts for different user_ids are isolated."""
    c_alice = Concept(name="BERT", concept_type="Technique", user_id="alice")
    c_bob = Concept(name="BERT", concept_type="Problem", user_id="bob")
    store.upsert_concept(c_alice)
    store.upsert_concept(c_bob)

    alice_result = store.get_concept_by_name("BERT", "alice")
    bob_result = store.get_concept_by_name("BERT", "bob")
    assert alice_result is not None and alice_result.concept_type == "Technique"
    assert bob_result is not None and bob_result.concept_type == "Problem"


# ---------------------------------------------------------------------------
# list_concepts with domain_tag filter
# ---------------------------------------------------------------------------


def test_list_concepts_no_filter(
    store: GraphStore,
    xgboost_concept: Concept,
    gradient_boosting_concept: Concept,
) -> None:
    store.upsert_concept(xgboost_concept)
    store.upsert_concept(gradient_boosting_concept)
    results = store.list_concepts("default")
    names = {c.name for c in results}
    assert "XGBoost" in names
    assert "Gradient Boosting" in names


def test_list_concepts_domain_tag_filter(
    store: GraphStore,
    xgboost_concept: Concept,
    gradient_boosting_concept: Concept,
) -> None:
    unrelated = Concept(name="BERT", concept_type="Technique", domain_tags=["nlp"])
    store.upsert_concept(xgboost_concept)
    store.upsert_concept(gradient_boosting_concept)
    store.upsert_concept(unrelated)

    tree_based = store.list_concepts("default", domain_tag="tree-based")
    names = {c.name for c in tree_based}
    assert "XGBoost" in names
    assert "Gradient Boosting" in names
    assert "BERT" not in names


# ---------------------------------------------------------------------------
# upsert_concept_relationship + get_relationships
# ---------------------------------------------------------------------------


def test_upsert_and_get_relationships(
    store: GraphStore,
    xgboost_concept: Concept,
    gradient_boosting_concept: Concept,
) -> None:
    store.upsert_concept(xgboost_concept)
    store.upsert_concept(gradient_boosting_concept)

    rel = ConceptRelationship(
        from_concept="XGBoost",
        to_concept="Gradient Boosting",
        relationship_type="BUILDS_ON",
        label="XGBoost adds L1/L2 regularization to standard gradient boosting.",
    )
    store.upsert_concept_relationship(rel)

    results = store.get_relationships("XGBoost", "default")
    assert len(results) == 1
    assert results[0].from_concept == "XGBoost"
    assert results[0].to_concept == "Gradient Boosting"
    assert results[0].relationship_type == "BUILDS_ON"
    assert "regularization" in results[0].label


def test_get_relationships_empty(store: GraphStore, xgboost_concept: Concept) -> None:
    store.upsert_concept(xgboost_concept)
    results = store.get_relationships("XGBoost", "default")
    assert results == []


# ---------------------------------------------------------------------------
# CHECK constraint violation raises IntegrityError (invalid relationship_type)
# ---------------------------------------------------------------------------


def test_invalid_relationship_type_raises_integrity_error(
    store: GraphStore,
    xgboost_concept: Concept,
    gradient_boosting_concept: Concept,
) -> None:
    store.upsert_concept(xgboost_concept)
    store.upsert_concept(gradient_boosting_concept)

    src_id = store._get_concept_id("XGBoost", "default")
    tgt_id = store._get_concept_id("Gradient Boosting", "default")

    with pytest.raises(sqlite3.IntegrityError):
        with store._conn:
            store._conn.execute(
                """
                INSERT INTO concept_relationships
                    (user_id, source_concept_id, target_concept_id,
                     relationship_type, label)
                VALUES ('default', ?, ?, 'INVALID_TYPE', 'bad')
                """,
                (src_id, tgt_id),
            )


# ---------------------------------------------------------------------------
# get_citation_delta math
# ---------------------------------------------------------------------------


def test_get_citation_delta(store: GraphStore) -> None:
    """Snapshot at T and T+4 weeks; delta returned correctly."""
    paper = PaperRecord(arxiv_id="1603.02754", title="XGBoost paper")
    store.upsert_paper(paper)

    # Older snapshot (4 weeks ago)
    snap_old = CitationSnapshot(
        arxiv_id="1603.02754",
        snapshot_date="2026-03-22",
        citation_count=100,
    )
    # More recent snapshot
    snap_new = CitationSnapshot(
        arxiv_id="1603.02754",
        snapshot_date="2026-04-19",
        citation_count=145,
    )
    store.write_citation_snapshot(snap_old)
    store.write_citation_snapshot(snap_new)

    delta = store.get_citation_delta("1603.02754", 4, "default")
    assert delta == 45


def test_get_citation_delta_no_snapshots(store: GraphStore) -> None:
    paper = PaperRecord(arxiv_id="1603.02754", title="XGBoost paper")
    store.upsert_paper(paper)
    delta = store.get_citation_delta("1603.02754", 4, "default")
    assert delta == 0


# ---------------------------------------------------------------------------
# loop_report conversion_rate calculation
# ---------------------------------------------------------------------------


def test_loop_report_empty(store: GraphStore) -> None:
    report = store.loop_report("default")
    assert report["queries_last_30d"] == 0
    assert report["publications_last_30d"] == 0
    assert report["conversion_rate"] == 0.0


def test_loop_report_with_data(store: GraphStore, xgboost_concept: Concept) -> None:
    store.upsert_concept(xgboost_concept)

    # Log 3 queries
    for _ in range(3):
        store.log_concept_query(
            ConceptQuery(concept_name="XGBoost", export_format="markdown")
        )

    # Log 1 publication
    store.log_content_publication(
        ContentPublication(concept_name="XGBoost", channel="linkedin")
    )

    report = store.loop_report("default")
    assert report["queries_last_30d"] == 3
    assert report["publications_last_30d"] == 1
    assert abs(report["conversion_rate"] - round(1 / 3, 4)) < 1e-6


def test_loop_report_no_division_by_zero(store: GraphStore, xgboost_concept: Concept) -> None:
    """No queries but a publication should return conversion_rate = 0.0."""
    store.upsert_concept(xgboost_concept)
    # Publication without prior query (edge case)
    store.log_content_publication(
        ContentPublication(concept_name="XGBoost", channel="youtube")
    )
    report = store.loop_report("default")
    # queries = 0 → rate = 0.0 (no division by zero)
    assert report["conversion_rate"] == 0.0


# ---------------------------------------------------------------------------
# write_citation_snapshot idempotency
# ---------------------------------------------------------------------------


def test_write_citation_snapshot_idempotent(store: GraphStore) -> None:
    """Writing the same (arxiv_id, snapshot_date) twice keeps one row with updated counts."""
    paper = PaperRecord(arxiv_id="2301.00001", title="Idempotency test paper")
    store.upsert_paper(paper)

    snap = CitationSnapshot(arxiv_id="2301.00001", snapshot_date="2026-04-01", citation_count=10)
    store.write_citation_snapshot(snap)

    # Write again with a higher count — INSERT OR REPLACE should update the row
    snap_updated = CitationSnapshot(
        arxiv_id="2301.00001", snapshot_date="2026-04-01", citation_count=25
    )
    store.write_citation_snapshot(snap_updated)

    rows = store._conn.execute(
        """
        SELECT citation_count FROM citation_snapshots
        WHERE check_date = '2026-04-01'
        """
    ).fetchall()
    assert len(rows) == 1, "Expected exactly one row after two writes of the same snapshot"
    assert rows[0]["citation_count"] == 25, "Expected updated citation_count from second write"


# ---------------------------------------------------------------------------
# get_resurrection_cohort
# ---------------------------------------------------------------------------


def test_resurrection_cohort_includes_qualifying_paper(store: GraphStore) -> None:
    """Paper with delta > min_delta and total < max_total appears in cohort."""
    paper = PaperRecord(arxiv_id="2101.00001", title="Resurrection paper")
    store.upsert_paper(paper)

    # Two snapshots 4 weeks apart: delta = 30, total = 45 (below max_total=50)
    store.write_citation_snapshot(
        CitationSnapshot(arxiv_id="2101.00001", snapshot_date="2026-03-22", citation_count=15)
    )
    store.write_citation_snapshot(
        CitationSnapshot(arxiv_id="2101.00001", snapshot_date="2026-04-19", citation_count=45)
    )

    cohort = store.get_resurrection_cohort(min_delta=5, max_total=50, user_id="default")
    arxiv_ids = [c.arxiv_id for c in cohort]
    assert "2101.00001" in arxiv_ids, "Qualifying paper should appear in resurrection cohort"
    match = next(c for c in cohort if c.arxiv_id == "2101.00001")
    assert match.delta_citations == 30
    assert match.weeks_observed == 4


def test_resurrection_cohort_excludes_high_total(store: GraphStore) -> None:
    """Paper with delta > min_delta but total > max_total is excluded from cohort."""
    paper = PaperRecord(arxiv_id="2101.00002", title="High-citation paper")
    store.upsert_paper(paper)

    # delta = 30, but total = 80 > max_total=50 → should be excluded
    store.write_citation_snapshot(
        CitationSnapshot(arxiv_id="2101.00002", snapshot_date="2026-03-22", citation_count=50)
    )
    store.write_citation_snapshot(
        CitationSnapshot(arxiv_id="2101.00002", snapshot_date="2026-04-19", citation_count=80)
    )

    cohort = store.get_resurrection_cohort(min_delta=5, max_total=50, user_id="default")
    arxiv_ids = [c.arxiv_id for c in cohort]
    assert "2101.00002" not in arxiv_ids, "High-total paper should be excluded from cohort"
