"""Unit tests for GraphStore against an in-memory SQLite database."""

import pytest

from src.models.graph import Edge, Node, WeeklyRun
from src.store.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> GraphStore:
    """Fresh in-memory GraphStore for each test."""
    return GraphStore(":memory:")


@pytest.fixture
def concept_node_data() -> dict:
    return {
        "node_id": "concept:transformer",
        "node_type": "concept",
        "label": "transformer",
        "properties": {
            "description": "Self-attention based sequence model.",
            "domain_tags": ["deep learning", "nlp"],
        },
    }


@pytest.fixture
def paper_node_data() -> dict:
    return {
        "node_id": "paper:1706.03762",
        "node_type": "paper",
        "label": "Attention Is All You Need",
        "properties": {
            "arxiv_id": "1706.03762",
            "tier": 1,
            "run_date": "2026-04-14",
        },
    }


# ---------------------------------------------------------------------------
# Schema existence
# ---------------------------------------------------------------------------


class TestSchema:
    def _table_exists(self, store: GraphStore, table: str) -> bool:
        row = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None

    def _index_exists(self, store: GraphStore, index: str) -> bool:
        row = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
            (index,),
        ).fetchone()
        return row is not None

    def test_nodes_table_exists(self, store: GraphStore) -> None:
        assert self._table_exists(store, "nodes")

    def test_edges_table_exists(self, store: GraphStore) -> None:
        assert self._table_exists(store, "edges")

    def test_weekly_runs_table_exists(self, store: GraphStore) -> None:
        assert self._table_exists(store, "weekly_runs")

    def test_all_six_indexes_exist(self, store: GraphStore) -> None:
        expected = [
            "idx_nodes_type",
            "idx_nodes_label",
            "idx_edges_source",
            "idx_edges_target",
            "idx_edges_type",
            "idx_runs_date",
        ]
        for name in expected:
            assert self._index_exists(store, name), f"Missing index: {name}"

    def test_nodes_columns(self, store: GraphStore) -> None:
        rows = store._conn.execute("PRAGMA table_info(nodes)").fetchall()
        columns = {r[1] for r in rows}
        assert {"id", "node_type", "label", "properties", "created_at", "updated_at"} == columns

    def test_edges_composite_primary_key(self, store: GraphStore) -> None:
        # SQLite stores PK info in PRAGMA index_list; composite PKs show up as sqlite_autoindex_*
        rows = store._conn.execute("PRAGMA index_list(edges)").fetchall()
        pk_index = next((r for r in rows if r[2] == 1), None)  # origin=pk
        assert pk_index is not None
        pk_cols = store._conn.execute(
            f"PRAGMA index_info('{pk_index[1]}')"
        ).fetchall()
        pk_col_names = {r[2] for r in pk_cols}
        assert pk_col_names == {"source_id", "target_id", "relationship_type"}

    def test_weekly_runs_columns(self, store: GraphStore) -> None:
        rows = store._conn.execute("PRAGMA table_info(weekly_runs)").fetchall()
        columns = {r[1] for r in rows}
        expected = {
            "id", "run_date", "started_at", "completed_at",
            "papers_fetched", "papers_classified", "papers_failed",
            "digest_path", "status", "error_message",
        }
        assert expected == columns


# ---------------------------------------------------------------------------
# upsert_node
# ---------------------------------------------------------------------------


class TestUpsertNode:
    def test_insert_new_node(self, store: GraphStore, concept_node_data: dict) -> None:
        store.upsert_node(**concept_node_data)
        node = store.get_node(concept_node_data["node_id"])
        assert node is not None
        assert node.id == "concept:transformer"
        assert node.node_type == "concept"
        assert node.label == "transformer"
        assert node.properties["domain_tags"] == ["deep learning", "nlp"]

    def test_update_properties_on_conflict(self, store: GraphStore, concept_node_data: dict) -> None:
        store.upsert_node(**concept_node_data)
        updated_props = {"description": "Updated description.", "domain_tags": ["nlp"]}
        store.upsert_node(
            node_id=concept_node_data["node_id"],
            node_type=concept_node_data["node_type"],
            label=concept_node_data["label"],
            properties=updated_props,
        )
        node = store.get_node(concept_node_data["node_id"])
        assert node is not None
        assert node.properties["description"] == "Updated description."
        assert node.properties["domain_tags"] == ["nlp"]

    def test_upsert_preserves_created_at(self, store: GraphStore, concept_node_data: dict) -> None:
        store.upsert_node(**concept_node_data)
        created_at_before = store._conn.execute(
            "SELECT created_at FROM nodes WHERE id=?", (concept_node_data["node_id"],)
        ).fetchone()[0]

        store.upsert_node(**{**concept_node_data, "properties": {"new": "value"}})
        created_at_after = store._conn.execute(
            "SELECT created_at FROM nodes WHERE id=?", (concept_node_data["node_id"],)
        ).fetchone()[0]

        assert created_at_before == created_at_after

    def test_upsert_multiple_nodes(self, store: GraphStore) -> None:
        for i in range(5):
            store.upsert_node(f"concept:c{i}", "concept", f"Concept {i}", {"idx": i})
        nodes = store.get_nodes_by_type("concept")
        assert len(nodes) == 5


# ---------------------------------------------------------------------------
# get_node / get_nodes_by_type
# ---------------------------------------------------------------------------


class TestGetNode:
    def test_returns_none_for_missing_id(self, store: GraphStore) -> None:
        assert store.get_node("concept:missing") is None

    def test_get_nodes_by_type_filters_correctly(
        self, store: GraphStore, concept_node_data: dict, paper_node_data: dict
    ) -> None:
        store.upsert_node(**concept_node_data)
        store.upsert_node(**paper_node_data)
        concepts = store.get_nodes_by_type("concept")
        papers = store.get_nodes_by_type("paper")
        assert len(concepts) == 1
        assert len(papers) == 1
        assert concepts[0].id == "concept:transformer"
        assert papers[0].id == "paper:1706.03762"

    def test_get_nodes_by_type_empty(self, store: GraphStore) -> None:
        assert store.get_nodes_by_type("concept") == []


# ---------------------------------------------------------------------------
# get_concept_index
# ---------------------------------------------------------------------------


class TestGetConceptIndex:
    def test_returns_all_concept_labels_sorted(self, store: GraphStore) -> None:
        store.upsert_node("concept:backprop", "concept", "backpropagation", {})
        store.upsert_node("concept:attention", "concept", "attention mechanism", {})
        store.upsert_node("concept:sgd", "concept", "stochastic gradient descent", {})
        index = store.get_concept_index()
        assert index == sorted(["backpropagation", "attention mechanism", "stochastic gradient descent"])

    def test_excludes_paper_nodes(self, store: GraphStore, paper_node_data: dict) -> None:
        store.upsert_node(**paper_node_data)
        store.upsert_node("concept:transformer", "concept", "transformer", {})
        index = store.get_concept_index()
        assert index == ["transformer"]
        assert "Attention Is All You Need" not in index

    def test_empty_when_no_concepts(self, store: GraphStore) -> None:
        assert store.get_concept_index() == []


# ---------------------------------------------------------------------------
# upsert_edge
# ---------------------------------------------------------------------------


class TestUpsertEdge:
    def test_insert_new_edge(
        self, store: GraphStore, concept_node_data: dict, paper_node_data: dict
    ) -> None:
        store.upsert_node(**concept_node_data)
        store.upsert_node(**paper_node_data)
        store.upsert_edge("paper:1706.03762", "concept:transformer", "BUILDS_ON")
        edges = store.get_edges_from("paper:1706.03762")
        assert len(edges) == 1
        assert edges[0].target_id == "concept:transformer"
        assert edges[0].relationship_type == "BUILDS_ON"

    def test_replace_edge_on_conflict(
        self, store: GraphStore, concept_node_data: dict, paper_node_data: dict
    ) -> None:
        store.upsert_node(**concept_node_data)
        store.upsert_node(**paper_node_data)
        store.upsert_edge("paper:1706.03762", "concept:transformer", "BUILDS_ON", weight=1.0)
        store.upsert_edge("paper:1706.03762", "concept:transformer", "BUILDS_ON", weight=0.9)
        edges = store.get_edges_from("paper:1706.03762", "BUILDS_ON")
        assert len(edges) == 1
        assert edges[0].weight == pytest.approx(0.9)

    def test_edge_default_weight(
        self, store: GraphStore, concept_node_data: dict, paper_node_data: dict
    ) -> None:
        store.upsert_node(**concept_node_data)
        store.upsert_node(**paper_node_data)
        store.upsert_edge("paper:1706.03762", "concept:transformer", "INTRODUCES")
        edges = store.get_edges_from("paper:1706.03762")
        assert edges[0].weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_edges_from / get_edges_to
# ---------------------------------------------------------------------------


class TestGetEdges:
    @pytest.fixture(autouse=True)
    def _seed(self, store: GraphStore) -> None:
        store.upsert_node("paper:1706.03762", "paper", "Attention Is All You Need", {})
        store.upsert_node("concept:transformer", "concept", "transformer", {})
        store.upsert_node("concept:attention", "concept", "attention mechanism", {})
        store.upsert_edge("paper:1706.03762", "concept:transformer", "INTRODUCES")
        store.upsert_edge("paper:1706.03762", "concept:attention", "BUILDS_ON")
        store.upsert_edge("concept:attention", "concept:transformer", "PREREQUISITE_OF")

    def test_get_edges_from_no_filter(self, store: GraphStore) -> None:
        edges = store.get_edges_from("paper:1706.03762")
        assert len(edges) == 2

    def test_get_edges_from_filtered_by_type(self, store: GraphStore) -> None:
        edges = store.get_edges_from("paper:1706.03762", "INTRODUCES")
        assert len(edges) == 1
        assert edges[0].target_id == "concept:transformer"

    def test_get_edges_to_no_filter(self, store: GraphStore) -> None:
        edges = store.get_edges_to("concept:transformer")
        assert len(edges) == 2  # INTRODUCES from paper + PREREQUISITE_OF from attention

    def test_get_edges_to_filtered_by_type(self, store: GraphStore) -> None:
        edges = store.get_edges_to("concept:transformer", "PREREQUISITE_OF")
        assert len(edges) == 1
        assert edges[0].source_id == "concept:attention"

    def test_get_edges_from_empty_node(self, store: GraphStore) -> None:
        assert store.get_edges_from("concept:transformer") == []


# ---------------------------------------------------------------------------
# paper_exists
# ---------------------------------------------------------------------------


class TestPaperExists:
    def test_returns_true_for_existing_paper(
        self, store: GraphStore, paper_node_data: dict
    ) -> None:
        store.upsert_node(**paper_node_data)
        assert store.paper_exists("1706.03762") is True

    def test_returns_false_for_missing_paper(self, store: GraphStore) -> None:
        assert store.paper_exists("9999.99999") is False

    def test_concept_node_not_counted_as_paper(self, store: GraphStore) -> None:
        # Concept nodes should not be returned by paper_exists
        store.upsert_node("concept:transformer", "concept", "transformer", {})
        assert store.paper_exists("transformer") is False


# ---------------------------------------------------------------------------
# log_run / get_latest_run
# ---------------------------------------------------------------------------


class TestWeeklyRuns:
    def test_log_run_returns_id(self, store: GraphStore) -> None:
        run_id = store.log_run(
            run_date="2026-04-11",
            papers_fetched=120,
            papers_classified=80,
            digest_path="digests/2026-04-11.html",
        )
        assert isinstance(run_id, int)
        assert run_id > 0

    def test_get_latest_run_returns_most_recent(self, store: GraphStore) -> None:
        store.log_run("2026-04-04", 100, 70, "digests/2026-04-04.html")
        store.log_run("2026-04-11", 120, 80, "digests/2026-04-11.html")
        run = store.get_latest_run()
        assert run is not None
        assert run.run_date == "2026-04-11"
        assert run.papers_fetched == 120
        assert run.papers_classified == 80
        assert run.status == "completed"

    def test_get_latest_run_returns_none_when_empty(self, store: GraphStore) -> None:
        assert store.get_latest_run() is None

    def test_log_run_records_papers_failed(self, store: GraphStore) -> None:
        store.log_run(
            run_date="2026-04-11",
            papers_fetched=120,
            papers_classified=75,
            digest_path="digests/2026-04-11.html",
            papers_failed=5,
        )
        run = store.get_latest_run()
        assert run is not None
        assert run.papers_failed == 5

    def test_log_run_with_error_status(self, store: GraphStore) -> None:
        store.log_run(
            run_date="2026-04-11",
            papers_fetched=0,
            papers_classified=0,
            digest_path="",
            status="failed",
            error_message="ArxivFetcher failed after 3 retries",
        )
        run = store.get_latest_run()
        assert run is not None
        assert run.status == "failed"
        assert run.error_message == "ArxivFetcher failed after 3 retries"

    # create_run / update_run (two-step pipeline pattern)

    def test_create_run_returns_id_with_running_status(self, store: GraphStore) -> None:
        run_id = store.create_run("2026-04-15")
        assert isinstance(run_id, int) and run_id > 0
        run = store.get_latest_run()
        assert run is not None
        assert run.status == "running"
        assert run.run_date == "2026-04-15"

    def test_update_run_sets_completed_status(self, store: GraphStore) -> None:
        run_id = store.create_run("2026-04-15")
        store.update_run(
            run_id,
            status="completed",
            papers_fetched=120,
            papers_classified=80,
            digest_path="digests/2026-04-15.html",
        )
        run = store.get_latest_run()
        assert run.status == "completed"
        assert run.papers_fetched == 120
        assert run.papers_classified == 80
        assert run.digest_path == "digests/2026-04-15.html"
        assert run.completed_at is not None

    def test_update_run_sets_failed_status_with_error(self, store: GraphStore) -> None:
        run_id = store.create_run("2026-04-15")
        store.update_run(run_id, status="failed", error_message="arXiv unreachable")
        run = store.get_latest_run()
        assert run.status == "failed"
        assert run.error_message == "arXiv unreachable"

    def test_create_run_initial_counts_are_zero(self, store: GraphStore) -> None:
        store.create_run("2026-04-15")
        run = store.get_latest_run()
        assert run.papers_fetched == 0
        assert run.papers_classified == 0
        assert run.papers_failed == 0
        assert run.digest_path is None
        assert run.completed_at is None


# ---------------------------------------------------------------------------
# get_papers_for_digest
# ---------------------------------------------------------------------------


class TestGetPapersForDigest:
    @pytest.fixture(autouse=True)
    def _seed(self, store: GraphStore) -> None:
        # Two papers from the target run date
        store.upsert_node(
            "paper:1706.03762",
            "paper",
            "Attention Is All You Need",
            {"arxiv_id": "1706.03762", "tier": 1, "run_date": "2026-04-11"},
        )
        store.upsert_node(
            "paper:2005.14165",
            "paper",
            "Language Models are Few-Shot Learners",
            {"arxiv_id": "2005.14165", "tier": 2, "run_date": "2026-04-11"},
        )
        # One paper from a different run date (should be excluded)
        store.upsert_node(
            "paper:9999.00001",
            "paper",
            "Old Paper",
            {"arxiv_id": "9999.00001", "tier": 3, "run_date": "2026-04-04"},
        )
        # Concept nodes
        store.upsert_node("concept:transformer", "concept", "transformer", {})
        store.upsert_node("concept:attention", "concept", "attention mechanism", {})
        # Edges
        store.upsert_edge("paper:1706.03762", "concept:transformer", "BUILDS_ON")
        store.upsert_edge("paper:1706.03762", "concept:attention", "BUILDS_ON")
        store.upsert_edge("paper:2005.14165", "concept:transformer", "BUILDS_ON")

    def test_returns_only_papers_for_run_date(self, store: GraphStore) -> None:
        papers = store.get_papers_for_digest("2026-04-11")
        paper_ids = {p["id"] for p in papers}
        assert paper_ids == {"paper:1706.03762", "paper:2005.14165"}
        assert "paper:9999.00001" not in paper_ids

    def test_includes_linked_concepts(self, store: GraphStore) -> None:
        papers = store.get_papers_for_digest("2026-04-11")
        paper_map = {p["id"]: p for p in papers}
        attention_concepts = paper_map["paper:1706.03762"]["concepts"]
        assert len(attention_concepts) == 2
        concept_labels = {c["label"] for c in attention_concepts}
        assert concept_labels == {"transformer", "attention mechanism"}

    def test_includes_properties(self, store: GraphStore) -> None:
        papers = store.get_papers_for_digest("2026-04-11")
        paper_map = {p["id"]: p for p in papers}
        assert paper_map["paper:1706.03762"]["properties"]["tier"] == 1

    def test_returns_empty_for_unknown_date(self, store: GraphStore) -> None:
        assert store.get_papers_for_digest("1999-01-01") == []


# ---------------------------------------------------------------------------
# add_node
# ---------------------------------------------------------------------------


class TestAddNode:
    def test_creates_new_node(self, store: GraphStore) -> None:
        store.add_node("concept:relu", "concept", "ReLU", {"description": "Activation function."})
        node = store.get_node("concept:relu")
        assert node is not None
        assert node.label == "ReLU"
        assert node.properties["description"] == "Activation function."

    def test_raises_if_node_exists(self, store: GraphStore, concept_node_data: dict) -> None:
        store.upsert_node(**concept_node_data)
        with pytest.raises(ValueError, match="already exists"):
            store.add_node(
                concept_node_data["node_id"],
                concept_node_data["node_type"],
                concept_node_data["label"],
                concept_node_data["properties"],
            )


# ---------------------------------------------------------------------------
# delete_node
# ---------------------------------------------------------------------------


class TestDeleteNode:
    def test_deletes_node_and_returns_zero_edges(self, store: GraphStore) -> None:
        store.upsert_node("concept:x", "concept", "X", {})
        removed = store.delete_node("concept:x")
        assert removed == 0
        assert store.get_node("concept:x") is None

    def test_deletes_outgoing_edges(self, store: GraphStore) -> None:
        store.upsert_node("paper:111", "paper", "P", {})
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_node("concept:b", "concept", "B", {})
        store.upsert_edge("paper:111", "concept:a", "BUILDS_ON")
        store.upsert_edge("paper:111", "concept:b", "BUILDS_ON")
        removed = store.delete_node("paper:111")
        assert removed == 2
        assert store.get_edges_from("paper:111") == []

    def test_deletes_incoming_edges(self, store: GraphStore) -> None:
        store.upsert_node("paper:111", "paper", "P", {})
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_edge("paper:111", "concept:a", "INTRODUCES")
        removed = store.delete_node("concept:a")
        assert removed == 1
        assert store.get_edges_from("paper:111") == []

    def test_deletes_both_directions(self, store: GraphStore) -> None:
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_node("concept:b", "concept", "B", {})
        store.upsert_node("concept:c", "concept", "C", {})
        store.upsert_edge("concept:a", "concept:b", "PREREQUISITE_OF")
        store.upsert_edge("concept:c", "concept:b", "PREREQUISITE_OF")
        store.upsert_edge("concept:b", "concept:c", "PREREQUISITE_OF")
        removed = store.delete_node("concept:b")
        assert removed == 3

    def test_raises_if_node_does_not_exist(self, store: GraphStore) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            store.delete_node("concept:nonexistent")


# ---------------------------------------------------------------------------
# update_node_properties
# ---------------------------------------------------------------------------


class TestUpdateNodeProperties:
    def test_merges_patch_into_existing(self, store: GraphStore) -> None:
        store.upsert_node("concept:x", "concept", "X", {"description": "Original", "tags": ["ml"]})
        store.update_node_properties("concept:x", {"description": "Updated"})
        node = store.get_node("concept:x")
        assert node.properties["description"] == "Updated"
        assert node.properties["tags"] == ["ml"]  # untouched

    def test_adds_new_keys(self, store: GraphStore) -> None:
        store.upsert_node("concept:x", "concept", "X", {"description": "Orig"})
        store.update_node_properties("concept:x", {"edited_by": "user"})
        node = store.get_node("concept:x")
        assert node.properties["edited_by"] == "user"
        assert node.properties["description"] == "Orig"

    def test_sets_updated_at(self, store: GraphStore) -> None:
        store.upsert_node("concept:x", "concept", "X", {})
        ts_before = store._conn.execute(
            "SELECT updated_at FROM nodes WHERE id = 'concept:x'"
        ).fetchone()[0]
        store.update_node_properties("concept:x", {"edited_by": "user"})
        ts_after = store._conn.execute(
            "SELECT updated_at FROM nodes WHERE id = 'concept:x'"
        ).fetchone()[0]
        assert ts_after >= ts_before

    def test_raises_if_node_does_not_exist(self, store: GraphStore) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            store.update_node_properties("concept:missing", {"x": 1})


# ---------------------------------------------------------------------------
# add_edge
# ---------------------------------------------------------------------------


class TestAddEdge:
    def test_creates_new_edge(self, store: GraphStore) -> None:
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_node("concept:b", "concept", "B", {})
        store.add_edge("concept:a", "concept:b", "PREREQUISITE_OF")
        edges = store.get_edges_from("concept:a", "PREREQUISITE_OF")
        assert len(edges) == 1
        assert edges[0].target_id == "concept:b"

    def test_creates_edge_with_weight_and_properties(self, store: GraphStore) -> None:
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_node("concept:b", "concept", "B", {})
        store.add_edge("concept:a", "concept:b", "PREREQUISITE_OF", weight=0.8, properties={"edited_by": "user"})
        edges = store.get_edges_from("concept:a")
        assert edges[0].weight == pytest.approx(0.8)
        assert edges[0].properties["edited_by"] == "user"

    def test_raises_if_edge_exists(self, store: GraphStore) -> None:
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_node("concept:b", "concept", "B", {})
        store.add_edge("concept:a", "concept:b", "PREREQUISITE_OF")
        with pytest.raises(ValueError, match="already exists"):
            store.add_edge("concept:a", "concept:b", "PREREQUISITE_OF")


# ---------------------------------------------------------------------------
# delete_edge
# ---------------------------------------------------------------------------


class TestDeleteEdge:
    def test_deletes_existing_edge(self, store: GraphStore) -> None:
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_node("concept:b", "concept", "B", {})
        store.upsert_edge("concept:a", "concept:b", "PREREQUISITE_OF")
        store.delete_edge("concept:a", "concept:b", "PREREQUISITE_OF")
        assert store.get_edges_from("concept:a", "PREREQUISITE_OF") == []

    def test_only_deletes_specific_edge(self, store: GraphStore) -> None:
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_node("concept:b", "concept", "B", {})
        store.upsert_edge("concept:a", "concept:b", "PREREQUISITE_OF")
        store.upsert_edge("concept:a", "concept:b", "BUILDS_ON")
        store.delete_edge("concept:a", "concept:b", "PREREQUISITE_OF")
        remaining = store.get_edges_from("concept:a")
        assert len(remaining) == 1
        assert remaining[0].relationship_type == "BUILDS_ON"

    def test_raises_if_edge_does_not_exist(self, store: GraphStore) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            store.delete_edge("concept:a", "concept:b", "PREREQUISITE_OF")


# ---------------------------------------------------------------------------
# get_technique_index / get_problem_index
# ---------------------------------------------------------------------------


class TestMVPIndexMethods:
    def test_get_technique_index_sorted(self, store: GraphStore) -> None:
        store.upsert_node("technique:xgboost", "technique", "XGBoost", {})
        store.upsert_node("technique:lightgbm", "technique", "LightGBM", {})
        store.upsert_node("technique:catboost", "technique", "CatBoost", {})
        index = store.get_technique_index()
        assert index == ["CatBoost", "LightGBM", "XGBoost"]

    def test_get_technique_index_excludes_other_types(self, store: GraphStore) -> None:
        store.upsert_node("technique:xgboost", "technique", "XGBoost", {})
        store.upsert_node("concept:boosting", "concept", "Boosting", {})
        index = store.get_technique_index()
        assert index == ["XGBoost"]

    def test_get_technique_index_empty(self, store: GraphStore) -> None:
        assert store.get_technique_index() == []

    def test_get_problem_index_sorted(self, store: GraphStore) -> None:
        store.upsert_node("problem:tabular", "problem", "Tabular prediction", {})
        store.upsert_node("problem:image", "problem", "Image classification", {})
        index = store.get_problem_index()
        assert index == ["Image classification", "Tabular prediction"]

    def test_get_problem_index_excludes_other_types(self, store: GraphStore) -> None:
        store.upsert_node("problem:tabular", "problem", "Tabular prediction", {})
        store.upsert_node("technique:xgboost", "technique", "XGBoost", {})
        index = store.get_problem_index()
        assert index == ["Tabular prediction"]

    def test_get_problem_index_empty(self, store: GraphStore) -> None:
        assert store.get_problem_index() == []


# ---------------------------------------------------------------------------
# get_nodes_created_since / get_edges_created_since
# ---------------------------------------------------------------------------


class TestTemporalQueries:
    def test_get_nodes_created_since_returns_recent(self, store: GraphStore) -> None:
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_node("concept:b", "concept", "B", {})
        # Both nodes were created "now", so querying from yesterday returns both
        nodes = store.get_nodes_created_since("2020-01-01")
        assert len(nodes) == 2

    def test_get_nodes_created_since_filters_by_type(self, store: GraphStore) -> None:
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_node("technique:b", "technique", "B", {})
        nodes = store.get_nodes_created_since("2020-01-01", node_type="concept")
        assert len(nodes) == 1
        assert nodes[0].node_type == "concept"

    def test_get_nodes_created_since_returns_empty_for_future(self, store: GraphStore) -> None:
        store.upsert_node("concept:a", "concept", "A", {})
        nodes = store.get_nodes_created_since("2099-01-01")
        assert nodes == []

    def test_get_nodes_created_since_no_type_filter(self, store: GraphStore) -> None:
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_node("problem:b", "problem", "B", {})
        store.upsert_node("technique:c", "technique", "C", {})
        nodes = store.get_nodes_created_since("2020-01-01")
        assert len(nodes) == 3

    def test_get_edges_created_since_returns_recent(self, store: GraphStore) -> None:
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_node("concept:b", "concept", "B", {})
        store.upsert_edge("concept:a", "concept:b", "PREREQUISITE_OF")
        edges = store.get_edges_created_since("2020-01-01")
        assert len(edges) == 1

    def test_get_edges_created_since_filters_by_type(self, store: GraphStore) -> None:
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_node("concept:b", "concept", "B", {})
        store.upsert_edge("concept:a", "concept:b", "PREREQUISITE_OF")
        store.upsert_edge("concept:a", "concept:b", "BUILDS_ON")
        edges = store.get_edges_created_since("2020-01-01", relationship_type="PREREQUISITE_OF")
        assert len(edges) == 1
        assert edges[0].relationship_type == "PREREQUISITE_OF"

    def test_get_edges_created_since_returns_empty_for_future(self, store: GraphStore) -> None:
        store.upsert_node("concept:a", "concept", "A", {})
        store.upsert_node("concept:b", "concept", "B", {})
        store.upsert_edge("concept:a", "concept:b", "PREREQUISITE_OF")
        edges = store.get_edges_created_since("2099-01-01")
        assert edges == []
