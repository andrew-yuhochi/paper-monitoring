# Unit tests for src/services/ground_truth_loader.py (TASK-M1-003).
# Uses in-memory SQLite via GraphStore(:memory:) — no file I/O to a real DB.
# Fixture .md files are written to pytest's tmp_path.

import logging
from pathlib import Path

import pytest

from src.models.concepts import Concept, ConceptRelationship
from src.services.ground_truth_loader import load_ground_truth
from src.store.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONCEPT_A_MD = """\
---
name: Decision Tree
concept_type: Technique
what_it_is: A tree-shaped model that splits data on feature thresholds.
what_problem_it_solves: Provides interpretable, non-parametric classification/regression.
innovation_chain:
  - step: CART
    why: Provides the binary splitting algorithm.
limitations:
  - Prone to overfitting on training data.
introduced_year: 1986
domain_tags:
  - Supervised Learning
  - Tree-based
source_refs:
  - "Breiman et al. 1984 - CART"
content_angles:
  - "Decision trees demystified: how information gain chooses splits"
relationships:
  - type: BELONGS_TO
    target: Random Forest
    label: Decision Trees are the base learners used inside a Random Forest ensemble.
---

## Decision Tree

A [[Decision Tree]] recursively partitions the feature space.
"""

_CONCEPT_B_MD = """\
---
name: Random Forest
concept_type: Technique
what_it_is: An ensemble of decision trees trained on bootstrap samples.
what_problem_it_solves: Reduces variance compared to a single decision tree.
innovation_chain:
  - step: Decision Tree
    why: Base learner of the ensemble.
limitations:
  - Less interpretable than a single tree.
introduced_year: 2001
domain_tags:
  - Ensemble Learning
  - Supervised Learning
source_refs:
  - "Breiman 2001 - Random Forests"
content_angles:
  - "Why averaging trees removes variance: the bias-variance tradeoff explained"
relationships:
  - type: BUILDS_ON
    target: Decision Tree
    label: Random Forest builds an ensemble of Decision Trees trained on bootstrap samples.
---

## Random Forest

[[Random Forest]] averages predictions from many [[Decision Tree]] models.
"""

_CONCEPT_C_MD = """\
---
name: Gradient Boosting
concept_type: Technique
what_it_is: A boosting algorithm that fits trees sequentially to the residuals.
what_problem_it_solves: Combines weak learners into a strong predictor with low bias.
innovation_chain:
  - step: Decision Tree
    why: Weak learner fitted to residuals each round.
limitations:
  - Sensitive to outliers in the loss function.
  - Can overfit if too many rounds are used.
introduced_year: 2001
domain_tags:
  - Ensemble Learning
  - Boosting
source_refs:
  - "Friedman 2001 - Gradient Boosting Machine"
content_angles:
  - "From residuals to gradient descent in function space: the Friedman insight"
relationships:
  - type: BUILDS_ON
    target: Decision Tree
    label: Gradient Boosting uses decision trees as weak learners fitted to pseudo-residuals.
  - type: ALTERNATIVE_TO
    target: Random Forest
    label: Both reduce variance via ensembling but use different strategies (sequential vs. parallel).
---

## Gradient Boosting

[[Gradient Boosting]] corrects errors of previous trees sequentially.
"""


def _write_fixtures(tmp_path: Path) -> Path:
    """Write 3 concept .md files to tmp_path and return the directory."""
    (tmp_path / "decision_tree.md").write_text(_CONCEPT_A_MD, encoding="utf-8")
    (tmp_path / "random_forest.md").write_text(_CONCEPT_B_MD, encoding="utf-8")
    (tmp_path / "gradient_boosting.md").write_text(_CONCEPT_C_MD, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store() -> GraphStore:
    """Fresh in-memory GraphStore for each test."""
    return GraphStore(":memory:")


@pytest.fixture()
def gt_dir(tmp_path: Path) -> Path:
    """Temp directory populated with 3 concept fixture files."""
    return _write_fixtures(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadGroundTruth:
    def test_concepts_and_relationships_loaded_correctly(
        self, gt_dir: Path, store: GraphStore
    ) -> None:
        """3 concept files → 3 concepts + correct relationship count in the DB."""
        concepts_n, rels_n = load_ground_truth(gt_dir, store)

        assert concepts_n == 3

        # Verify all three concepts are retrievable by name.
        dt = store.get_concept_by_name("Decision Tree", "default")
        rf = store.get_concept_by_name("Random Forest", "default")
        gb = store.get_concept_by_name("Gradient Boosting", "default")

        assert dt is not None, "Decision Tree not found in store"
        assert rf is not None, "Random Forest not found in store"
        assert gb is not None, "Gradient Boosting not found in store"

        # Spot-check field mapping.
        assert dt.concept_type == "Technique"
        assert dt.introduced_year == 1986
        assert "Supervised Learning" in dt.domain_tags
        assert dt.source == "manual"

        # Relationship count:
        # decision_tree → Random Forest (1)
        # random_forest → Decision Tree (1)
        # gradient_boosting → Decision Tree (1), gradient_boosting → Random Forest (1)
        # Total = 4
        assert rels_n == 4, f"Expected 4 relationships, got {rels_n}"

    def test_idempotent_rerun(self, gt_dir: Path, store: GraphStore) -> None:
        """Running the loader twice against the same store yields the same concept count."""
        concepts_n_first, _ = load_ground_truth(gt_dir, store)
        concepts_n_second, _ = load_ground_truth(gt_dir, store)

        assert concepts_n_first == 3
        assert concepts_n_second == 3

        # List all concepts — should still be 3, not 6.
        all_concepts = store.list_concepts("default")
        assert len(all_concepts) == 3

    def test_unresolved_relationship_target_logged_and_skipped(
        self, tmp_path: Path, store: GraphStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A relationship pointing to a non-existent concept is WARNING-logged and skipped."""
        md_content = """\
---
name: AdaBoost
concept_type: Technique
what_it_is: Adaptive Boosting — re-weights misclassified samples each round.
what_problem_it_solves: Builds a strong classifier from many weak ones.
limitations:
  - Sensitive to noisy data and outliers.
introduced_year: 1995
domain_tags:
  - Ensemble Learning
  - Boosting
source_refs:
  - "Freund & Schapire 1997 - A Decision-Theoretic Generalization of On-Line Learning"
content_angles:
  - "How AdaBoost learns from its mistakes: exponential loss demystified"
relationships:
  - type: BUILDS_ON
    target: NonExistentConcept
    label: This relationship should be skipped because the target does not exist.
  - type: BELONGS_TO
    target: AdaBoost
    label: Self-referential edge — also unresolved because AdaBoost is the only concept here, but target lookup is against known_names built in first pass so this IS valid.
---

## AdaBoost
"""
        # Note: 'AdaBoost' IS in known_names (it's the concept we loaded), but
        # 'NonExistentConcept' is not. The self-referential BELONGS_TO should
        # resolve; the BUILDS_ON with NonExistentConcept should not.
        (tmp_path / "adaboost.md").write_text(md_content, encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="src.services.ground_truth_loader"):
            concepts_n, rels_n = load_ground_truth(tmp_path, store)

        assert concepts_n == 1

        # Both relationships are skipped: NonExistentConcept is not in known_names,
        # and the self-referential edge (AdaBoost → AdaBoost) is rejected because
        # the schema enforces source_concept_id != target_concept_id.
        assert rels_n == 0, f"Expected 0 resolved relationships, got {rels_n}"

        # Confirm the WARNING was logged for the unresolved target.
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "NonExistentConcept" in msg for msg in warning_messages
        ), f"Expected WARNING mentioning 'NonExistentConcept', got: {warning_messages}"

    def test_empty_directory_returns_zero_counts(
        self, tmp_path: Path, store: GraphStore
    ) -> None:
        """An empty directory returns (0, 0) without crashing."""
        concepts_n, rels_n = load_ground_truth(tmp_path, store)
        assert concepts_n == 0
        assert rels_n == 0

    def test_user_id_propagated(self, gt_dir: Path, store: GraphStore) -> None:
        """Concepts are stored under the provided user_id."""
        load_ground_truth(gt_dir, store, user_id="test-user")

        concepts = store.list_concepts("test-user")
        assert len(concepts) == 3

        # Default user should have nothing.
        default_concepts = store.list_concepts("default")
        assert len(default_concepts) == 0
