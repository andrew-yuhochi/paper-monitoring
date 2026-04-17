"""Hand-crafted tree-based models prototype for Milestone 1 validation.

Creates ~17 nodes (1 Problem + 6 Techniques + 8 Concepts + 1 Category + 2 Papers)
connected with all 7 relationship types. All nodes are marked edited_by: "user".

Idempotent — uses upsert_node/upsert_edge, safe to re-run.

Usage:
    python -m src.seeds.tree_based_prototype
"""

import logging
import sys
from pathlib import Path

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import settings
from src.store.graph_store import GraphStore
from src.utils.normalize import normalize_concept_name

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Node definitions
# ═══════════════════════════════════════════════════════════════════════

PROBLEM_NODES = [
    {
        "label": "How to build accurate predictive models from tabular data",
        "properties": {
            "description": "Given structured/tabular datasets with mixed feature types "
            "(numeric, categorical, ordinal), build models that maximise predictive "
            "accuracy while remaining interpretable and robust to overfitting.",
            "edited_by": "user",
        },
    },
]

TECHNIQUE_NODES = [
    {
        "label": "Decision Tree",
        "properties": {
            "approach": "Recursively partition the feature space using axis-aligned splits "
            "that maximise a purity criterion (information gain or Gini impurity). "
            "Each leaf holds a prediction.",
            "innovation_type": "architecture",
            "practical_relevance": "Baseline model for tabular data. Highly interpretable, "
            "handles mixed feature types natively, requires no feature scaling.",
            "limitations": "High variance — small data changes produce very different trees. "
            "Prone to overfitting without pruning or ensemble wrapping.",
            "edited_by": "user",
        },
    },
    {
        "label": "Random Forest",
        "properties": {
            "approach": "Train an ensemble of decorrelated decision trees on bootstrap samples "
            "with random feature subsets at each split. Aggregate predictions by majority "
            "vote (classification) or averaging (regression).",
            "innovation_type": "architecture",
            "practical_relevance": "Go-to model for tabular data when interpretability is secondary. "
            "Strong out-of-the-box accuracy, parallelisable, few hyperparameters.",
            "limitations": "Cannot extrapolate beyond training range. Feature importance can be "
            "biased toward high-cardinality features.",
            "edited_by": "user",
        },
    },
    {
        "label": "Gradient Boosting",
        "properties": {
            "approach": "Sequentially fit shallow decision trees to the negative gradient "
            "(pseudo-residuals) of a differentiable loss function. Each tree corrects the "
            "errors of the cumulative ensemble.",
            "innovation_type": "architecture",
            "practical_relevance": "State-of-the-art for structured data competitions. "
            "Flexible loss functions enable classification, regression, and ranking.",
            "limitations": "Sequential training is slow. Sensitive to learning rate and "
            "number of boosting rounds — requires careful tuning.",
            "edited_by": "user",
        },
    },
    {
        "label": "XGBoost",
        "properties": {
            "approach": "Optimised gradient boosting with regularised objective (L1/L2 on leaf "
            "weights), approximate split finding via weighted quantile sketch, sparsity-aware "
            "handling of missing values, and column-block parallel tree construction.",
            "innovation_type": "architecture",
            "practical_relevance": "Dominant algorithm on Kaggle for tabular data (2015-2020). "
            "Excellent speed/accuracy trade-off, built-in handling of missing values.",
            "limitations": "Many hyperparameters to tune. Incremental/online learning is limited.",
            "edited_by": "user",
        },
    },
    {
        "label": "LightGBM",
        "properties": {
            "approach": "Gradient boosting with leaf-wise (best-first) tree growth, "
            "Gradient-based One-Side Sampling (GOSS) to skip low-gradient instances, "
            "and Exclusive Feature Bundling (EFB) to reduce dimensionality.",
            "innovation_type": "architecture",
            "practical_relevance": "Fastest gradient boosting implementation for large datasets. "
            "Handles categorical features natively. Memory-efficient.",
            "limitations": "Leaf-wise growth can overfit on small datasets. "
            "Sensitive to num_leaves parameter.",
            "edited_by": "user",
        },
    },
    {
        "label": "CatBoost",
        "properties": {
            "approach": "Gradient boosting with ordered boosting (permutation-driven) to "
            "reduce prediction shift, symmetric (oblivious) decision trees, and native "
            "target encoding for categorical features.",
            "innovation_type": "architecture",
            "practical_relevance": "Best out-of-the-box model for datasets with many categorical "
            "features. Minimal preprocessing needed.",
            "limitations": "Slower training than LightGBM due to ordered boosting. "
            "Symmetric tree constraint limits expressiveness on some tasks.",
            "edited_by": "user",
        },
    },
]

CONCEPT_NODES = [
    {
        "label": "Information Gain",
        "properties": {
            "description": "The reduction in entropy (uncertainty) achieved by partitioning "
            "a dataset on a given feature. Used as the split criterion in ID3 and C4.5 "
            "decision trees. Calculated as the difference between the parent node entropy "
            "and the weighted average entropy of child nodes.",
            "domain_tags": ["decision trees", "information theory"],
            "edited_by": "user",
        },
    },
    {
        "label": "Gini Impurity",
        "properties": {
            "description": "A measure of how often a randomly chosen element would be "
            "incorrectly classified if labelled according to the class distribution in "
            "the node. Used as the default split criterion in CART trees and scikit-learn. "
            "Ranges from 0 (pure) to 0.5 (maximally impure for binary classification).",
            "domain_tags": ["decision trees", "classification"],
            "edited_by": "user",
        },
    },
    {
        "label": "Bagging",
        "properties": {
            "description": "Bootstrap Aggregating — train multiple models on different bootstrap "
            "samples (random samples with replacement) of the training data, then aggregate "
            "their predictions. Reduces variance without increasing bias. The foundation of "
            "Random Forests.",
            "domain_tags": ["ensemble methods", "variance reduction"],
            "edited_by": "user",
        },
    },
    {
        "label": "Boosting",
        "properties": {
            "description": "A meta-algorithm that combines weak learners sequentially, where each "
            "subsequent learner focuses on the errors of the previous ensemble. Unlike bagging, "
            "boosting reduces bias. Key variants: AdaBoost (reweight misclassified samples), "
            "gradient boosting (fit to residuals).",
            "domain_tags": ["ensemble methods", "bias reduction"],
            "edited_by": "user",
        },
    },
    {
        "label": "Ensemble Methods",
        "properties": {
            "description": "Techniques that combine multiple models to produce a single prediction "
            "that is typically more accurate and robust than any individual model. The two main "
            "paradigms are bagging (parallel, variance reduction) and boosting (sequential, bias "
            "reduction). Stacking and blending are additional ensemble strategies.",
            "domain_tags": ["machine learning", "model combination"],
            "edited_by": "user",
        },
    },
    {
        "label": "Feature Importance",
        "properties": {
            "description": "A score indicating how useful each feature is for making predictions. "
            "In tree-based models, commonly measured by impurity decrease (mean decrease in Gini "
            "or entropy across all splits using that feature) or permutation importance (drop in "
            "accuracy when a feature's values are shuffled).",
            "domain_tags": ["interpretability", "feature selection"],
            "edited_by": "user",
        },
    },
    {
        "label": "Overfitting",
        "properties": {
            "description": "When a model learns the training data's noise and idiosyncrasies rather "
            "than the underlying pattern, resulting in excellent training performance but poor "
            "generalisation to unseen data. In decision trees, overfitting manifests as very deep "
            "trees with many leaves that memorise training examples.",
            "domain_tags": ["model selection", "generalisation"],
            "edited_by": "user",
        },
    },
    {
        "label": "Pruning",
        "properties": {
            "description": "Removing branches from a fully grown decision tree to reduce complexity "
            "and prevent overfitting. Pre-pruning (early stopping) limits tree growth during "
            "training by setting max depth, min samples per leaf, etc. Post-pruning (cost-complexity "
            "pruning) grows a full tree then removes branches that contribute least.",
            "domain_tags": ["decision trees", "regularisation"],
            "edited_by": "user",
        },
    },
]

CATEGORY_NODES = [
    {
        "label": "Tree-Based Models",
        "properties": {
            "description": "Models built on decision tree structures — including single trees "
            "and ensemble methods (bagging, boosting) that combine multiple trees.",
            "edited_by": "user",
        },
    },
]

PAPER_NODES = [
    {
        "label": "Random Forests (Breiman 2001)",
        "arxiv_id": None,  # pre-arXiv publication
        "properties": {
            "description": "Leo Breiman's seminal paper introducing Random Forests — "
            "an ensemble of decision trees trained on bootstrap samples with random "
            "feature subsets at each split. Published in Machine Learning, 2001.",
            "authors": ["Leo Breiman"],
            "published_date": "2001-10-01",
            "edited_by": "user",
        },
    },
    {
        "label": "XGBoost: A Scalable Tree Boosting System (Chen & Guestrin 2016)",
        "arxiv_id": "1603.02754",
        "properties": {
            "description": "Introduces XGBoost — a scalable end-to-end tree boosting system "
            "with a regularised learning objective, approximate split finding, sparsity-aware "
            "algorithm, and cache-aware parallel computation. KDD 2016.",
            "authors": ["Tianqi Chen", "Carlos Guestrin"],
            "published_date": "2016-03-09",
            "arxiv_url": "https://arxiv.org/abs/1603.02754",
            "edited_by": "user",
        },
    },
]

# ═══════════════════════════════════════════════════════════════════════
# Edge definitions (all 7 relationship types)
# ═══════════════════════════════════════════════════════════════════════

EDGES = [
    # PREREQUISITE_OF: concept -> concept
    ("concept", "Information Gain", "technique", "Decision Tree", "PREREQUISITE_OF"),
    ("concept", "Gini Impurity", "technique", "Decision Tree", "PREREQUISITE_OF"),
    ("concept", "Overfitting", "concept", "Pruning", "PREREQUISITE_OF"),
    ("concept", "Bagging", "concept", "Ensemble Methods", "PREREQUISITE_OF"),
    ("concept", "Boosting", "concept", "Ensemble Methods", "PREREQUISITE_OF"),

    # ADDRESSES: technique -> problem
    ("technique", "Decision Tree", "problem", "How to build accurate predictive models from tabular data", "ADDRESSES"),
    ("technique", "Random Forest", "problem", "How to build accurate predictive models from tabular data", "ADDRESSES"),
    ("technique", "Gradient Boosting", "problem", "How to build accurate predictive models from tabular data", "ADDRESSES"),
    ("technique", "XGBoost", "problem", "How to build accurate predictive models from tabular data", "ADDRESSES"),
    ("technique", "LightGBM", "problem", "How to build accurate predictive models from tabular data", "ADDRESSES"),
    ("technique", "CatBoost", "problem", "How to build accurate predictive models from tabular data", "ADDRESSES"),

    # BASELINE_OF: technique -> technique (established baseline)
    ("technique", "Decision Tree", "technique", "Random Forest", "BASELINE_OF"),
    ("technique", "Decision Tree", "technique", "Gradient Boosting", "BASELINE_OF"),

    # ALTERNATIVE_TO: technique <-> technique (bidirectional)
    ("technique", "XGBoost", "technique", "LightGBM", "ALTERNATIVE_TO"),
    ("technique", "XGBoost", "technique", "CatBoost", "ALTERNATIVE_TO"),
    ("technique", "LightGBM", "technique", "CatBoost", "ALTERNATIVE_TO"),

    # BUILDS_ON: technique -> concept
    ("technique", "Random Forest", "concept", "Bagging", "BUILDS_ON"),
    ("technique", "Random Forest", "technique", "Decision Tree", "BUILDS_ON"),
    ("technique", "Gradient Boosting", "concept", "Boosting", "BUILDS_ON"),
    ("technique", "XGBoost", "concept", "Boosting", "BUILDS_ON"),
    ("technique", "LightGBM", "concept", "Boosting", "BUILDS_ON"),
    ("technique", "CatBoost", "concept", "Boosting", "BUILDS_ON"),

    # BELONGS_TO: technique/concept -> category
    ("technique", "Decision Tree", "category", "Tree-Based Models", "BELONGS_TO"),
    ("technique", "Random Forest", "category", "Tree-Based Models", "BELONGS_TO"),
    ("technique", "Gradient Boosting", "category", "Tree-Based Models", "BELONGS_TO"),
    ("technique", "XGBoost", "category", "Tree-Based Models", "BELONGS_TO"),
    ("technique", "LightGBM", "category", "Tree-Based Models", "BELONGS_TO"),
    ("technique", "CatBoost", "category", "Tree-Based Models", "BELONGS_TO"),
    ("concept", "Ensemble Methods", "category", "Tree-Based Models", "BELONGS_TO"),

    # INTRODUCES: paper -> technique
    ("paper", "Random Forests (Breiman 2001)", "technique", "Random Forest", "INTRODUCES"),
    ("paper", "XGBoost: A Scalable Tree Boosting System (Chen & Guestrin 2016)", "technique", "XGBoost", "INTRODUCES"),
]


def _node_id(node_type: str, label: str) -> str:
    """Generate a node ID from type and label."""
    return f"{node_type}:{normalize_concept_name(label)}"


def seed_tree_based_prototype(store: GraphStore) -> dict:
    """Seed the tree-based models prototype graph.

    Returns a summary dict with counts of nodes and edges created.
    """
    node_count = 0
    edge_count = 0

    # --- Create nodes ---
    for node in PROBLEM_NODES:
        nid = _node_id("problem", node["label"])
        store.upsert_node(nid, "problem", node["label"], node["properties"])
        node_count += 1
        logger.info("Seeded problem: %s", node["label"])

    for node in TECHNIQUE_NODES:
        nid = _node_id("technique", node["label"])
        store.upsert_node(nid, "technique", node["label"], node["properties"])
        node_count += 1
        logger.info("Seeded technique: %s", node["label"])

    for node in CONCEPT_NODES:
        nid = _node_id("concept", node["label"])
        store.upsert_node(nid, "concept", node["label"], node["properties"])
        node_count += 1
        logger.info("Seeded concept: %s", node["label"])

    for node in CATEGORY_NODES:
        nid = _node_id("category", node["label"])
        store.upsert_node(nid, "category", node["label"], node["properties"])
        node_count += 1
        logger.info("Seeded category: %s", node["label"])

    for node in PAPER_NODES:
        if node["arxiv_id"]:
            nid = f"paper:{node['arxiv_id']}"
        else:
            nid = _node_id("paper", node["label"])
        store.upsert_node(nid, "paper", node["label"], node["properties"])
        node_count += 1
        logger.info("Seeded paper: %s", node["label"])

    # --- Create edges ---
    for src_type, src_label, tgt_type, tgt_label, rel_type in EDGES:
        # Resolve node IDs
        if src_type == "paper":
            paper = next((p for p in PAPER_NODES if p["label"] == src_label), None)
            src_id = f"paper:{paper['arxiv_id']}" if paper and paper["arxiv_id"] else _node_id("paper", src_label)
        else:
            src_id = _node_id(src_type, src_label)

        if tgt_type == "paper":
            paper = next((p for p in PAPER_NODES if p["label"] == tgt_label), None)
            tgt_id = f"paper:{paper['arxiv_id']}" if paper and paper["arxiv_id"] else _node_id("paper", tgt_label)
        else:
            tgt_id = _node_id(tgt_type, tgt_label)

        store.upsert_edge(src_id, tgt_id, rel_type, properties={"edited_by": "user"})
        edge_count += 1
        logger.info("Seeded edge: %s -[%s]-> %s", src_label, rel_type, tgt_label)

    summary = {
        "nodes": node_count,
        "edges": edge_count,
        "problems": len(PROBLEM_NODES),
        "techniques": len(TECHNIQUE_NODES),
        "concepts": len(CONCEPT_NODES),
        "categories": len(CATEGORY_NODES),
        "papers": len(PAPER_NODES),
    }
    logger.info("Prototype seeding complete: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    store = GraphStore(settings.db_path)
    try:
        result = seed_tree_based_prototype(store)
        print(f"\nTree-based prototype seeded successfully:")
        print(f"  Nodes: {result['nodes']} "
              f"({result['problems']}P + {result['techniques']}T + "
              f"{result['concepts']}C + {result['categories']}Cat + {result['papers']}Paper)")
        print(f"  Edges: {result['edges']}")
        print(f"\nOpen the dashboard to explore: bash run_dashboard.sh")
    finally:
        store.close()
