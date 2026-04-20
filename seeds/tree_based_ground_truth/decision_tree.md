---
name: Decision Tree
concept_type: Algorithm
what_it_is: A Decision Tree is a supervised learning algorithm that recursively partitions the feature space into axis-aligned rectangular regions by selecting splits that maximize a purity criterion (e.g., information gain, Gini impurity, or variance reduction), producing a hierarchical tree of binary or multi-way decisions.
what_problem_it_solves: It addresses the need for an interpretable, non-parametric model that can capture non-linear decision boundaries and handle both categorical and continuous features without requiring feature scaling or distributional assumptions.
innovation_chain:
- step: Boolean concept learning (Hunt's CLS, 1966)
  why: Established the idea of recursively partitioning a hypothesis space to classify examples, forming the conceptual ancestor of tree-based splitting.
- step: Information Gain (ID3/C4.5)
  why: Provided a principled entropy-based criterion for choosing which feature to split on at each node, making greedy tree construction statistically grounded.
- step: Decision Tree
  why: Unified recursive binary partitioning with a learnable split threshold at each node, enabling application to continuous features and regression targets via CART's mean-squared-error criterion.
limitations:
- 'High variance: a small change in training data can produce a completely different tree topology, because each greedy split is locally optimal but not globally optimal.'
- Axis-aligned splits cannot efficiently approximate diagonal decision boundaries — a boundary at 45° requires an exponential number of splits to approximate.
- Prone to overfitting deep trees on small datasets; pruning (cost-complexity or reduced-error) mitigates this but introduces a tuning hyperparameter with non-trivial sensitivity.
introduced_year: 1986
domain_tags:
- supervised-learning
- interpretable-ml
- classification
- regression
source_refs:
- Quinlan 1986 - Induction of Decision Trees (Machine Learning)
- Breiman, Friedman, Olshen, Stone 1984 - Classification and Regression Trees (CART)
- 'Quinlan 1993 - C4.5: Programs for Machine Learning'
content_angles:
- 'The ''good enough'' algorithm that launched an entire field: how the greedy split heuristic in Decision Trees, despite provably not finding the globally optimal tree, became the engine inside Random Forest, XGBoost, and LightGBM'
- 'Visual walkthrough: how a Decision Tree actually carves up feature space — and why the axis-aligned constraint is both its greatest strength (interpretability) and its deepest flaw (diagonal boundary failure)'
- 'The forgotten rivalry: ID3 vs. CART in the 1980s — how two competing split criteria (information gain vs. Gini impurity) led to different algorithmic dynasties, and which one secretly powers most modern boosting frameworks'
relationships:
- type: PREREQUISITE_OF
  target: Random Forest
  label: Random Forest's core mechanism — aggregating independently trained trees — requires understanding how a single Decision Tree is grown, split, and overfit before ensemble correction makes sense.
- type: PREREQUISITE_OF
  target: Gradient Boosting
  label: Gradient Boosting uses Decision Trees as weak learners fit to pseudo-residuals, so understanding tree structure, depth control, and leaf predictions is essential before the boosting framework can be interpreted.
- type: BASELINE_OF
  target: Bagging
  label: Decision Trees are the canonical base learner used to motivate Bagging, since their high variance (low bias) makes them the ideal candidate for variance reduction through averaging.
- type: INTRODUCES
  target: CART
  label: The Decision Tree framework introduced the concept of binary recursive partitioning on continuous features, which CART formalized into a unified regression-and-classification algorithm with cost-complexity pruning.
- type: BUILDS_ON
  target: Information Gain (ID3/C4.5)
  label: Decision Tree construction in ID3/C4.5 builds directly on information gain as the split criterion, applying it greedily at each node to select the most discriminative feature.
- type: BASELINE_OF
  target: Extra Trees
  label: Extra Trees is benchmarked against standard Decision Trees to demonstrate that additional randomization in split selection can reduce variance further while maintaining comparable bias.
- type: BELONGS_TO
  target: Isolation Forest
  label: Isolation Forest belongs to the same tree-based model family as Decision Trees, reusing the recursive partitioning mechanism but repurposing split randomness for anomaly scoring rather than class purity.
---

## Decision Tree

A Decision Tree learns a prediction function by recursively asking binary questions about features. At each internal node, an algorithm selects a feature $x_j$ and a threshold $t$ to minimize an impurity criterion — Gini impurity or entropy for classification, mean squared error for regression — across the resulting child partitions. The process continues until a stopping condition is met (maximum depth, minimum samples per leaf, or negligible impurity reduction), at which point leaf nodes store the majority class or mean target value. The elegance of this formulation is that the final model is a piecewise-constant function over axis-aligned hyperrectangles in feature space, directly readable as a set of human-interpretable if-then rules. The two dominant historical instantiations are [[Information Gain (ID3/C4.5)]], which uses entropy and handles multi-way categorical splits, and [[CART]], which enforces strictly binary splits and introduced the cost-complexity pruning parameter $\alpha$ for post-hoc tree simplification.

The central algorithmic challenge in Decision Tree learning is that finding the globally optimal tree is NP-hard (the number of possible trees grows super-exponentially with depth and feature count), so all practical algorithms use greedy top-down induction: each split is locally optimal given the current node's data, with no backtracking. This local greediness is the root cause of the algorithm's notorious instability — two datasets differing by a single point can trigger a different first split, cascading into an entirely different tree. This high-variance, low-bias characteristic turns out to be the very property that makes Decision Trees ideal as base learners for ensemble methods: [[Bagging]] reduces variance by averaging many independently perturbed trees, while [[Boosting]] corrects residual bias by sequentially fitting trees to errors. [[Random Forest]] extends Bagging by additionally subsampling features at each split, decorrelating trees and reducing ensemble variance below what Bagging alone achieves.

The legacy of the Decision Tree extends throughout the modern tree-based model landscape. [[Gradient Boosting]], [[XGBoost]], [[LightGBM]], and [[CatBoost]] all use shallow Decision Trees (often depth 3–8) as their weak learners, inheriting the axis-aligned split structure but layering gradient-based optimization over the ensemble. [[Oblivious Trees]] impose the additional constraint that the same feature-threshold pair is used at every node at a given depth, sacrificing flexibility for cache-efficient inference. Even [[Isolation Forest]] reuses the recursive partitioning skeleton, replacing the purity-maximizing split criterion with random splits to exploit the insight that anomalies are isolated in fewer partitions than normal points. Understanding the Decision Tree — its split criteria, its greedy construction, and its overfitting behavior — is therefore not merely historical context but the mechanical prerequisite for every advanced tree-based algorithm in use today.