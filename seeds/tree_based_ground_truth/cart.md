---
name: CART
concept_type: Algorithm
what_it_is: CART (Classification and Regression Trees) is a binary recursive partitioning algorithm introduced by Breiman, Friedman, Olshen, and Stone in 1984 that builds decision trees by repeatedly splitting data on the feature and threshold that minimizes a purity criterion — Gini impurity for classification, mean squared error for regression.
what_problem_it_solves: Prior tree methods (ID3, AID) handled only categorical splits or classification tasks; CART unified both classification and regression under a single, coherent binary-split framework that could handle continuous features natively without discretization.
innovation_chain:
  - step: AID (Automatic Interaction Detection, Morgan & Sonquist 1963)
    why: Established the idea of recursive binary partitioning on data, but was limited to linear splits and continuous targets only.
  - step: ID3 (Quinlan 1979)
    why: Introduced information-theoretic split criteria for classification trees but restricted splits to categorical features and multi-way branches.
  - step: CART
    why: Unified classification and regression into one framework using binary splits, introduced Gini impurity as a differentiable-friendly criterion, and formalized cost-complexity pruning to control overfitting.
limitations:
  - Greedy top-down splitting finds locally optimal splits at each node but cannot backtrack, so it may miss globally better tree structures.
  - Binary splits on high-cardinality categorical features require exponential search over all possible subset partitions, making it computationally expensive without heuristics.
  - Unstable to small perturbations in training data — a slightly different sample can produce a dramatically different tree topology, leading to high variance predictions.
introduced_year: 1984
domain_tags:
  - supervised-learning
  - tree-based-models
  - interpretable-ml
source_refs:
  - "Breiman, Friedman, Olshen, Stone (1984) - Classification and Regression Trees (Wadsworth)"
  - "Hastie, Tibshirani, Friedman (2009) - The Elements of Statistical Learning, Ch. 9"
content_angles:
  - "The 1984 textbook that secretly runs modern ML: how CART's binary split + Gini impurity became the atomic unit inside Random Forests, Gradient Boosting, and XGBoost — four decades later"
  - "Cost-complexity pruning explained visually: the one CART mechanism that prevents decision trees from memorizing noise, and why most tutorials skip it entirely"
  - "CART vs. ID3 — the split criterion war of the 1980s: why Gini impurity beat information gain as the industry default, and whether that choice even matters mathematically"
relationships:
  - type: BELONGS_TO
    target: Decision Tree
    label: CART is the dominant concrete instantiation of the decision tree family, specifying binary splits, Gini/MSE criteria, and cost-complexity pruning as its defining implementation choices.
  - type: ALTERNATIVE_TO
    target: Information Gain (ID3/C4.5)
    label: CART uses Gini impurity and supports continuous features with binary splits, whereas ID3/C4.5 use entropy-based information gain with multi-way categorical splits.
  - type: PREREQUISITE_OF
    target: Random Forest
    label: Random Forest grows an ensemble of independently trained CART trees, so understanding CART's split mechanics and overfitting behavior is necessary to reason about why bagging helps.
  - type: PREREQUISITE_OF
    target: Gradient Boosting
    label: Gradient Boosting uses CART regression trees as weak learners to fit residuals, making CART's MSE-minimizing split criterion central to understanding how the ensemble accumulates predictive power.
  - type: BASELINE_OF
    target: Bagging
    label: CART's notorious high variance under data perturbation is the canonical motivating example that bagging was designed to reduce, making single CART trees the implicit baseline for ensemble methods.
  - type: INTRODUCES
    target: Extra Trees
    label: CART's deterministic best-split search is the mechanism that Extra Trees deliberately randomizes — replacing it with random thresholds — so CART defines the computational and variance baseline being relaxed.
  - type: BUILDS_ON
    target: Boosting
    label: Boosting algorithms from AdaBoost onward adopt CART stumps or shallow CART trees as weak learners, inheriting CART's splitting infrastructure while adding iterative reweighting on top.
---

## CART

CART formalized recursive binary partitioning into a complete, self-contained algorithm with four well-defined stages: tree growing, pruning, cross-validation-based tree selection, and prediction. The split at each internal node is chosen by exhaustive search over all features and all candidate thresholds, selecting the pair that maximally reduces the impurity of the resulting child nodes. For classification, [[Information Gain (ID3/C4.5)]]-style entropy was deliberately passed over in favor of Gini impurity — a measure of the probability that a randomly chosen pair of elements are misclassified — partly because it does not require a logarithm computation and is differentiable in a useful sense. For regression, CART minimizes the within-node sum of squared errors, predicting the mean of target values at each leaf. This dual-mode design — the same algorithmic skeleton handling both output types — was a genuine architectural leap over contemporaneous methods.

The subtlest and most practically important innovation in the original CART book is **cost-complexity pruning** (also called weakest-link pruning). Instead of stopping tree growth early using a threshold, CART grows a maximally large tree and then prunes it back by minimizing the penalized cost function R_α(T) = R(T) + α|T̃|, where R(T) is the tree's resubstitution error, |T̃| is the number of terminal nodes, and α is a complexity parameter swept over a sequence of values. For each α, there is a unique smallest minimizing subtree; cross-validation selects the α (and thus the subtree) with the best generalization estimate. This regularization-through-pruning philosophy directly anticipates the explicit L1/L2 regularization terms later introduced in [[XGBoost]]. The high variance of unpruned CART trees — a single different training point can restructure entire subtrees — was also the direct motivation for [[Bagging]] and [[Random Forest]], which average many CART trees trained on bootstrap samples to suppress that variance.

CART's position in the tree lineage is that of a load-bearing ancestor: virtually every modern tree-based method treats a CART tree (or a restricted version of it) as its atomic component. [[Gradient Boosting]] fits shallow CART regression trees to negative gradients. [[AdaBoost]] in its most common form uses depth-1 CART stumps. [[Extra Trees]] keeps CART's splitting structure but randomizes the threshold selection. [[LightGBM]] and [[CatBoost]] replace CART's level-wise growth with leaf-wise strategies and histogram approximations, yet they still optimize MSE or log-loss splits inherited directly from CART. Understanding which design choices in CART are load-bearing (binary splits, impurity minimization) versus incidental (exact threshold search, level-wise growth) is the conceptual key to reading the entire subsequent history of tree-based ML.