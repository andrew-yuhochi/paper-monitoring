---
name: Random Forest
concept_type: Algorithm
what_it_is: Random Forest is an ensemble learning algorithm that constructs a large number of decorrelated decision trees at training time and outputs the majority vote (classification) or mean prediction (regression) across all trees.
what_problem_it_solves: Individual decision trees overfit training data and are highly sensitive to small perturbations in the dataset; Random Forest reduces variance without substantially increasing bias by combining many trees trained on bootstrapped samples with randomized feature subsets.
innovation_chain:
  - step: Decision Tree
    why: Provides the base learner whose high variance is the fundamental problem Random Forest is engineered to tame.
  - step: Bagging
    why: Established the principle of training base learners on bootstrap samples to reduce variance, which Random Forest directly extends.
  - step: Random Forest
    why: Adds feature-level randomization at each split — selecting a random subset of sqrt(p) features — to decorrelate trees beyond what bagging alone achieves, breaking the dominance of strong predictors.
limitations:
  - Predictions are not easily interpretable as a single logical rule set; the ensemble of hundreds of trees sacrifices the white-box transparency of a single CART tree.
  - Memory and inference cost scale linearly with the number of trees, making very large forests (1000+ trees) impractical on edge devices or latency-sensitive APIs.
  - Random Forest does not extrapolate beyond the range of training data; all predictions are bounded by the observed target values in the training set, making it unsuitable for out-of-distribution regression tasks.
introduced_year: 2001
domain_tags:
  - ensemble_methods
  - supervised_learning
  - tree_based_models
source_refs:
  - "Breiman 2001 - Random Forests (Machine Learning, Vol. 45)"
  - "Breiman 1996 - Bagging Predictors (Machine Learning, Vol. 24)"
  - "Ho 1995 - Random Decision Forests (ICDAR)"
content_angles:
  - "The one hyperparameter that makes Random Forest work: why sqrt(p) features per split decorrelates trees better than any tuning of tree depth — with the variance-bias math shown visually"
  - "Random Forest vs. Gradient Boosting on tabular data: a practitioner's decision guide covering dataset size, training time, interpretability needs, and the specific regimes where RF still wins in 2024"
  - "How Leo Breiman unified two separate ideas — bootstrap aggregating and random subspace method — into a single algorithm in 2001, and why the ML community almost ignored it for four years before Kaggle made it famous"
relationships:
  - type: BUILDS_ON
    target: Bagging
    label: Random Forest extends Bagging by adding random feature subset selection at each split to further decorrelate the ensemble trees beyond bootstrap sampling alone.
  - type: BUILDS_ON
    target: Decision Tree
    label: Random Forest uses unpruned CART or similar decision trees as its base learners, deliberately exploiting their high variance to benefit from ensemble averaging.
  - type: ADDRESSES
    target: Decision Tree
    label: Random Forest was explicitly designed to correct the overfitting and instability that plagues single decision trees trained to full depth.
  - type: ALTERNATIVE_TO
    target: Gradient Boosting
    label: Both Random Forest and Gradient Boosting produce strong tabular predictions from decision trees but via orthogonal strategies — parallel variance reduction versus sequential bias reduction.
  - type: BASELINE_OF
    target: XGBoost
    label: Random Forest serves as the standard benchmark that XGBoost and other boosting methods are compared against when demonstrating superiority on structured datasets.
  - type: PREREQUISITE_OF
    target: Extra Trees
    label: Understanding Random Forest's feature randomization mechanism is necessary to appreciate how Extra Trees further randomizes split thresholds to reduce variance even more aggressively.
  - type: BELONGS_TO
    target: Bagging
    label: Random Forest is a specialized and dominant member of the Bagging family of ensemble methods.
---

## Random Forest

Random Forest, introduced by Leo Breiman in his landmark 2001 paper, is built on two foundational pillars: [[Bagging]] (bootstrap aggregating) and the random subspace method. [[Bagging]] alone reduces variance by training each tree on a different bootstrap sample of the training data, but trees trained on overlapping datasets remain correlated — when one tree makes an error driven by a dominant feature, many others will too. Breiman's critical innovation was to inject an additional layer of randomness at each node: rather than searching all *p* features for the best split, each node considers only a random subset of size *√p* for classification or *p/3* for regression. This decorrelation mechanism is what gives Random Forest its characteristic bias-variance profile — it accepts a small increase in bias per tree in exchange for a dramatic reduction in the correlation between trees, and by the bias-variance decomposition of ensemble error, the ensemble variance drops toward zero as the number of trees grows.

The mathematical guarantee underpinning Random Forest is Breiman's generalization bound: the ensemble error is bounded by a function of the mean correlation between trees and the average strength (negative error rate) of individual trees. Minimizing correlation — which feature randomization directly achieves — is thus as important as maximizing individual tree accuracy. [[Extra Trees]] pushes this logic further by also randomizing split thresholds, while [[Isolation Forest]] repurposes the same random-splitting architecture for unsupervised anomaly detection. Each tree in the forest is grown to maximum depth without pruning, a deliberate departure from the practice used with [[Decision Tree]] or [[CART]] in isolation; the ensemble mechanism substitutes for explicit regularization through pruning.

In the tree-based model lineage, Random Forest occupies the position of the first widely deployed ensemble method that required minimal hyperparameter tuning — the number of trees and *√p* feature subset size are robust defaults. This contrasts sharply with [[AdaBoost]] and [[Gradient Boosting]], which are sensitive to learning rate and tree depth and require careful early stopping to avoid overfitting. Random Forest's parallelizability (each tree is independent) also made it the dominant algorithm for tabular data through the mid-2010s, until [[XGBoost]]'s regularized sequential boosting consistently outperformed it on structured Kaggle benchmarks. Understanding the decorrelation logic of Random Forest remains a prerequisite for understanding why [[LightGBM]] and [[CatBoost]] invest so heavily in tree diversity mechanisms despite their fundamentally different boosting frameworks.