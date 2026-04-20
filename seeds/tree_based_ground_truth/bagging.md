---
name: Bagging
concept_type: Technique
what_it_is: Bagging (Bootstrap AGGregating) is an ensemble meta-algorithm that trains multiple copies of a base learner on independently drawn bootstrap samples of the training set and combines their predictions by averaging (regression) or majority vote (classification).
what_problem_it_solves: High-variance models such as fully grown decision trees overfit heavily to any single training set; bagging reduces this variance by averaging over models trained on overlapping but distinct resampled datasets, without increasing bias appreciably.
innovation_chain:
  - step: Decision Tree
    why: Fully grown decision trees are prototypical high-variance, low-bias models — exactly the profile that benefits most from variance reduction via ensemble averaging.
  - step: Bootstrap Resampling
    why: Efron's 1979 bootstrap established that drawing n samples with replacement from a dataset of size n produces a distribution estimate with quantifiable variance, providing the statistical foundation Bagging exploits.
  - step: Bagging
    why: Breiman's 1996 innovation was to apply bootstrap resampling not to estimate statistics but to deliberately diversify base learners, proving that instability in a learner is a necessary condition for bagging to yield gains.
limitations:
  - Bagging provides minimal benefit for stable (low-variance) learners such as shallow decision trees or linear models because the bootstrap samples produce nearly identical models.
  - Predictions from bagged ensembles are opaque — the averaging step destroys the interpretability of any single decision tree, negating a primary reason practitioners chose trees in the first place.
  - Bagging does not reduce bias; if the base learner is systematically wrong (high bias), aggregating its predictions yields an ensemble that is confidently wrong, and adding more trees cannot fix this.
introduced_year: 1996
domain_tags:
  - ensemble-learning
  - variance-reduction
  - tree-based-models
source_refs:
  - "Breiman 1996 - Bagging Predictors, Machine Learning 24(2)"
  - "Efron 1979 - Bootstrap Methods: Another Look at the Jackknife"
content_angles:
  - "The 'instability' prerequisite Breiman buried in his 1996 paper: why bagging a logistic regression does almost nothing, and how this insight directly explains Random Forest's design choices"
  - "Out-of-bag error as a free cross-validation estimate: how the ~36.8% of training points excluded per bootstrap sample give you model evaluation without a separate validation fold"
  - "Bagging was invented before GPUs made parallelism cheap — and it turned out to be embarrassingly parallel by design, a lucky property that makes it trivially scalable on modern hardware in ways boosting still cannot match"
relationships:
  - type: BUILDS_ON
    target: Decision Tree
    label: Bagging uses fully grown, unpruned decision trees as its canonical base learner precisely because their high variance makes them ideal candidates for variance-reduction through averaging.
  - type: ADDRESSES
    target: Decision Tree
    label: Bagging was designed specifically to counteract the overfitting and instability that characterize high-variance decision trees trained on finite datasets.
  - type: PREREQUISITE_OF
    target: Random Forest
    label: Random Forest extends Bagging by adding feature subsampling at each split, so understanding bootstrap aggregation is required before the additional decorrelation mechanism of Random Forest makes sense.
  - type: ALTERNATIVE_TO
    target: Boosting
    label: Both bagging and boosting are ensemble strategies for trees, but bagging reduces variance by parallel averaging over resampled datasets while boosting reduces bias by sequential reweighting of errors.
  - type: BASELINE_OF
    target: Random Forest
    label: Bagged decision trees (without feature subsampling) serve as the natural performance baseline against which Random Forest's extra decorrelation step is evaluated.
  - type: BELONGS_TO
    target: Boosting
    label: Bagging belongs to the broader ensemble learning family alongside boosting, representing the variance-reduction branch as opposed to boosting's bias-reduction branch.
  - type: INTRODUCES
    target: Extra Trees
    label: Bagging introduces the bootstrap-and-aggregate paradigm that Extra Trees inherits, though Extra Trees replaces bootstrap sampling with the full dataset and instead randomizes split thresholds.
---

## Bagging

Bagging, introduced by Leo Breiman in 1996, operationalizes a deceptively simple idea: if a learning algorithm is *unstable* — meaning small changes to the training data produce large changes in the learned model — then averaging multiple models trained on slightly different versions of the data will smooth out those fluctuations. Formally, given a training set of size n, Bagging draws B bootstrap samples each of size n with replacement, fits a base learner (typically a [[Decision Tree]]) to each, and combines predictions by averaging for regression or plurality vote for classification. Breiman proved that the expected prediction error of a bagged ensemble is bounded above by a term that decreases as the correlation among base learners decreases and their individual error decreases — a variance decomposition that makes explicit why instability is a *feature*, not a bug, of a good bagging candidate.

The statistical engine underneath Bagging is Efron's 1979 bootstrap: each bootstrap sample contains on average 1 − (1 − 1/n)^n ≈ 63.2% unique training points, leaving roughly 36.8% as out-of-bag (OOB) observations for any given tree. This OOB set can be used to estimate generalization error without a separate held-out fold, giving practitioners a nearly free model evaluation tool. This property also enables OOB feature importance — a technique later central to [[Random Forest]]. Bagging sits in direct contrast to [[Boosting]], which builds trees sequentially and reweights misclassified examples; bagging builds trees in parallel and weights them equally, making it embarrassingly parallelizable — a computational advantage that boosting methods like [[AdaBoost]] and [[Gradient Boosting]] cannot match by design.

Within the tree-based model lineage, Bagging is the direct intellectual parent of [[Random Forest]]: Breiman's 2001 Random Forest paper adds a single crucial modification — sampling a random subset of features at each candidate split — to decorrelate the trees further and reduce the ensemble's variance beyond what bootstrap resampling alone achieves. [[Extra Trees]] (Extremely Randomized Trees) pushes decorrelation even further by randomizing split thresholds and forgoing bootstrap sampling entirely, using the full dataset per tree. Understanding Bagging's variance-reduction mechanism is therefore prerequisite to understanding why these extensions work — and why methods like [[XGBoost]] and [[LightGBM]], which target bias rather than variance, require a fundamentally different theoretical lens.