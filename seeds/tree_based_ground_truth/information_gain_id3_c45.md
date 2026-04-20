---
name: Information Gain (ID3/C4.5)
concept_type: Mechanism
what_it_is: Information Gain is a split-selection criterion for decision trees that quantifies the reduction in Shannon entropy achieved by partitioning a dataset on a given feature, forming the core of the ID3 and C4.5 tree-learning algorithms developed by Ross Quinlan.
what_problem_it_solves: Given a set of candidate features at each tree node, it answers which feature to split on by measuring how much knowledge about the target class label is gained — replacing arbitrary heuristics with a principled, information-theoretic objective.
innovation_chain:
  - step: Shannon Entropy (Information Theory, 1948)
    why: Provides the mathematical foundation — entropy H(S) = -Σ p_i log₂(p_i) — that Information Gain uses to measure class impurity before and after a split.
  - step: ID3 Algorithm (Quinlan, 1979/1986)
    why: First operationalized Shannon entropy as a greedy, recursive split criterion to grow decision trees for classification tasks.
  - step: Information Gain (ID3/C4.5)
    why: C4.5 extended ID3 by normalizing raw Information Gain with split entropy (Gain Ratio) to penalize high-cardinality features and adding support for continuous attributes and missing values.
limitations:
  - Raw Information Gain is biased toward features with many distinct values (e.g., a unique ID column achieves maximum gain), necessitating the Gain Ratio correction in C4.5.
  - The greedy, top-down induction provides no global optimality guarantee — a locally optimal split at depth k may prevent globally better splits at depth k+1.
  - Entropy computation requires log evaluations over all class-label frequencies, making it slower at each node than Gini impurity (used in CART), especially for multiclass problems with many labels.
introduced_year: 1986
domain_tags:
  - decision-trees
  - classification
  - information-theory
source_refs:
  - "Quinlan 1986 - Induction of Decision Trees (Machine Learning, Vol. 1)"
  - "Quinlan 1993 - C4.5: Programs for Machine Learning"
  - "Shannon 1948 - A Mathematical Theory of Communication"
content_angles:
  - "The bias that almost broke decision trees: how a unique customer-ID column can score perfect Information Gain — and why C4.5's Gain Ratio was the fix that saved the algorithm"
  - "Entropy vs. Gini: a side-by-side numerical walkthrough showing why both criteria produce nearly identical splits 95% of the time, and the exact edge cases where they diverge"
  - "Ross Quinlan wrote ID3 in the late 1970s without publishing it for years — tracing the quiet lineage from a RAND Corporation report to the paper that launched a thousand decision-tree libraries"
relationships:
  - type: INTRODUCES
    target: Decision Tree
    label: Information Gain is the defining split criterion through which ID3/C4.5 introduced the first widely-used, entropy-driven decision tree learning algorithm.
  - type: ALTERNATIVE_TO
    target: CART
    label: Information Gain (used in ID3/C4.5) and Gini impurity (used in CART) are competing split criteria for classification trees that almost always agree but differ in computational cost and multiclass behavior.
  - type: PREREQUISITE_OF
    target: Random Forest
    label: Understanding how Information Gain or Gini selects splits in a single tree is necessary before grasping how Random Forest decorrelates an ensemble of such trees via feature subsampling.
  - type: PREREQUISITE_OF
    target: Boosting
    label: The mechanics of how a single decision tree greedily selects splits via an impurity criterion underpin the weak learners that boosting algorithms combine sequentially.
  - type: BELONGS_TO
    target: Decision Tree
    label: Information Gain is a split-quality mechanism that belongs to the family of greedy induction algorithms for decision trees.
---

## Information Gain (ID3/C4.5)

At its core, Information Gain measures the expected reduction in [[Shannon Entropy]] when a dataset S is partitioned by feature A: IG(S, A) = H(S) − Σ_v (|S_v|/|S|) · H(S_v), where the sum runs over all values v that A takes and S_v is the subset of S with A = v. [[ID3]] (Iterative Dichotomiser 3), Quinlan's 1979/1986 algorithm, applies this formula greedily at every node, selecting the feature that maximally reduces entropy and recursing on the resulting partitions until leaves are pure or no features remain. The elegance of grounding tree induction in Claude Shannon's 1948 communication theory — treating class uncertainty exactly as channel uncertainty — was a conceptual leap that distinguished ID3 from contemporary rule-induction systems operating on ad-hoc scoring functions.

C4.5, Quinlan's 1993 successor, patched the primary failure mode of raw Information Gain: its systematic bias toward attributes with high cardinality. A feature with n unique values can create n singleton leaves, each trivially pure, earning maximum gain without any generalizable structure. C4.5 addresses this with the **Gain Ratio**: GR(S, A) = IG(S, A) / SplitInfo(S, A), where SplitInfo = −Σ_v (|S_v|/|S|) log₂(|S_v|/|S|) penalizes splits that fragment the data into many small partitions. C4.5 also introduced binarization of continuous features via threshold search, probabilistic handling of missing values, and post-pruning via pessimistic error estimation — transforming ID3 from a research prototype into an industrial-strength learner. The resulting classifier, J48 in the Weka toolkit, remained a practical benchmark well into the 2010s.

[[CART]] (Classification and Regression Trees), developed independently by Breiman et al. (1984), chose Gini impurity over entropy as its splitting criterion, yielding binary splits rather than multi-way ones and enabling regression as well as classification. The two lineages — Quinlan's entropy-based ID3/C4.5 and Breiman's Gini-based [[CART]] — represent the two dominant philosophical approaches to greedy tree induction, and their split criteria are the prerequisite concepts underlying every ensemble method that followed: [[Bagging]], [[Random Forest]], [[Gradient Boosting]], and ultimately [[XGBoost]] and [[LightGBM]] all inherit the basic question "how do we score a candidate split?" first answered rigorously by Information Gain.