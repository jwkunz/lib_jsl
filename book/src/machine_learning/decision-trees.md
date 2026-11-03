# Decision trees and random forests

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Classify by a sequence of yes/no questions about single features, learned from data; average many randomised trees for accuracy |
| **Devised by** | James Morgan and John Sonquist (1963), AID; Leo Breiman, Jerome Friedman, Richard Olshen and Charles Stone (1984), CART; Ross Quinlan (1986, 1993), ID3 and C4.5; Tin Kam Ho (1995), random subspaces; Breiman (1996), bagging, and (2001), random forests |
| **Accuracy** | Single trees are high-variance; forests are among the most accurate off-the-shelf classifiers for tabular data |
| **Cost** | Growing a tree $O(pn\log^2n)$ (a sort per feature per node); a forest of $T$ trees $T$ times that on $\sqrt p$ features per split |
| **API** | [`DecisionTree`](../../api/lib_jsl_machine_learning/supervised/classification/random_forest/struct.DecisionTree.html) (`new(max_depth)`, `with_min_samples_split`, `predict_proba`, `depth`), [`RandomForest`](../../api/lib_jsl_machine_learning/supervised/classification/random_forest/struct.RandomForest.html) (`new(trees, max_depth, seed)`, `oob_accuracy`, `feature_importances`) |

</div>

## Overview

A **decision tree** partitions the feature space into boxes by repeated splits on one feature at a time ("is $x_3 \le 2.7$?") and predicts
the majority class in each box. Trees are easy to read, need no feature scaling, and capture interactions, but a deep tree
memorises its training data and a shallow one is crude: small changes in the data change the tree completely. A **random
forest** grows hundreds of deep trees on bootstrap resamples of the data, each choosing among a random subset of features
at every split, and averages them. The averaging removes most of the variance while keeping the flexibility.

## History

James Morgan and John Sonquist (1963) introduced tree-structured "automatic interaction detection" for survey data. Leo
Breiman, Jerome Friedman, Richard Olshen and Charles Stone's book *Classification and Regression Trees* (1984) set out CART:
binary splits chosen by impurity, then pruning. Ross Quinlan's ID3 (1986) and C4.5 (1993) did the same with information gain
in machine learning. Breiman's **bagging** (1996) averaged trees grown on bootstrap samples; Tin Kam Ho (1995) and Yali Amit
and Donald Geman (1997) randomised the features; Breiman (2001) combined the two as random forests, with out-of-bag error
estimates and variable importance. Forests and gradient-boosted trees remain the strongest general methods for
tabular data.

## The idea

### Growing a tree by Gini impurity

At a node with class proportions $p_c$, the **Gini impurity** $G = 1 - \sum_cp_c^2$ is the probability that two points drawn at random have different labels:
zero for a pure node. For each feature, sort the node's points by it and consider every threshold between consecutive distinct
values; choose the split that most reduces the size-weighted impurity of the two children. Recurse until the node is pure, too
small, or at the depth limit; a leaf stores the class proportions, which serve as probabilities.

| | | |
|:---:|:---:|:---:|
| ![Depth 2](../generated/machine_learning/tree_depth2.svg) | ![Deep tree](../generated/machine_learning/tree_deep.svg) | ![Forest](../generated/machine_learning/forest.svg) |

A tree of depth 2 draws three axis-parallel cuts; a fully grown tree carves out a box for every stray point, including thin
slivers; a forest of 200 trees averages them into a boundary that follows the moons.

### Bagging and random features

Averaging $T$ predictors with variance $\sigma^2$ and pairwise correlation $\rho$ gives variance $\rho\sigma^2 + (1 - \rho)\sigma^2/T$: more trees remove the second term, but
the first remains. Bootstrap resampling makes the trees differ; restricting each split to a random subset of $m$ features
(default $\sqrt p$) decorrelates them further, because a dominant feature can no longer head every tree.

### Out-of-bag error and feature importance

Each bootstrap sample omits about 37% of the points. Scoring each point with only the trees that did not see it gives the
**out-of-bag** accuracy, an estimate of test accuracy at no extra cost:

![Accuracy against the number of trees](../generated/machine_learning/forest_trees.svg)

The forest beats the single deep tree by eight points of accuracy with a few dozen trees, and more trees never hurt. The
out-of-bag estimate (available once every point has been left out at least once) runs a little below the test accuracy here;
with 300 training points its standard error is about 0.02.

The library's **feature importance** is the share of all splits that use each feature. On four informative and six pure-noise
features:

{{#include ../generated/machine_learning/forest_importances.md}}

The informative features rank first, but the noise features still receive nearly half the splits: deep trees keep splitting on
noise once the signal is exhausted. Impurity-weighted or permutation importances separate them more clearly.

## Algorithm

1. **Tree**: at each node, if pure, too small or at the depth limit, make a leaf. Otherwise, for each candidate feature, sort and scan
   the thresholds, tracking the class counts on the left; split at the best; recurse.
2. **Forest**: for each tree, draw a bootstrap sample and grow a tree with $m$ random candidate features per split; record out-of-bag
   votes. Predict by averaging the trees' class proportions.

## Implementation

```rust,ignore
{{#include ../../../crates/machine_learning/src/supervised/classification/random_forest.rs:cart_split}}
```

```rust,ignore
{{#include ../../../crates/machine_learning/src/supervised/classification/random_forest.rs:forest}}
```

- Splits with zero impurity decrease are allowed, so that problems such as XOR, where no single split helps, can still be
  separated; the depth and node-size limits stop the growth.
- There is no pruning: limit the depth or use a forest.

## In practice

```rust,ignore
{{#include ../../../crates/machine_learning/examples/ml_decision_trees.rs}}
```

```text
tree, depth limit  1 (grown to 1): training 0.607, test 0.614
tree, depth limit  2 (grown to 2): training 0.857, test 0.822
tree, depth limit  4 (grown to 4): training 0.913, test 0.846
tree, depth limit  8 (grown to 8): training 0.997, test 0.827
tree, depth limit 50 (grown to 10): training 1.000, test 0.827
forest of 200 trees: test 0.893, out-of-bag estimate 0.900
share of splits per feature: [0.299, 0.251, 0.246, 0.204]
```

**Strengths**

- Trees: interpretable, no scaling, handle interactions and mixed feature scales.
- Forests: accurate with little tuning, robust to irrelevant features, with a free out-of-bag estimate.

**Weaknesses**

- Single trees are unstable and overfit; boundaries are axis-parallel staircases.
- Forests lose interpretability and are slower to train and predict; split-count importances are biased.
- Classification only; no regression trees in this module (see gradient boosting in the ensembles module).

## Exercises

1. Show that the Gini impurity is the expected misclassification rate of a rule that labels randomly with the node's proportions.
2. Compute the variance of the average of $T$ equicorrelated predictors, and its limit as $T \to \infty$.
3. Show that the probability that a given point is left out of a bootstrap sample of size $n$ tends to $e^{-1} \approx 0.368$.
4. Construct the XOR problem and show that no single axis-parallel split reduces the Gini impurity, but two levels separate it.
5. Implement permutation importance (the drop in out-of-bag accuracy when a feature's values are shuffled) and compare with split counts.

## Further reading

- L. Breiman, J. H. Friedman, R. A. Olshen and C. J. Stone, *Classification and Regression Trees*, Wadsworth, 1984.
- L. Breiman, "Random forests", *Machine Learning* 45 (2001) 5–32.
- L. Breiman, "Bagging predictors", *Machine Learning* 24 (1996) 123–140.
- G. Louppe, *Understanding Random Forests: From Theory to Practice*, PhD thesis, University of Liège, 2014.
