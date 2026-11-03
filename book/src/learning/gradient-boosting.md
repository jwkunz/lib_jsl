# Gradient boosting

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Build an accurate predictor as a sum of many small regression trees, each fitted to the errors of the sum so far |
| **Devised by** | Robert Schapire (1990) and Yoav Freund and Schapire (1997), boosting and AdaBoost; Leo Breiman (1997), boosting as gradient descent; Jerome Friedman (2001), gradient boosting machines, and (2002), stochastic gradient boosting; Tianqi Chen and Carlos Guestrin (2016), XGBoost |
| **Accuracy** | Training loss decreases at every stage; test error is controlled by the learning rate, tree depth, subsampling and the number of stages |
| **Cost** | Each tree $O(dpn\log n)$ for depth $d$ (a sort per feature per level); $K$ trees per stage for $K$ classes |
| **API** | [`GradientBoostingRegressor`](../../api/lib_jsl_machine_learning/ensembles/struct.GradientBoostingRegressor.html) (`feature_importances`, `training_loss`), [`GradientBoostingClassifier`](../../api/lib_jsl_machine_learning/ensembles/struct.GradientBoostingClassifier.html) (`predict_proba`, `training_loss`), [`BoostingParameters`](../../api/lib_jsl_machine_learning/ensembles/struct.BoostingParameters.html) (`stages`, `learning_rate`, `max_depth`, `min_samples_leaf`, `subsample`, `seed`), [`RegressionTree`](../../api/lib_jsl_machine_learning/ensembles/struct.RegressionTree.html) |

</div>

## Overview

A [random forest](../machine_learning/decision-trees.md) averages deep trees grown independently. **Boosting** does the opposite: it grows *shallow* trees one after another, each
correcting what the previous ones got wrong. **Gradient boosting** makes this precise. At each stage it computes the gradient of the
loss with respect to the current predictions, fits a small regression tree to the negative gradient, and adds a fraction of that tree to
the model. It is gradient descent in the space of functions, with trees as steps. For squared error the negative gradient is
just the residual, so each tree fits what is left to explain. Gradient-boosted trees, in implementations such as XGBoost and LightGBM, are
the most successful general method for tabular data.

## History

Michael Kearns and Leslie Valiant (1988) asked whether "weak" learners, slightly better than guessing, could be combined into a
strong one. Robert Schapire (1990) proved they could, and Yoav Freund and Schapire's AdaBoost (1997) made it practical by
reweighting misclassified examples. Leo Breiman (1997) and Jerome Friedman, Trevor Hastie and Robert Tibshirani (2000) showed
that AdaBoost minimises an exponential loss stage by stage. Friedman (2001) generalised this to any differentiable loss as
gradient boosting, with shrinkage and Newton steps in the leaves, and (2002) added row subsampling. Tianqi Chen and Carlos
Guestrin's XGBoost (2016), Microsoft's LightGBM (2017) and Yandex's CatBoost (2017) added second-order splits, histograms and
careful engineering, and dominate machine-learning competitions on tabular data.

## The idea

### Functional gradient descent

The model is an additive expansion

$$
F_M(\mathbf x) = F_0 + \nu\sum_{m=1}^Mh_m(\mathbf x),
$$

where $F_0$ is a constant (the mean, or the log class priors), each $h_m$ is a small regression tree, and $\nu \in (0, 1]$ is the **learning rate**. At stage $m$, compute
the **pseudo-residuals** $r_i = -\partial L(y_i, F)/\partial F$ at $F = F_{m-1}(\mathbf x_i)$, fit a tree $h_m$ to them by least squares, and add $\nu h_m$. For squared error $L = \frac12(y - F)^2$ the
pseudo-residual is the ordinary residual $y_i - F_{m-1}(\mathbf x_i)$:

![Boosting a sine](../generated/learning/gb_stages.svg)

One depth-2 tree, scaled by $\nu = 0.1$, barely moves from the mean. Ten stages capture the shape coarsely, and a hundred follow the
sine as a fine staircase.

### Newton steps in the leaves

The tree structure comes from a least-squares fit to the gradient, but the value in each leaf can do better than the mean gradient.
A Newton step for the loss restricted to the leaf's points $R$ is

$$
\gamma = \frac{\sum_{i \in R}g_i}{\sum_{i \in R}h_i},
$$

with $g_i$ the negative gradient and $h_i$ the second derivative. For squared error $h_i = 1$, giving the mean residual. For classification with $K$ classes, the
model keeps one score $F_k$ per class with softmax probabilities $p_k$, and minimises the cross-entropy (multinomial deviance). Each stage fits one tree per
class to $y_{ik} - p_{ik}$, with leaf values $\frac{K - 1}{K}\sum(y - p)/\sum p(1 - p)$ (Friedman's factor $\frac{K - 1}{K}$ allows for the softmax's redundant degree of freedom).

![Gradient boosting on moons](../generated/learning/gb_moons.svg)

The boundary is built from axis-parallel pieces, like a tree's, but a hundred small trees give it far more resolution than one
deep tree.

### Shrinkage and the number of stages

The learning rate and the number of stages trade off. A small $\nu$ needs more stages but generalises better, a regularisation
effect Friedman found empirically, related to the lasso's path. On a nonlinear regression problem with 300 training points:

![Test error against stages](../generated/learning/gb_shrinkage.svg)

{{#include ../generated/learning/gb_shrinkage.md}}

With $\nu = 1$ the test error bottoms out after 10 stages, then rises as the trees fit noise. With $\nu = 0.1$ it keeps improving to 1000 stages and
ends far lower. With $\nu = 0.03$, 1000 stages are not yet enough. In practice one sets a small $\nu$ and chooses the number of stages by
[cross-validation](model-selection.md) or a held-out set.

### Stochastic gradient boosting

Fitting each tree to a random fraction of the rows (`subsample`) both speeds training up and, by decorrelating the trees, reduces
overfitting:

{{#include ../generated/learning/gb_subsample.md}}

Half the rows per tree gives the lowest error here. With subsampling the training loss is no longer guaranteed to fall at every
stage.

## Algorithm

1. $F_0$: the mean target (regression) or smoothed log class priors (classification).
2. For $m = 1, \ldots, M$: draw the subsample; compute pseudo-residuals and second derivatives; grow a regression tree of depth at most $d$ on
   the subsample, choosing splits that maximise the reduction in squared error of the residuals; set each leaf to its Newton step;
   add $\nu$ times the tree to every prediction.
3. Feature importances: the total squared-error reduction from splits on each feature, normalised to sum to one.

## Implementation

```rust,ignore
{{#include ../../../crates/machine_learning/src/ensembles.rs:boost_regression}}
```

```rust,ignore
{{#include ../../../crates/machine_learning/src/ensembles.rs:boost_classification}}
```

- Split search sorts each feature at each node, $O(pn\log n)$ per level. Histogram-based implementations bin the features once and are much faster
  on large data.
- Classification leaf values are clamped to ±20 to avoid overflow when a leaf is pure.
- Predictions check the number of features. (Before this chapter, a row with too few features panicked.)

## In practice

```rust,ignore
{{#include ../../../crates/machine_learning/examples/ml_gradient_boosting.rs}}
```

```text
regression: test R² 0.855
feature importances [0.087, 0.108, 0.323, 0.210, 0.012, 0.046, 0.012, 0.202]
classification: test accuracy 0.889, training deviance 0.973 → 0.050
```

Feature 2, which enters both linearly (coefficient −2.41) and quadratically, is the most important. Features 3 and 7 have the
largest linear coefficients (−2.65 and −2.59); features 0 and 1 matter partly through their product, which trees can only
approximate in pieces; features 4 and 6, with coefficients 0.16 and −0.24, are almost unused. The classifier drives its
training deviance close to zero; the gap to its test accuracy is the usual warning that training loss says little about
generalisation.

**Strengths**

- State-of-the-art accuracy on tabular data; handles mixed feature scales, interactions and missing structure with little preprocessing.
- Any differentiable loss; feature importances come for free.

**Weaknesses**

- Sequential: stages cannot be trained in parallel, and many settings need tuning (rate, stages, depth, subsampling).
- Can overfit with too many stages or too large a rate; piecewise-constant predictions do not extrapolate.

## Exercises

1. Show that the pseudo-residual of the squared loss is the residual, and that of the absolute loss $|y - F|$ is the sign of the residual.
2. Derive the Newton leaf value for the binary logistic loss.
3. Show that boosting depth-1 trees (stumps) gives an additive model $\sum_jf_j(x_j)$ with no interactions.
4. Explain why the training loss of full-sample boosting with Newton leaves for squared error can never increase.
5. Implement early stopping: hold out 20% of the training data and stop when its error has not improved for 20 stages.

## Further reading

- J. H. Friedman, "Greedy function approximation: a gradient boosting machine", *Annals of Statistics* 29 (2001) 1189–1232.
- J. H. Friedman, "Stochastic gradient boosting", *Computational Statistics and Data Analysis* 38 (2002) 367–378.
- T. Chen and C. Guestrin, "XGBoost: a scalable tree boosting system", *Proceedings of the 22nd ACM SIGKDD Conference* (2016) 785–794.
- T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009, chapter 10.
