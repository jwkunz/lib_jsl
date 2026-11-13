# Foundations: models, data, preprocessing and metrics

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Provide the common interface of the learning algorithms, reproducible data to try them on, feature scaling and splitting, and measures of how well a model predicts |
| **Devised by** | Mervyn Stone (1974) and Seymour Geisser (1975), held-out validation; Karl Pearson (1904), contingency tables; Cornelis van Rijsbergen (1979), the F-measure; Lawrence Hubert and Phipps Arabie (1985), the adjusted Rand index; Peter Rousseeuw (1987), silhouettes |
| **Accuracy** | Exact |
| **Cost** | Linear in the size of the data |
| **API** | [`common`](../../api/lib_jsl_machine_learning/common/index.html) (`Regressor`, `Classifier`, `Clusterer`, `Transformer`, `MlRng`), [`datasets`](../../api/lib_jsl_machine_learning/datasets/index.html), [`preprocessing`](../../api/lib_jsl_machine_learning/preprocessing/index.html) (`StandardScaler`, `MinMaxScaler`, `train_test_split`, `polynomial_features`), [`metrics`](../../api/lib_jsl_machine_learning/metrics/index.html) |

</div>

## Overview

Every model in the crate follows the same pattern: create it with its settings, `fit` it to training data, and `predict` on
new data. Four traits express this: a `Regressor` maps rows to numbers, a `Classifier` maps rows to labels $0, \ldots, K - 1$, a
`Clusterer` groups rows without labels, and a `Transformer` learns a mapping of the rows (a scaling, a projection). Data
are `ndarray` matrices with one sample per row. Around the models sit the tools of a sound workflow: synthetic data sets,
scaling, splitting into training and test sets, and metrics.

## History

The idea of judging a prediction method on data held out from fitting goes back at least to the psychometricians of the
1930s; Mervyn Stone (1974) and Seymour Geisser (1975) formalised cross-validation. The confusion matrix descends from Karl
Pearson's contingency tables (1904); precision and recall come from information retrieval, where Cornelis van Rijsbergen
(1979) combined them into the F-measure. For clusterings, Lawrence Hubert and Phipps Arabie (1985) corrected William Rand's
agreement index (1971) for chance, and Peter Rousseeuw (1987) introduced silhouettes. Synthetic benchmarks such as the
"moons" and "circles" were popularised by scikit-learn (2011).

## The idea

### The model traits

```rust,ignore
{{#include ../../../crates/machine_learning/src/common.rs:traits}}
```

Because the traits are object-safe, models can be stored as `Box<dyn Classifier>` and compared in a loop, as the
[comparison table](index.md#comparing-classifiers) does.

### Training and test data

A model's error on its own training data is optimistic, often wildly so: a 1-nearest-neighbour classifier or a fully grown
tree has zero training error. `train_test_split` shuffles the sample indices and holds out a fraction for testing, and
`select_rows` and `select_labels` gather the pieces. Any step that learns from data, including scaling, must be fitted on the
training part only, or information leaks from the test set. Cross-validation, which averages over several splits, is covered with model selection in a
later chapter.

### Scaling

Penalties, distances and kernels treat all features alike, so a feature measured in millimetres dominates one measured in
metres. `StandardScaler` subtracts each column's mean and divides by its standard deviation (constant columns are only
centred); `MinMaxScaler` maps each column onto $[0, 1]$. Trees and naive Bayes do not need scaling; nearly everything else does.
`polynomial_features` expands one column into $x, x^2, \ldots, x^d$, turning linear regression into polynomial regression.

### Synthetic data sets

Every generator takes a seed: `make_blobs` (Gaussian clusters), `make_moons` and `make_circles` (two classes with curved
boundaries), `make_classification` (three overlapping, correlated classes in four dimensions), `make_regression` (a linear
model with noise, returning the true weights), `make_sine` and `make_swiss_roll` (a manifold for
dimensionality reduction).

### Metrics

For regression, the **mean squared error**, its root, the **mean absolute error** and the **coefficient of determination**
$R^2 = 1 - \sum(y_i - \hat y_i)^2/\sum(y_i - \bar y)^2$, the fraction of variance explained (negative for a model worse than the mean). For
classification, **accuracy** is the fraction correct; the **confusion matrix** $m_{tp}$ counts true class $t$ predicted as $p$; and per class,
**precision** (of the points predicted as $c$, the fraction that are $c$), **recall** (of the points in $c$, the fraction found) and their
harmonic mean **F1**. For a random forest on the three-class problem:

<p class="caption" id="tab-11-4"><strong>Table 11.4.</strong> Columns: true class, predicted 0, predicted 1, predicted 2, and more.</p>

{{#include ../generated/machine_learning/confusion.md}}

Class 0, in the middle, is confused with both others. For clusterings, the **silhouette** compares each point's mean
distance to its own cluster with that to the nearest other cluster, and the **adjusted Rand index** compares two labellings,
scoring 1 for identical partitions and about 0 for chance agreement.

## Algorithm

**`train_test_split`**, with a test fraction $f$ and a seed:

1. list the indices $0, \ldots, n - 1$ and shuffle them by the Fisher-Yates method, driven by a
   generator seeded from the seed: for $i$ from $n - 1$ down to 1, swap position $i$ with a
   position chosen uniformly from $0, \ldots, i$;
2. take the last $\operatorname{round}(fn)$ shuffled indices as the test set (at least one, and
   leaving at least one for training), and the rest as the training set;
3. return the two lists of indices, from which `select_rows` and `select_labels` gather the
   data.

**`StandardScaler`**:

1. *fit*: compute the mean and the population standard deviation of each column of the
   training data;
2. *transform*: from each value subtract its column's mean and divide by its column's standard
   deviation; a column with zero standard deviation is only centred.

The same fitted means and standard deviations are applied to the test data. Fitting the scaler
again on the test data would use information from it.

**A classification metric from a confusion matrix**: count $m_{tp}$, the number of points of
true class $t$ predicted as $p$. The accuracy is the sum of the diagonal divided by $n$. For
class $c$, the precision is $m_{cc}$ divided by the sum of column $c$, the recall is $m_{cc}$
divided by the sum of row $c$, and F1 is their harmonic mean.

## Implementation

- `MlRng` wraps the xoshiro256++ generator of the [random numbers](../random/generators.md) chapter, with uniform, normal,
  shuffle and sampling helpers; every stochastic algorithm takes a seed.
- Models return `Misconfiguration` errors when used before fitting and `IncompatibleArraySizes` for mismatched inputs.

## In practice

```rust,ignore
{{#include ../../../crates/machine_learning/examples/ml_foundations.rs}}
```

```text
test accuracy 0.904 on 135 points
class 0: [34, 6, 1]  precision 0.85 recall 0.83 F1 0.84
class 1: [4, 41, 0]  precision 0.87 recall 0.91 F1 0.89
class 2: [2, 0, 47]  precision 0.98 recall 0.96 F1 0.97
```

**Strengths**

- One interface for all models; reproducible data and splits.

**Weaknesses**

- Dense `f64` matrices only: no sparse data, categorical encoding or missing-value handling.
- No probability-based metrics (log-loss, ROC curves); compute them from `predict_proba`.

## Exercises

1. Show that fitting a `StandardScaler` on all the data before splitting leaks information, by constructing an example where
   it changes the test accuracy.
2. For a binary problem with 1% positives, what accuracy does always predicting "negative" achieve? What are its precision and recall?
3. Show that F1 is the harmonic mean of precision and recall and lies between their minimum and their arithmetic mean.
4. Compute the adjusted Rand index of two random labellings of 1000 points into 3 groups, over many trials.

<details>
<summary>Hints and answers</summary>

1. Construct data in which the feature mean is very different in the test portion (for instance a drifting feature). Scaling with statistics from all the data moves the test points toward the training distribution, changing the test accuracy from what an unseen test set would give.
2. Accuracy $99\%$, recall $0$, and precision is $0/0$ (undefined; conventionally 0), so F1 is also 0: accuracy is misleading on imbalanced data.
3. $F_1 = 2PR/(P + R)$ is the harmonic mean. For $P \le R$, $\min \le \text{harmonic mean} \le \text{geometric mean} \le \text{arithmetic mean}$.
4. The adjusted Rand index of independent random labellings has mean $0$ and standard deviation of order $10^{-3}$ for 1000 points; values near zero (and sometimes slightly negative) are what the adjustment is for.

</details>

## Further reading

- T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 (free online).
- G. James, D. Witten, T. Hastie and R. Tibshirani, *An Introduction to Statistical Learning*, 2nd ed., Springer, 2021.
- F. Pedregosa et al., "Scikit-learn: machine learning in Python", *Journal of Machine Learning Research* 12 (2011) 2825–2830.
