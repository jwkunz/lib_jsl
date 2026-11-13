# Machine learning: regression and classification

**Supervised learning** fits a function from examples: given feature vectors $\mathbf x_i$ and targets $y_i$, learn $f$ with $f(\mathbf x) \approx y$
for new $\mathbf x$. When $y$ is a number the problem is **regression**; when it is a category it is **classification**. The aim is not to fit the
training data, which a flexible enough model can always do, but to **generalise**: to predict well on data the model has
not seen. Every method in this chapter has a knob that trades flexibility against stability (a penalty, a number of
neighbours, a depth, a kernel width), and choosing it well matters more than the choice of method.

## Background: bias, variance and generalisation

The first central question below, and many entries in this and later chapters, rest on one decomposition of prediction error.
Suppose the data obey $y = f(\mathbf x) + \epsilon$, where $f$ is the true function and the noise $\epsilon$ has mean zero and
variance $\sigma^2$, independent of everything else. A method turns a random training set into a fitted function $\hat f$,
so $\hat f(\mathbf x)$ is itself a random quantity: a different training sample would have produced a different fit. Fix a
point $\mathbf x$ and take expectations over both the training sets and the noise in a new observation $y$. Because $\epsilon$ is independent of
$\hat f$, the cross term vanishes:

$$
\mathbb E\big[(y - \hat f)^2\big] = \mathbb E\big[(f - \hat f)^2\big] + \sigma^2 .
$$

Now add and subtract the average fit $\bar f = \mathbb E[\hat f(\mathbf x)]$ inside the first term. Again the cross term has mean zero, and
what is left is

$$
\mathbb E\big[(y - \hat f)^2\big] = \underbrace{\big(f(\mathbf x) - \bar f\big)^2}_{\text{bias}^2}
+ \underbrace{\mathbb E\big[(\hat f - \bar f)^2\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible noise}} .
$$

**Bias** is the systematic error: how far the *average* fit is from the truth, caused by a model too rigid to represent $f$.
**Variance** is the sensitivity to the particular sample: how much the fit moves from one training set to the next. The
noise $\sigma^2$ cannot be reduced by any method. A flexible model has low bias and high variance, a rigid one the reverse, and
the best predictor balances the two.

**A case that can be solved exactly.** Estimate an unknown mean $\mu$ from $n$ independent observations of variance
$\sigma^2$, using the shrunken average $\hat\mu = c\,\bar x$ for a constant $c \in [0, 1]$. Its expectation is $c\mu$ and its
variance $c^2\sigma^2/n$, so

$$
\text{error}^2 = \underbrace{(1 - c)^2\mu^2}_{\text{bias}^2} + \underbrace{c^2\sigma^2/n}_{\text{variance}} .
$$

With $\mu = 1$, $\sigma = 2$ and $n = 4$ the plain average ($c = 1$) has no bias and variance $1$, for total $1$. Shrinking to
$c = \tfrac12$ introduces a bias$^2$ of $\tfrac14$ but cuts the variance to $\tfrac14$, for a total of $\tfrac12$: *adding
bias halved the error*. Minimising over $c$ gives $c = \mu^2/(\mu^2 + \sigma^2/n)$, which here is exactly $\tfrac12$. The
optimum depends on the unknown $\mu$, which is why the knob has to be tuned on held-out data rather than computed.
This is the mechanism behind ridge regression, whose penalty shrinks coefficients towards zero.

**A case that shows the trade-off along a knob.** The $k$-nearest-neighbour regressor averages the $k$ nearest targets. The noise in
the average has variance $\sigma^2/k$, so variance falls as $k$ grows. But the neighbours used lie further and further from
$\mathbf x$, so the average of $f$ over them drifts away from $f(\mathbf x)$ and the bias grows. The best $k$ lies between $1$ (all variance) and
$n$ (all bias).

The same trade-off appears whenever a method has a smoothing knob:

| Method | Turning the knob towards... | ...raises bias | ...lowers variance |
|---|---|---|---|
| Ridge, lasso | a larger penalty $\lambda$ | yes | yes |
| $k$-nearest neighbours | larger $k$ | yes | yes |
| Decision tree | shallower depth, more pruning | yes | yes |
| Boosting, neural network | fewer rounds or earlier stopping | yes | yes |
| [Kernel density estimate](../statistics/kernel-density.md) | a larger bandwidth $h$ | yes | yes |
| [Welch's spectrum estimate](../spectral/power-spectra.md) | shorter segments (more averaged) | yes (coarser resolution) | yes |

Training error always falls as flexibility grows, so it cannot locate the optimum; error on data the model has not seen
can. That is the job of [cross-validation](../learning/model-selection.md).

## The central questions

- **Bias or variance?** (See the background section above.) A rigid model misses real structure (bias); a flexible one fits the noise (variance). Penalties,
  averaging and early stopping control the trade-off, and held-out data measure it.
- **What shape can the decision boundary take?** Linear models (least squares, logistic regression, linear SVMs) draw
  hyperplanes; kernels, neighbours and trees draw curved or piecewise boundaries.
- **Generative or discriminative?** Naive Bayes models how each class generates its features; logistic regression models
  the class probabilities directly. The first learns faster from little data, the second is more accurate with plenty.
- **Are the features on comparable scales?** Penalties, distances and kernels all depend on the units of the features,
  so standardise them first.

## Testing the library

Each fitted model was checked against the optimality conditions of its own objective. The gradient of least squares, ridge
and logistic regression and the subgradient conditions of the lasso and elastic net vanish to the solver tolerance:

<p class="caption" id="tab-11-1"><strong>Table 11.1.</strong> Optimality conditions of the fitted models.</p>

{{#include ../generated/machine_learning/optimality.md}}

### Against SciPy

Each model's objective was reproduced with SciPy 1.17.1's linear algebra, optimisers and distributions (`book/generator/data/make_machine_learning_refs.py`), so the references are the same mathematical solutions computed independently:

<p class="caption" id="tab-11-2"><strong>Table 11.2.</strong> Supervised models against SciPy.</p>

{{#include ../generated/machine_learning/scipy_reference.md}}

The closed-form fits agree to rounding. The lasso, elastic net and logistic regression are solved iteratively on both sides, and agree to the stopping tolerances of the optimisers.

## Comparing classifiers

<p class="caption" id="tab-11-3"><strong>Table 11.3.</strong> Test accuracy of the classifiers on three data sets.</p>

{{#include ../generated/machine_learning/comparison.md}}

No method wins everywhere. Linear logistic regression fails on the concentric circles, where Gaussian naive Bayes succeeds
only because the two rings have different spreads; neighbours, forests and kernel machines adapt to curved boundaries; a
single tree is the least accurate on the smooth Gaussian problem.

## In the library

The crate is `lib_jsl_machine_learning` ([API reference](../../api/lib_jsl_machine_learning/index.html)). Feature matrices are
`ndarray` arrays with one sample per row; models implement `Regressor` or `Classifier`.

| You need | Use | Entry |
|---|---|---|
| Train/test splits, scaling, metrics, synthetic data | `preprocessing`, `metrics`, `datasets` | [Foundations](foundations.md) |
| A linear fit, possibly penalised or weighted | `LinearRegression`, `Ridge`, `ElasticNet`, `GeneralLeastSquares` | [Linear regression](linear-regression.md) |
| Class probabilities from a linear model | `LogisticRegression` | [Logistic regression](logistic-regression.md) |
| A fast probabilistic classifier, text classification | `GaussianNaiveBayes`, `MultinomialNaiveBayes` | [Naive Bayes](naive-bayes.md) |
| A classifier with no training | `KNearestNeighbors` | [Nearest neighbours](nearest-neighbours.md) |
| Interpretable rules, or a strong default | `DecisionTree`, `RandomForest` | [Decision trees and random forests](decision-trees.md) |
| Nonlinear boundaries or robust regression with kernels | `SupportVectorMachine`, `SupportVectorRegression`, `Kernel` | [Support vector machines](support-vector-machines.md) |
