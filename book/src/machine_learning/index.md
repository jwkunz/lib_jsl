# Machine learning: regression and classification

**Supervised learning** fits a function from examples: given feature vectors $\mathbf x_i$ and targets $y_i$, learn $f$ with $f(\mathbf x) \approx y$
for new $\mathbf x$. When $y$ is a number the problem is **regression**; when it is a category it is **classification**. The aim is not to fit the
training data, which a flexible enough model can always do, but to **generalise**: to predict well on data the model has
not seen. Every method in this chapter has a knob that trades flexibility against stability (a penalty, a number of
neighbours, a depth, a kernel width), and choosing it well matters more than the choice of method.

## The central questions

- **Bias or variance?** A rigid model misses real structure (bias); a flexible one fits the noise (variance). Penalties,
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

{{#include ../generated/machine_learning/optimality.md}}

Writing this chapter fixed multinomial naive Bayes without smoothing, which dropped a class from the vote (a NaN from
$0 \times (-\infty)$) whenever a word unseen in that class was absent from the document.

### Against SciPy

Each model's objective was reproduced with SciPy 1.17.1's linear algebra, optimisers and distributions (`book/generator/data/make_machine_learning_refs.py`), so the references are the same mathematical solutions computed independently:

{{#include ../generated/machine_learning/scipy_reference.md}}

The closed-form fits agree to rounding. The lasso, elastic net and logistic regression are solved iteratively on both sides, and agree to the stopping tolerances of the optimisers.

## Comparing classifiers

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
