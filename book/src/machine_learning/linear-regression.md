# Linear regression: least squares, ridge, lasso and elastic net

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Fit $y \approx X\mathbf w + b$ to data, optionally with weighted or correlated errors, or with a penalty that shrinks or selects the weights |
| **Devised by** | Adrien-Marie Legendre (1805) and Carl Friedrich Gauss (1809), least squares; Alexander Aitken (1935), generalised least squares; Andrey Tikhonov (1943) and Arthur Hoerl and Robert Kennard (1970), ridge regression; Robert Tibshirani (1996), the lasso; Hui Zou and Trevor Hastie (2005), the elastic net; Jerome Friedman, Trevor Hastie and Robert Tibshirani (2007), coordinate descent |
| **Accuracy** | Optimality conditions satisfied to $10^{-16}$ (least squares, ridge) and $10^{-10}$ (lasso, elastic net) |
| **Cost** | Least squares and ridge: $O(np^2)$ by SVD or normal equations. Elastic net: $O(np)$ per sweep of coordinate descent |
| **API** | [`LinearRegression`](../../api/lib_jsl_machine_learning/supervised/regression/least_squares/struct.LinearRegression.html), [`GeneralLeastSquares`](../../api/lib_jsl_machine_learning/supervised/regression/general_least_squares/struct.GeneralLeastSquares.html), [`Ridge`](../../api/lib_jsl_machine_learning/supervised/regression/ridge/struct.Ridge.html), [`ElasticNet`](../../api/lib_jsl_machine_learning/supervised/regression/elastic_net/struct.ElasticNet.html) |

</div>

## Overview

Linear regression predicts a number as a weighted sum of the features plus an intercept. It is the oldest learning
method and still the first to try: fast, interpretable, and with a complete statistical theory. Polynomial and other basis
expansions make it fit curves, since the model only needs to be linear in the *weights*. Its weakness is variance: with many
or correlated features, or few samples, the least-squares weights become large and unstable. **Penalties** cure this:
**ridge** shrinks all the weights, the **lasso** sets some exactly to zero, and the **elastic net** combines the two.

## History

Adrien-Marie Legendre published the method of least squares in 1805 to fit comet orbits; Carl Friedrich Gauss, who claimed
to have used it since 1795, justified it in 1809 by normal errors, and later (1821) proved it optimal among linear unbiased
estimators, the Gauss-Markov theorem. Francis Galton's work on heredity gave "regression" its name (1886). Alexander Aitken
(1935) extended least squares to correlated errors. Andrey Tikhonov (1943) regularised ill-posed integral equations, and
Arthur Hoerl and Robert Kennard (1970) introduced the same idea to statistics as ridge regression. Robert Tibshirani's lasso
(1996) brought the $\ell_1$ penalty and variable selection, Bradley Efron and colleagues (2004) traced its whole solution path with
LARS, and Hui Zou and Trevor Hastie (2005) proposed the elastic net. Jerome Friedman, Hastie and Tibshirani (2007, 2010)
showed that simple coordinate descent solves these problems faster than anything else, the basis of the glmnet package.

## The idea

### Least squares

Minimise the residual sum of squares $\lVert\mathbf y - X\mathbf w - b\rVert^2$. Appending a column of ones to $X$ for the intercept, the solution satisfies the
**normal equations** $X^{\mathsf T}X\mathbf w = X^{\mathsf T}\mathbf y$: the residual is orthogonal to every column. The library solves the problem through the
[singular value decomposition](../linear_algebra/svd.md), which returns the minimum-norm solution when features are
collinear or outnumber the samples, rather than forming $X^{\mathsf T}X$ and squaring its condition number (see
[linear least squares](../linear_algebra/least-squares.md)).

A flexible enough linear model interpolates noise. With 20 points, a polynomial of degree 15 swings wildly between them,
while degree 3 and a penalised degree-15 fit follow the underlying curve:

![Polynomial fits](../generated/machine_learning/polynomial_fits.svg)

### Weighted and generalised least squares

When the errors have covariance $\Sigma$, the best linear unbiased estimator minimises $(\mathbf y - X\mathbf w)^{\mathsf T}\Sigma^{-1}(\mathbf y - X\mathbf w)$. Factorising $\Sigma = LL^{\mathsf T}$
([Cholesky](../linear_algebra/cholesky.md)) and multiplying by $L^{-1}$ **whitens** the problem into ordinary least squares. Independent errors
with variances $\sigma_i^2$ give **weighted least squares** with weights $1/\sigma_i^2$: noisy points count for less.

![Weighted least squares](../generated/machine_learning/wls.svg)

{{#include ../generated/machine_learning/wls.md}}

Both estimators are unbiased, but weighting by the inverse variance cuts the spread of the slope by 40% and of the intercept
by a factor of 3.7.

### Ridge regression

Minimise $\lVert\mathbf y - X\mathbf w - b\rVert^2 + \alpha\lVert\mathbf w\rVert^2$. The solution $\mathbf w = (X^{\mathsf T}X + \alpha I)^{-1}X^{\mathsf T}\mathbf y$ (on centred data, so the intercept is not penalised)
exists for any $\alpha > 0$, even with more features than samples. In terms of the SVD $X = U\Sigma V^{\mathsf T}$, ridge multiplies each component of the
least-squares solution by $\sigma_j^2/(\sigma_j^2 + \alpha)$, damping the directions with small singular values, where noise is amplified most. The
penalty trades bias for variance; the test error is lowest at an intermediate $\alpha$:

![Ridge error](../generated/machine_learning/ridge_error.svg)

### The lasso and the elastic net

The lasso replaces the squared penalty by $\alpha\lVert\mathbf w\rVert_1$. The corners of the $\ell_1$ ball make the solution **sparse**: as $\alpha$ grows, weights reach
exactly zero one by one, selecting features. The elastic net minimises

$$
\frac{1}{2n}\lVert\mathbf y - X\mathbf w - b\rVert^2 + \alpha\Big(\rho\lVert\mathbf w\rVert_1 + \frac{1 - \rho}{2}\lVert\mathbf w\rVert^2\Big),
$$

with $\rho = 1$ the lasso and $\rho = 0$ a scaled ridge. Among correlated features the lasso tends to pick one arbitrarily; the ridge part spreads
the weight among them. On eight features of which three matter (and one is correlated with the first):

| | |
|:---:|:---:|
| ![Ridge paths](../generated/machine_learning/ridge_path.svg) | ![Lasso paths](../generated/machine_learning/lasso_path.svg) |

Ridge shrinks all weights smoothly towards zero and never reaches it; the lasso sets the irrelevant weights to zero at once
and the real ones only for large $\alpha$.

### Coordinate descent

The elastic-net objective is not differentiable where a weight is zero, but it is easy one coordinate at a time. With the
others fixed, the optimal $w_j$ is

$$
w_j = \frac{S\big(\tfrac1n\mathbf x_j^{\mathsf T}\mathbf r^{(j)}, \alpha\rho\big)}{\tfrac1n\lVert\mathbf x_j\rVert^2 + \alpha(1 - \rho)}, \qquad S(z, \gamma) = \operatorname{sign}(z)\max(|z| - \gamma, 0),
$$

where $\mathbf r^{(j)}$ is the residual without feature $j$ and $S$ the **soft-thresholding** operator. Cycling through the coordinates, keeping the residual
up to date, converges to the global minimum because the objective is convex and its non-smooth part separable.

## Algorithm

1. **Least squares**: append a column of ones; solve by SVD for the minimum-norm solution.
2. **Generalised**: Cholesky factor $\Sigma = LL^{\mathsf T}$ (or $\sqrt{w_i}$ for weights); solve the whitened least-squares problem.
3. **Ridge**: centre $X$ and $\mathbf y$; solve $(X_c^{\mathsf T}X_c + \alpha I)\mathbf w = X_c^{\mathsf T}\mathbf y_c$; $b = \bar y - \bar{\mathbf x}^{\mathsf T}\mathbf w$.
4. **Elastic net**: centre; sweep coordinates with the soft-threshold update until the largest change is below the tolerance.

## Implementation

```rust,ignore
{{#include ../../../crates/machine_learning/src/supervised/regression/least_squares.rs:ols_fit}}
```

```rust,ignore
{{#include ../../../crates/machine_learning/src/supervised/regression/general_least_squares.rs:whitening}}
```

```rust,ignore
{{#include ../../../crates/machine_learning/src/supervised/regression/ridge.rs:ridge_fit}}
```

```rust,ignore
{{#include ../../../crates/machine_learning/src/supervised/regression/elastic_net.rs:coordinate_descent}}
```

- Ridge's $\alpha$ multiplies a sum of squares; the elastic net's multiplies a *mean* ($\frac{1}{2n}$), as in scikit-learn, so the two scales differ by $2n$.
- Coordinate descent stops after `max_iterations` sweeps without an error; check `iterations()`.

## In practice

```rust,ignore
{{#include ../../../crates/machine_learning/examples/ml_linear_regression.rs}}
```

```text
least squares                  test MSE 4.001
ridge, α = 10                  test MSE 2.068
lasso, α = 0.2                 test MSE 1.352
elastic net, α = 0.2, ρ = 0.5  test MSE 1.417
lasso keeps 7 of 20 weights: w0 = +2.91, w1 = -2.47, w2 = +1.27, w4 = -0.13, w8 = -0.11, w14 = +0.01, w16 = +0.16
```

With 30 samples and 20 features, least squares has a test error four times the noise variance; the lasso finds the three
real features, drops the near-copy $x_3$, and comes closest to the noise level of 1.

**Strengths**

- Fast, deterministic and interpretable; SVD-based least squares handles collinearity.
- Penalties give stable predictions with many features, and the lasso selects them.

**Weaknesses**

- Linear in the features: curves need basis expansions chosen by hand.
- Squared error is sensitive to outliers (see [support vector regression](support-vector-machines.md#support-vector-regression) for an ε-insensitive loss).
- Penalties depend on feature scales; standardise first. The penalty strength must be chosen, typically by cross-validation.

## Exercises

1. Derive the normal equations by setting the gradient of $\lVert\mathbf y - X\mathbf w\rVert^2$ to zero, and show that the residual is orthogonal to the columns of $X$.
2. Show that ridge regression is least squares on the data augmented with $p$ rows $\sqrt\alpha I$ and zero targets.
3. Derive the soft-threshold update of coordinate descent for the lasso from the subgradient of $|w_j|$.
4. Show that if $\alpha \ge \max_j\frac1n|\mathbf x_j^{\mathsf T}\mathbf y_c|$ the lasso solution is $\mathbf w = 0$.
5. Show that the generalised least-squares estimator has covariance $(X^{\mathsf T}\Sigma^{-1}X)^{-1}$, and that it is no larger than that of ordinary least squares.

## Further reading

- T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009, chapter 3.
- R. Tibshirani, "Regression shrinkage and selection via the lasso", *Journal of the Royal Statistical Society B* 58 (1996) 267–288.
- H. Zou and T. Hastie, "Regularization and variable selection via the elastic net", *Journal of the Royal Statistical Society B* 67 (2005) 301–320.
- J. Friedman, T. Hastie and R. Tibshirani, "Regularization paths for generalized linear models via coordinate descent", *Journal of
  Statistical Software* 33 (2010) 1–22.
- A. E. Hoerl and R. W. Kennard, "Ridge regression: biased estimation for nonorthogonal problems", *Technometrics* 12 (1970) 55–67.
