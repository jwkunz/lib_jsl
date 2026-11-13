# Nonlinear least squares and curve fitting

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Minimise a sum of squares $\tfrac12\sum_i r_i(\mathbf p)^2$ of residuals, typically to fit a model $y = \phi(x; \mathbf p)$ to data $(x_i, y_i)$, and estimate the uncertainty of the fitted parameters |
| **Devised by** | Least squares: Adrien-Marie Legendre (1805) and Carl Friedrich Gauss (1809); Gauss-Newton: Gauss; damping: Kenneth Levenberg (1944) and Donald Marquardt (1963); the damping update used here: Hans Bruun Nielsen (1999) |
| **Accuracy** | Quadratic convergence for zero-residual problems, fast linear convergence for small residuals; parameter standard errors from the covariance $(J^{\mathsf T}J)^{-1}$ |
| **Cost** | Per iteration: one Jacobian (analytic, or $n$ extra residual evaluations by differences) and an $n \times n$ solve, with $O(mn^2)$ work to form $J^{\mathsf T}J$ |
| **API** | [`LevenbergMarquardt`](../../api/lib_jsl_optimization/least_squares/struct.LevenbergMarquardt.html), [`curve_fit`](../../api/lib_jsl_optimization/least_squares/fn.curve_fit.html), [`CurveFitResult`](../../api/lib_jsl_optimization/least_squares/struct.CurveFitResult.html) |

</div>

## Overview

Fitting a model to measurements is the most common optimisation problem in science. Given data
$(x_i, y_i)$, $i = 1, \ldots, m$, and a model $\phi(x; \mathbf p)$ with $n < m$ parameters, the **least-squares**
estimate minimises

$$
F(\mathbf p) = \tfrac12\sum_{i=1}^m r_i(\mathbf p)^2, \qquad r_i(\mathbf p) = \frac{y_i - \phi(x_i; \mathbf p)}{\sigma_i},
$$

where $\sigma_i$ is the standard deviation of the noise in $y_i$. If the noise is Gaussian, this is the
maximum-likelihood estimate. When $\phi$ depends linearly on $\mathbf p$, $F$ is quadratic and the minimum
comes from one [linear least-squares solve](../../api/lib_jsl_linalg/index.html); this entry is about
the nonlinear case.

The sum-of-squares structure gives an approximate Hessian for free, which the **Gauss-Newton**
method exploits. The **Levenberg-Marquardt** method makes it robust by blending it with gradient
descent, and it is the standard algorithm for curve fitting: SciPy's `curve_fit`, MINPACK's
`lmder`, gnuplot's `fit` and this library's `curve_fit` all use it.

## History

Least squares was published by Adrien-Marie Legendre in 1805, in an appendix on the orbits of
comets, and by Carl Friedrich Gauss in 1809, who claimed to have used it since 1795 and justified it
by the normal distribution of errors. Gauss's most famous application was the recovery of the
asteroid Ceres in 1801, from a few weeks of observations, by fitting an orbit to them. To solve the
nonlinear equations he linearised them about the current estimate and solved the linear least-squares
problem: the Gauss-Newton method.

Kenneth Levenberg, at the Frankford Arsenal in Philadelphia, proposed in 1944 to damp the
Gauss-Newton step to keep it from overshooting. Donald Marquardt, a statistician at DuPont,
rediscovered and refined the idea in 1963, scaling the damping by the diagonal of $J^{\mathsf T}J$; the
resulting algorithm, distributed as a FORTRAN program, spread rapidly. Jorge Moré's MINPACK
implementation (1978) recast it as a trust-region method, and Hans Bruun Nielsen (1999) proposed
the smooth damping update used here.

## The idea

### Gauss-Newton

$F$ could be minimised by any general method, but its special form, a sum of squares, makes a
better one possible. Start by differentiating. Let $J$ be the $m \times n$ **Jacobian** of the
residuals, $J_{ij} = \partial r_i/\partial p_j$: row $i$ says how residual $i$ responds to each
parameter. By the chain rule,

$$
\frac{\partial F}{\partial p_j} = \sum_i r_i\,\frac{\partial r_i}{\partial p_j},
\qquad
\frac{\partial^2 F}{\partial p_j\,\partial p_k} = \sum_i \frac{\partial r_i}{\partial p_j}\frac{\partial r_i}{\partial p_k} + \sum_i r_i\,\frac{\partial^2 r_i}{\partial p_j\,\partial p_k},
$$

or in matrix form

$$
\nabla F = J^{\mathsf T}\mathbf r, \qquad \nabla^2 F = J^{\mathsf T}J + \sum_i r_i\,\nabla^2 r_i .
$$

The Hessian has two parts. The first, $J^{\mathsf T}J$, involves only first derivatives of the model.
The second involves second derivatives, each multiplied by a residual. That second part is small
in two common situations: when the residuals are small (the model fits the data well), or when
the model is nearly linear in the parameters (its second derivatives are small).

Gauss's idea is to drop the second part. The approximate Hessian $J^{\mathsf T}J$ comes for free
once the gradient has been computed, and it is always positive semidefinite, so it never points
the step uphill. [Newton's step](newton-quasi-newton.md#newtons-method) with this Hessian is the
**Gauss-Newton** step,

$$
J^{\mathsf T}J\,\mathbf h = -J^{\mathsf T}\mathbf r .
$$

There is a second way to arrive at the same step, which explains what it means. Linearise the
residuals about the current parameters, $\mathbf r(\mathbf p + \mathbf h) \approx \mathbf r + J\mathbf h$, and minimise
the length of the linearised residual, $\min_{\mathbf h}\|\mathbf r + J\mathbf h\|$. That is a *linear*
least-squares problem, and the equations above are its
[normal equations](../linear_algebra/least-squares.md). Gauss-Newton thus solves a nonlinear
fitting problem as a sequence of linear ones.

Gauss-Newton converges quadratically when the residuals at the solution are zero, and rapidly
when they are small. But it has no safeguard: from a poor start the linearisation can be badly
wrong and the iteration can diverge, and when $J$ is rank deficient (two parameters with the same
effect on the model, say) the system has no unique solution.

### The Levenberg-Marquardt method

Levenberg-Marquardt repairs both defects with one modification: it adds a positive **damping**
term to the matrix,

$$
(J^{\mathsf T}J + \mu D)\,\mathbf h = -J^{\mathsf T}\mathbf r, \qquad D = \operatorname{diag}(J^{\mathsf T}J).
$$

The damping parameter $\mu \ge 0$ moves the step continuously between two extremes.

- For small $\mu$ the added term is negligible and the step is the Gauss-Newton step: fast, but
  trustworthy only near the solution.
- For large $\mu$ the term $\mu D$ dominates, and $\mathbf h \approx -D^{-1}J^{\mathsf T}\mathbf r/\mu$. Since
  $J^{\mathsf T}\mathbf r$ is the gradient of $F$, this is a short step in the (scaled) steepest descent
  direction: slow, but certain to reduce $F$ if short enough.

The damped matrix is also positive definite whenever $\mu > 0$, so the system can always be
solved. Marquardt's choice of $D$, the diagonal of $J^{\mathsf T}J$ instead of the identity, makes
the method give the same iterates if a parameter is measured in different units.

**Adapting the damping.** After each trial step, compare the reduction in $F$ that actually
occurred with the reduction predicted by the linearised model. Their ratio $\rho$ measures how far
the linearisation can be trusted.

- If $\rho > 0$, $F$ decreased. The step is accepted and $\mu$ is reduced, by the factor
  $\max(\tfrac13, 1 - (2\rho - 1)^3)$, moving towards Gauss-Newton. The better the agreement, the
  larger the reduction.
- Otherwise the step is rejected and $\mu$ is increased, by a factor that doubles with each
  consecutive failure, moving towards short gradient steps.

In effect the damping controls the radius of a **trust region** around the current point, within
which the linear model is believed. The method behaves like gradient descent while far from the
solution and like Gauss-Newton as it closes in.

### Curve fitting and uncertainties

A fit is not complete without an estimate of how well the parameters are determined. Noise in the
data moves the minimum of $F$, and the parameters inherit an uncertainty from it.

Near the solution $F$ is approximately quadratic, with Hessian $J^{\mathsf T}J$. A direction in
parameter space along which $F$ rises steeply is well determined by the data; a direction along
which it is nearly flat is poorly determined. This is why the **covariance matrix** of the fitted
parameters is approximately the *inverse* of the Hessian,

$$
C = (J^{\mathsf T}J)^{-1},
$$

provided the residuals were weighted by the true noise levels $\sigma_i$. The diagonal entries
$C_{jj}$ are the variances of the individual parameters, so the **standard errors** are
$\sqrt{C_{jj}}$; the off-diagonal entries say how the errors of different parameters are correlated.

The quality of the fit as a whole is measured by $\chi^2 = \sum_i r_i^2$. Each residual, being an
error divided by its standard deviation, should have a square of about 1 on average, but fitting
$n$ parameters uses up $n$ of the $m$ degrees of freedom. For a correct model and correct
$\sigma_i$ the expected value is $m - n$, so the **reduced chi-squared** $\chi^2/(m - n)$ should be
close to 1. A value much larger than 1 means the model does not describe the data, or the
$\sigma_i$ are too small; a value much smaller than 1 means the $\sigma_i$ are too large.

If the noise level is unknown, this argument can be run backwards: assume the model is right and
estimate the noise from the scatter of the residuals. `curve_fit` (called with `sigma = None`)
does this by scaling $C$ by the reduced chi-squared, as SciPy does.

A Gaussian peak on a baseline, $y = a\,e^{-(x - \mu)^2/2s^2} + b$, fitted to 60 noisy points from a poor
starting guess:

![Fitting a Gaussian peak](../generated/optimization/curve_fit.svg)

{{#include ../generated/optimization/curve_fit.md}}

Every estimate lies within one standard error of the truth, as it should about two-thirds of the
time.

## Algorithm

**Levenberg-Marquardt.** Input: the residual function, optionally its Jacobian, and starting
parameters $\mathbf p$.

1. **Initialise.** Evaluate $\mathbf r$ and $J$ at $\mathbf p$. Set the damping to
   $\mu = 10^{-3}\max_i (J^{\mathsf T}J)_{ii}$, a small fraction of the scale of the matrix it will be
   added to.
2. **Test the gradient.** If the largest component of $J^{\mathsf T}\mathbf r$ in magnitude is tiny,
   stop: $\mathbf p$ is a stationary point.
3. **Compute a trial step.** Solve $(J^{\mathsf T}J + \mu D)\mathbf h = -J^{\mathsf T}\mathbf r$. If $\mathbf h$ is tiny
   relative to $\mathbf p$, stop: the parameters have stopped changing.
4. **Evaluate it.** Compute $\mathbf r$ at $\mathbf p + \mathbf h$ (projected onto the bounds, if set), and
   the gain ratio $\rho$: the actual reduction in $F$ divided by the reduction the linear model
   predicted.
5. **Accept or reject.**
   - If $\rho > 0$: move to $\mathbf p + \mathbf h$, recompute $J$ there, and decrease $\mu$. If $F$ fell
     by only a relatively tiny amount, stop.
   - Otherwise: stay at $\mathbf p$ and increase $\mu$, by a factor that doubles at each consecutive
     rejection.
6. Go to 2.

Each trial step costs one evaluation of the residuals and an $O(n^3)$ solve; each accepted step
also costs a Jacobian, which by forward differences is $n$ more evaluations of the residuals.

**Curve fitting** (`curve_fit`). Build the residuals $r_i = (y_i - \phi(x_i; \mathbf p))/\sigma_i$,
run the iteration above, and then compute $C = (J^{\mathsf T}J)^{-1}$ at the solution, the standard
errors $\sqrt{C_{jj}}$ and $\chi^2$. If no $\sigma_i$ were given, use $\sigma_i = 1$ and multiply
$C$ by the reduced chi-squared.

## Implementation

```rust,ignore
{{#include ../../../crates/optimization/src/least_squares.rs:lm_step}}
```

- `LevenbergMarquardt::solve` takes any residual function $\mathbb R^n \to \mathbb R^m$ and an optional analytic
  Jacobian (forward differences otherwise), and returns the parameters, residuals, cost and the
  Jacobian at the solution.
- `curve_fit` builds the weighted residuals from a model $\phi(x, \mathbf p)$ and data, uses difference
  Jacobians, and returns the covariance, standard errors and chi-squared. A rank-deficient Jacobian
  (parameters that the data cannot separate) makes the covariance undefined and is an error.
- Box bounds are enforced by projection.

## In practice

```rust,ignore
{{#include ../../../crates/optimization/examples/optimization_least_squares.rs}}
```

**Strengths**

- Fast and robust for fitting problems; only first derivatives are needed.
- Gives parameter uncertainties as a by-product.

**Weaknesses**

- Finds a local minimum: models with several minima (sums of exponentials or sinusoids) need good
  starting values.
- Converges slowly for problems with large residuals at the solution, where the dropped Hessian
  term matters.
- The covariance is a linearisation, unreliable for strongly nonlinear models or poorly
  determined parameters.

## Exercises

1. Show that for a linear model $\phi = A\mathbf p$ the Gauss-Newton method converges in one step.
2. Show that the Levenberg-Marquardt step minimises $\|\mathbf r + J\mathbf h\|^2$ subject to $\|D^{1/2}\mathbf h\| \le \Delta$ for some
   $\Delta$ depending on $\mu$.
3. Fit $y = a e^{-bx}$ to data both directly and by linear regression of $\ln y$. Why do the answers
   differ, and which is correct for additive Gaussian noise?
4. Repeat the Gaussian-peak fit for 1000 noise realisations and check that the spread of the
   estimates matches the standard errors.
5. Fit a sum of two Gaussians to data from a single Gaussian. What happens to the covariance?

## Further reading

- K. Levenberg, "A method for the solution of certain non-linear problems in least squares",
  *Quarterly of Applied Mathematics* 2 (1944) 164–168.
- D. W. Marquardt, "An algorithm for least-squares estimation of nonlinear parameters", *Journal of
  the Society for Industrial and Applied Mathematics* 11 (1963) 431–441.
- J. J. Moré, "The Levenberg-Marquardt algorithm: implementation and theory", in *Numerical Analysis*,
  Lecture Notes in Mathematics 630, Springer, 1978, 105–116.
- K. Madsen, H. B. Nielsen and O. Tingleff, *Methods for Non-Linear Least Squares Problems*, 2nd ed.,
  Technical University of Denmark, 2004.
- P. R. Bevington and D. K. Robinson, *Data Reduction and Error Analysis for the Physical Sciences*, 3rd
  ed., McGraw-Hill, 2003.
