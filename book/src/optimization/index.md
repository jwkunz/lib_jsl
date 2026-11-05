# Optimisation

To optimise is to find the best value of something: the parameters of a model that fit data
most closely, the design of a structure that is lightest for its strength, the schedule that
costs least, the policy that earns most. Mathematically, it is to find a minimum (or, by changing
sign, a maximum) of an **objective function** $f$, perhaps subject to **constraints** on the
variables:

$$
\min_{\mathbf x \in \mathbb R^n} f(\mathbf x) \quad\text{subject to}\quad h_i(\mathbf x) = 0, \;\; g_j(\mathbf x) \le 0 .
$$

Optimisation is closely related to [root finding](../root_finding/index.md): at a smooth
unconstrained minimum the gradient vanishes, so minimisation is root finding for $\nabla f$. But there
is more structure to exploit (the Hessian is symmetric, and progress can be measured by the
value of $f$) and more ways to fail (a stationary point may be a maximum or a saddle).

## The central questions

- **What information is available?** Only values of $f$ (direct search, stochastic methods), also
  the gradient (gradient descent, conjugate gradients, quasi-Newton methods), or also the Hessian
  (Newton's method). Each extra derivative buys faster convergence, at the cost of computing it.
- **Local or global?** Most methods find a **local** minimum, the bottom of the valley they start
  in. If $f$ is **convex** (every chord lies above the graph), every local minimum is global, and
  the problem is in a sense easy. Otherwise finding the global minimum is hard in general, and
  the stochastic methods trade many evaluations for a better chance of success.
- **Are there constraints?** Bounds on the variables, linear constraints (and a linear objective:
  [linear programming](linear-programming.md)), or general nonlinear constraints
  ([constrained optimisation](constrained.md)).
- **Is it a sum of squares?** Fitting a model to data minimises $\sum r_i^2$, and
  [nonlinear least squares](least-squares.md) methods exploit that structure.
- **Is it a sequence of decisions?** Then [dynamic programming](dynamic-programming.md) may
  break it into stages and solve it exactly.

The Rosenbrock function $f(x, y) = 100(y - x^2)^2 + (1 - x)^2$, with its long curved valley and
minimum at $(1, 1)$, is the standard test of a local method; it appears throughout the chapter.

![Methods compared on the Rosenbrock function](../generated/optimization/rosenbrock_convergence.svg)

## Choosing a method

| Your problem | Recommended | Entry |
|---|---|---|
| One variable, an interval containing the minimum | Golden section search | [One-dimensional minimisation](one-dimensional.md#golden-section-search) |
| Smooth, gradient available, moderate $n$ | BFGS | [Newton and quasi-Newton methods](newton-quasi-newton.md#bfgs) |
| Smooth, gradient available, large $n$ | L-BFGS, or nonlinear conjugate gradients | [L-BFGS](newton-quasi-newton.md#l-bfgs), [conjugate gradients](gradient-methods.md#nonlinear-conjugate-gradients) |
| Smooth, Hessian available, good starting point | Newton's method | [Newton's method](newton-quasi-newton.md#newtons-method) |
| No derivatives, few variables | Nelder-Mead or Powell | [Direct search](direct-search.md) |
| A sum of squares (fitting a model to data) | Levenberg-Marquardt | [Nonlinear least squares](least-squares.md) |
| Nonlinear constraints | SQP, or the augmented Lagrangian method | [Constrained optimisation](constrained.md) |
| Linear objective and constraints | The simplex method | [Linear programming](linear-programming.md) |
| Discrete stages or a Markov decision process | Dynamic programming | [Dynamic programming](dynamic-programming.md) |
| Many local minima, a bounded box | Differential evolution or particle swarm | [Stochastic global optimisation](stochastic.md) |

## Testing against SciPy

The references come from SciPy 1.17.1, written by `book/generator/data/make_optimization_refs.py`. For the unconstrained problems the table gives each solution's distance from the true minimiser, for the library's method and the closest SciPy method, both run to tight tolerances:

{{#include ../generated/optimization/scipy_unconstrained.md}}

Both Newton implementations stop at the five-variable Rosenbrock function's other local minimum, near $(-1, 1, 1, 1, 1)$, from this starting point. Nelder-Mead's distance reflects its stopping rule, which tests the spread of function values, so near a quadratic minimum it places $x$ to about the square root of the tolerance; SciPy's adaptive variant also requires the simplex itself to shrink. Golden-section search is limited the same way, and Brent's parabolic steps do better.

{{#include ../generated/optimization/scipy_other.md}}

The curve fits agree to the convergence tolerance of MINPACK's Levenberg-Marquardt, and the standard errors to the accuracy of the finite-difference Jacobians from which both compute the covariance. The linear programs match HiGHS to rounding, and the knapsack matches HiGHS's branch and bound on the same items. Differential evolution is stochastic. With this seed SciPy's run ended in a local minimum of Rastrigin's function while the library's reached the global one, which says more about the seeds than the implementations.

## In the library

The optimisation crate is `lib_jsl_optimization`
([API reference](../../api/lib_jsl_optimization/index.html)). Most minimisers share a small set of
traits from
[`optimization_traits`](../../api/lib_jsl_optimization/optimization_traits/index.html):

- the problem implements `ObjectiveFunction` (and, as the method requires, `GradientFunction`
  and `HessianFunction`);
- the method implements `MinimizationControls` (maximum iterations, tolerance and optional box
  bounds, each `Option`al with a method-specific default) and one of
  `GradientFreeMinimizationEngine`, `GradientBasedMinimizationEngine` or
  `HessianBasedMinimizationEngine`;
- the result is an `OptimizationResult` with the parameters, the objective value, the number of
  iterations and a `converged` flag.

The meaning of "iteration" and of the tolerance differs between methods, and is documented with
each. Least squares, constrained optimisation, linear and dynamic programming have their own
interfaces, described in their entries.
