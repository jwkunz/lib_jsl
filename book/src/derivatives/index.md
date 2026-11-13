# Numerical differentiation

Derivatives are everywhere in numerical computation: Newton's method for equations needs them,
optimisation algorithms follow gradients, stiff ODE solvers need Jacobians, and sensitivity analysis
asks how outputs change with inputs. When a formula for the derivative is not available, or the
function is a procedure rather than a formula, the derivative must be computed numerically.

Differentiation is in one sense the opposite of integration, and its numerical behaviour is
opposite too. [Integration](../integration/index.md) averages values and is forgiving: errors in
individual function values are smoothed out. Differentiation subtracts nearby values and divides by
a small number: it **amplifies** errors. Every numerical differentiation method is a compromise
between the error of approximating the derivative (which wants small steps) and the amplification
of rounding errors (which wants large ones).

## Choosing a method

| Your situation | Recommended | Entry |
|---|---|---|
| A function you can only evaluate, moderate accuracy | Central differences with a balanced step | [Finite difference derivatives](finite-differences.md) |
| Higher accuracy from function values alone | Five-point stencils, or Richardson extrapolation | [Finite difference derivatives](finite-differences.md#higher-order-stencils) |
| Gradients, Hessians and Jacobians | Coordinate-wise central differences | [Finite difference derivatives](finite-differences.md#gradients-hessians-and-jacobians) |
| An analytic function you can evaluate at complex points | Cauchy integral formula | [Complex contour integration](../integration/contour-integration.md#cauchys-integral-formula-and-derivatives) |
| A smooth periodic function or one on an interval, sampled on a grid | Spectral differentiation | [Spectral methods](../integration/spectral-methods.md) |

## Testing against SciPy

SciPy 1.17.1's `scipy.differentiate` estimates derivatives adaptively: it applies central differences with a sequence of shrinking steps and extrapolates (Richardson), stopping when the estimates agree. The library uses one fixed step chosen to balance truncation and rounding. The references come from `book/generator/data/make_derivatives_refs.py`; the tables give relative errors against the exact derivatives:

<p class="caption" id="tab-1-1"><strong>Table 1.1.</strong> Columns: function, f': central difference, f': five-point stencil, f': SciPy `derivative`, and more.</p>

{{#include ../generated/derivatives/scipy_scalar.md}}

<p class="caption" id="tab-1-2"><strong>Table 1.2.</strong> Columns: derivative (SciPy reference), library: relative error, SciPy: relative error.</p>

{{#include ../generated/derivatives/scipy_vectors.md}}

Adaptive extrapolation reaches 13 to 15 digits where the fixed-step formulas give 9 to 14, at the cost of many more evaluations, but it needs a sensible first step. SciPy's default first step of 0.5 jumps past zero for $\ln x$ at $x = 0.1$ and returns NaN; the library's step is relative to $\max(|x|, 1)$, so it stays near the point, but for the same reason it is too large for a function that varies on a scale much smaller than 1, which costs it accuracy at $x = 0.1$. The second derivatives of the five-point stencil lose more digits, as the step must be larger to keep rounding error, now divided by $h^2$, in check.

## In the library

The differentiation crate is `lib_jsl_derivatives`
([API reference](../../api/lib_jsl_derivatives/index.html)).
