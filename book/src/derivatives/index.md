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

## In the library

The differentiation crate is `lib_jsl_derivatives`
([API reference](../../api/lib_jsl_derivatives/index.html)).
