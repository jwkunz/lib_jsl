# Direct search: the Nelder-Mead and Powell methods

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Minimise a function of a few variables using only its values: no derivatives, which may be unavailable, expensive or meaningless (noisy or non-smooth $f$) |
| **Devised by** | Simplex search: William Spendley, George Hext and Francis Himsworth (1962); the adaptive simplex: John Nelder and Roger Mead (1965). Conjugate directions without derivatives: Michael Powell (1964) |
| **Accuracy** | Nelder-Mead: no general convergence guarantee, but usually reaches a local minimum to the tolerance on function values. Powell: converges on smooth functions, finitely on quadratics with exact line searches |
| **Cost** | Nelder-Mead: one or two evaluations per iteration, $n + 1$ for a shrink. Powell: $n + 1$ line searches per iteration. Both are practical for up to about 10 to 20 variables |
| **API** | [`NelderMeadMethod`](../../api/lib_jsl_optimization/nelder_mead_method/struct.NelderMeadMethod.html), [`PowellsMethod`](../../api/lib_jsl_optimization/powells_method/struct.PowellsMethod.html) |

</div>

## Overview

Sometimes the gradient is not available: $f$ is the output of a simulation, a laboratory
measurement or a black-box program; or it is not differentiable, with kinks, steps or noise. **Direct
search** methods minimise using only comparisons and values of $f$.

The **Nelder-Mead** method moves a **simplex** of $n + 1$ points (a triangle in two dimensions, a
tetrahedron in three) through the space, reflecting its worst point through the others,
stretching along promising directions and shrinking around the best point when nothing else
works. It is simple, robust in practice and very widely used; MATLAB's `fminsearch` and SciPy's
default derivative-free method are Nelder-Mead. **Powell's method** performs one-dimensional
minimisations along a set of directions, updating the set so that it becomes conjugate. On smooth
functions it can be much more accurate.

## History

William Spendley, George Hext and Francis Himsworth, statisticians at Imperial Chemical Industries,
proposed in 1962 to optimise industrial processes by moving a regular simplex of experimental
conditions, reflecting the worst. John Nelder and Roger Mead, at the National Vegetable Research
Station in Wellesbourne, England, let the simplex change shape, expanding and contracting to adapt
to the landscape, in a 1965 paper that became one of the most cited in numerical computing. Its
theory lagged far behind its popularity: Jeffrey Lagarias, James Reeds, Margaret and Paul Wright
(1998) proved convergence in one and two dimensions for strictly convex functions, and Ken McKinnon
(1998) gave a smooth convex function in two dimensions on which it converges to a point that is not
a minimum.

Michael Powell, at the UK Atomic Energy Research Establishment at Harwell, published his method
in 1964, showing how to generate conjugate directions without derivatives. He later developed a
long line of derivative-free methods based on quadratic models (COBYLA, NEWUOA, BOBYQA).

## The idea

### The Nelder-Mead simplex method

Order the $n + 1$ vertices so that $f(\mathbf x_1) \le \cdots \le f(\mathbf x_{n+1})$, and let $\bar{\mathbf x}$ be the centroid of all
but the worst. Each iteration tries to replace the worst vertex $\mathbf x_{n+1}$ by a point on the line
through it and $\bar{\mathbf x}$:

- **reflect**: $\mathbf x_r = \bar{\mathbf x} + (\bar{\mathbf x} - \mathbf x_{n+1})$. If it is better than the second worst but not
  the best, accept it.
- **expand**: if $\mathbf x_r$ is the best point so far, try going twice as far, $\mathbf x_e = \bar{\mathbf x} + 2(\bar{\mathbf x} - \mathbf x_{n+1})$, and
  keep the better.
- **contract**: if $\mathbf x_r$ is no better than the second worst, try a point halfway between $\bar{\mathbf x}$ and $\mathbf x_r$
  (outside) or $\mathbf x_{n+1}$ (inside).
- **shrink**: if that fails too, move every vertex halfway towards the best.

The simplex thus rolls downhill, stretches out along valleys, squeezes through narrow passages and
finally contracts onto a minimum. It stops when the function values at the vertices agree to the
tolerance (their standard deviation falls below it). The initial simplex steps 5% along each
coordinate from the starting point, as `fminsearch` does.

![Direct search on the Rosenbrock function](../generated/optimization/direct_search.svg)

### Powell's method

Powell's method keeps a set of $n$ directions, initially the coordinate axes. Each iteration
minimises $f$ along each direction in turn, by a one-dimensional search (bracketing, then Brent's
parabolic interpolation), and then considers the overall displacement $\mathbf x_n - \mathbf x_0$ of the iteration
as a new direction. On a quadratic, after $n$ iterations, the directions generated this way are
mutually conjugate and the minimum is reached exactly. To keep the directions from becoming linearly
dependent, the new direction replaces the old one along which $f$ decreased most, and only when a
test shows the replacement is worthwhile.

## Algorithm

**Nelder-Mead** with coefficients 1 (reflection), 2 (expansion), ½ (contraction), ½ (shrink): sort
the vertices; stop if the spread of their values is below the tolerance; otherwise reflect, expand,
contract or shrink as above. Trial points are projected onto the bounds, if set.

**Powell**: for each direction, minimise along it from the current point; stop when an iteration
reduces $f$ by less than the relative tolerance; otherwise try the extrapolated point $2\mathbf x_n - \mathbf x_0$,
and, if the test passes, minimise along $\mathbf x_n - \mathbf x_0$ and make it a direction.

## Implementation

```rust,ignore
{{#include ../../../crates/optimization/src/nelder_mead_method.rs:nelder_mead_step}}
```

```rust,ignore
{{#include ../../../crates/optimization/src/powells_method.rs:powell_iteration}}
```

- Bounds are enforced by projecting every trial point onto the box. Projection can flatten a simplex
  against a face: a start on an upper bound once had its initial step projected straight back onto the
  bound, so the simplex never left that face and returned a wrong point as converged. The initial
  step now points inward in that case.

On the Rosenbrock function (evaluations to convergence, and the final value):

{{#include ../generated/optimization/direct_search.md}}

Nelder-Mead stops when its vertices' values agree to $10^{-8}$, so its answers are accurate only to about
that level; Powell's method keeps going to the bottom, at a higher price. A gradient method, when a
gradient is available, is far cheaper.

## In practice

```rust,ignore
{{#include ../../../crates/optimization/examples/optimization_direct_search.rs}}
```

**Strengths**

- Only function values are needed, and $f$ need not be smooth.
- Nelder-Mead is robust and makes rapid early progress on small problems; Powell's method is
  accurate on smooth ones.

**Weaknesses**

- Slow and unreliable beyond about 10 to 20 variables.
- Nelder-Mead has no convergence guarantee and can stall at non-minima (McKinnon's example); restarting
  from the solution is a cheap check.
- Both find local minima only.

## Exercises

1. Run Nelder-Mead on $f(x, y) = x^2 + y^2$ from $(1, 1)$ and draw the first ten simplices.
2. Show that if $\mathbf p$ and $\mathbf q$ are minima of a quadratic along two parallel lines with direction $\mathbf d$, then
   $\mathbf q - \mathbf p$ is conjugate to $\mathbf d$. This is the principle of Powell's method.
3. Minimise $f(x, y) = |x| + 2|y|$ (non-differentiable at the minimum) with Nelder-Mead, Powell and BFGS.
4. Implement McKinnon's function and observe Nelder-Mead converging to a non-stationary point.
5. Add noise of size $10^{-6}$ to the Rosenbrock function and compare the accuracy reached by Nelder-Mead
   and by BFGS with finite-difference gradients.

## Further reading

- J. A. Nelder and R. Mead, "A simplex method for function minimization", *The Computer Journal* 7
  (1965) 308–313.
- M. J. D. Powell, "An efficient method for finding the minimum of a function of several variables
  without calculating derivatives", *The Computer Journal* 7 (1964) 155–162.
- J. C. Lagarias, J. A. Reeds, M. H. Wright and P. E. Wright, "Convergence properties of the
  Nelder-Mead simplex method in low dimensions", *SIAM Journal on Optimization* 9 (1998) 112–147.
- K. I. M. McKinnon, "Convergence of the Nelder-Mead simplex method to a nonstationary point", *SIAM
  Journal on Optimization* 9 (1998) 148–158.
- A. R. Conn, K. Scheinberg and L. N. Vicente, *Introduction to Derivative-Free Optimization*, SIAM,
  2009.
