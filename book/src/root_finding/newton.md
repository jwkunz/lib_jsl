# Newton's method and the secant method

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Find a root of a smooth function $f$ near a starting guess, without a bracket |
| **Devised by** | Newton's method: Isaac Newton (1669, for polynomials), Joseph Raphson (1690), Thomas Simpson (1740, for general functions); convergence theory by Fourier, Cauchy and Kantorovich (1948). The secant method: as old as false position; its order was analysed in the twentieth century |
| **Accuracy** | Full precision near a simple root; convergence is local and not guaranteed |
| **Cost** | Newton: one $f$ and one $f'$ per step, order 2. Secant: one $f$ per step, order $(1 + \sqrt 5)/2 \approx 1.618$ |
| **API** | [`newton_raphson`](../../api/lib_jsl_root_finding/newton_raphson/fn.newton_raphson.html), [`secant_method`](../../api/lib_jsl_root_finding/secant_method/fn.secant_method.html) |

</div>

## Overview

**Newton's method** replaces $f$ by its tangent line at the current point and takes the tangent's
zero as the next point:

$$
x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}.
$$

Near a simple root this doubles the number of correct digits at every step, which makes it the most
important iteration in numerical analysis: it computes square roots and reciprocals in hardware,
solves the implicit equations of [stiff ODE solvers](../integration/stiff-ode.md), drives
[optimisation](../../api/lib_jsl_optimization/index.html), and extends to
[systems of equations](nonlinear-systems.md). The **secant method** replaces the tangent by the line
through the last two points, needing no derivative, and is almost as fast.

Both are **open** methods: they do not keep the root bracketed, and from a poor starting point they
can wander off, cycle or diverge. They are best used when a good starting guess is available, or to
finish off after a [bracketing method](bracketing.md) has done the rough work.

## History

Newton described his method in *De analysi* (written in 1669, published in 1711), solving the cubic
$x^3 - 2x - 5 = 0$ by repeatedly substituting $x = 2 + p$, neglecting the higher powers of $p$, and
repeating with the correction; John Wallis published the example in 1685. Newton's process worked on
the polynomial, not on the sequence of approximations. Joseph Raphson (1690) gave the simpler form
that updates $x$ directly, and Thomas Simpson (1740) first stated it with derivatives (fluxions) for
general functions and for systems of two equations. Joseph Fourier (1818) and Augustin-Louis Cauchy
(1829) studied when it converges, and Leonid Kantorovich (1948) proved the theorem that guarantees
convergence from conditions at the starting point alone.

The secant method is a descendant of the ancient rule of false position, without the bracket. Its
order of convergence, the golden ratio, follows from a recurrence of Fibonacci type for the errors.

## The idea

### Newton's method

Expanding $f$ about the current point,
$0 = f(r) = f(x_k) + f'(x_k)(r - x_k) + \tfrac12 f''(\xi)(r - x_k)^2$. Dropping the quadratic term and solving
for $r$ gives the Newton step. Keeping it shows how the error $e_k = x_k - r$ evolves:

$$
e_{k+1} = \frac{f''(\xi)}{2f'(x_k)}\,e_k^2 \approx \frac{f''(r)}{2f'(r)}\,e_k^2,
$$

**quadratic convergence**. Three conditions are hidden in this formula: the root must be **simple**
($f'(r) \ne 0$), $f$ must be twice differentiable, and $e_k$ must already be small enough that
$|f''/2f'|\,|e_k| < 1$.

At a root of multiplicity $m$, $f(x) \approx c\,(x - r)^m$ and the Newton step reduces the error only by
the factor $(m - 1)/m$: convergence becomes **linear**, by a half per step at a double root. If $m$ is
known, the step $x_{k+1} = x_k - m\,f(x_k)/f'(x_k)$ restores quadratic convergence.

![Orders of convergence](../generated/root_finding/orders.svg)

### When Newton's method fails

From a poor starting point the tangent can point anywhere. The classic example is $\arctan x = 0$,
whose only root is 0: from $|x_0| < 1.3917$ the iteration converges, from $|x_0| = 1.3917\ldots$ it cycles
between $\pm x_0$ forever, and from beyond it oscillates with growing amplitude and diverges.

![Newton's method on arctan](../generated/root_finding/newton_atan.svg)

The orange path ($x_0 = 1.39$) shows the knife edge: it oscillates for several steps before falling
into the region of quadratic convergence. Other failures: a zero derivative (a horizontal tangent)
throws the iterate to infinity, and for $f(x) = x^{1/3}$ the iteration doubles the error at each step.
Newton's method in the complex plane, on $z^3 = 1$ say, makes the sensitivity to the starting point
visible as a fractal boundary between the basins of the three roots.

### The secant method

Replacing $f'(x_k)$ by the slope of the line through the last two iterates,

$$
x_{k+1} = x_k - f(x_k)\,\frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})},
$$

needs only one new value of $f$ per step. The errors satisfy $e_{k+1} \approx \frac{f''}{2f'}\,e_k\,e_{k-1}$,
whose solution grows like $e_{k+1} \sim e_k^{\,\varphi}$ with $\varphi = (1 + \sqrt5)/2 \approx 1.618$.
Measured per evaluation of a function, secant is often *faster* than Newton: two secant steps
(order $\varphi^2 \approx 2.6$) cost the same as one Newton step (order 2) when $f'$ costs as much as $f$.

## Algorithm

**Newton**: from $x_0$, repeat $x \leftarrow x - f(x)/f'(x)$ until the step is smaller than the tolerance.

**Secant**: from $x_0, x_1$, repeat the secant step until the step is smaller than the tolerance and
the two latest points confirm the root (see below).

## Implementation

```rust,ignore
{{#include ../../../crates/root_finding/src/newton_raphson.rs:newton}}
```

The secant method needs more care in deciding that it has converged:

```rust,ignore
{{#include ../../../crates/root_finding/src/secant_method.rs:secant}}
```

- A small step usually means convergence, but not always. On $x^{10} - 1$ from 0 and 1.3, the secant
  method throws an iterate out to $2.2 \times 10^6$, where $f$ is $10^{63}$; the secant through that point is
  almost vertical, and the next step is tiny, at $x = 0.18$, nowhere near a root. An earlier version of
  the library returned 0.18. Now a small step is accepted only if $f$ changes sign across it, or if the
  slope between the two nearby points (not the distant one) puts the root within the tolerance.
- Newton's method reports an error for a zero or non-finite derivative or a non-finite iterate; when
  it diverges, the derivative usually underflows to zero first, and that is the error reported.
- Both give up after 100 iterations.

## In practice

```rust,ignore
{{#include ../../../crates/root_finding/examples/root_finding_newton.rs}}
```

**Strengths**

- Quadratic (Newton) or superlinear (secant) convergence: a few steps from a good guess.
- No bracket needed, so roots of even multiplicity can be found (slowly), and the methods generalise
  to complex roots and to [systems](nonlinear-systems.md).

**Weaknesses**

- Convergence is only local: a poor guess can lead to divergence, cycling or a different root.
- Newton's method needs the derivative, which may be expensive or unavailable.
- Multiple roots reduce convergence to linear.

## Exercises

1. Show that Newton's method for $f(x) = x^2 - a$ is the Babylonian square-root iteration
   $x \leftarrow (x + a/x)/2$, and that for $f(x) = 1/x - a$ it computes $1/a$ without division.
2. Find the critical starting point for $\arctan$: the $x_0$ with $x_1 = -x_0$. Solve the equation
   $2x = (1 + x^2)\arctan x$ with Brent's method.
3. Show that Newton's method at a root of multiplicity $m$ has error ratio $(m - 1)/m$, and that the
   modified step with $m f/f'$ is quadratically convergent.
4. Derive the recurrence $e_{k+1} \approx C e_k e_{k-1}$ for the secant method and show that the order of
   convergence is the golden ratio.
5. Compare Newton and secant on Kepler's equation for eccentricities $e = 0.1, 0.5, 0.9, 0.99$ starting
   from $E_0 = M$. Which starting guess would you use for $e$ close to 1?

## Further reading

- T. J. Ypma, "Historical development of the Newton-Raphson method", *SIAM Review* 37 (1995) 531–551.
- J. E. Dennis and R. B. Schnabel, *Numerical Methods for Unconstrained Optimization and Nonlinear
  Equations*, SIAM, 1996, chapter 2.
- C. T. Kelley, *Solving Nonlinear Equations with Newton's Method*, SIAM, 2003.
- A. M. Ostrowski, *Solution of Equations and Systems of Equations*, Academic Press, 1960, on orders of
  convergence and efficiency.
