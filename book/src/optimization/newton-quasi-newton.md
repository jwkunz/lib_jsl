# Newton and quasi-Newton methods: Newton, BFGS and L-BFGS

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Minimise a smooth function of many variables, using second-order information computed (Newton) or learned from gradients (quasi-Newton) |
| **Devised by** | Newton's method for minimisation: from Newton and Raphson via Thomas Simpson (1740). Quasi-Newton methods: William Davidon (1959), Roger Fletcher and Michael Powell (1963); BFGS: Charles Broyden, Roger Fletcher, Donald Goldfarb and David Shanno, independently (1970); L-BFGS: Jorge Nocedal (1980), Dong Liu and Nocedal (1989) |
| **Accuracy** | Newton: quadratic convergence near a minimum with positive definite Hessian. BFGS and L-BFGS: superlinear (BFGS) or fast linear (L-BFGS) convergence |
| **Cost** | Newton: a Hessian and an $O(n^3)$ solve per iteration. BFGS: one gradient and $O(n^2)$ work and memory. L-BFGS: one gradient and $O(mn)$ work and memory for $m$ stored pairs |
| **API** | [`NewtonRaphsonMethod`](../../api/lib_jsl_optimization/newton_raphson_method/struct.NewtonRaphsonMethod.html), [`Bfgs`](../../api/lib_jsl_optimization/quasi_newton/struct.Bfgs.html), [`Lbfgs`](../../api/lib_jsl_optimization/quasi_newton/struct.Lbfgs.html) |

</div>

## Overview

Gradient methods use a linear model of $f$; Newton's method uses a **quadratic** model, built from
the gradient and the Hessian, and jumps to the model's minimum. Near a minimum this is extremely
effective: the number of correct digits doubles at each step, and the ill-conditioning that cripples
gradient descent is irrelevant, because the Hessian accounts for it exactly. The costs are
computing the $n^2$ second derivatives and solving an $n \times n$ system at every step.

**Quasi-Newton** methods keep most of the benefit without the Hessian. They build an approximation
to it (or to its inverse) from the changes in the gradient observed along the way. **BFGS** is the
most successful of them, and the default local minimiser in most software. **L-BFGS** stores only
the last few gradient changes instead of an $n \times n$ matrix, which makes it the method of choice for
large problems.

## History

Newton's method for minimisation is Newton's method for the root of $\nabla f$. Thomas Simpson
(1740) first applied it to find a stationary point.

The quasi-Newton idea came from William Davidon, a physicist at Argonne National Laboratory, who in
1959 wanted to speed up a slow minimisation running on an unreliable computer. His report,
"Variable metric method for minimization", was rejected by a journal and circulated as a technical
report. Roger Fletcher and Michael Powell (1963) recognised its power, simplified it and published
the DFP method, which made it famous; the original report was finally published in the first issue
of the *SIAM Journal on Optimization* in 1991. In 1970 four people, Broyden, Fletcher, Goldfarb and
Shanno, independently derived a better update, the dual of DFP, now called BFGS. Jorge Nocedal
(1980) showed how to apply BFGS with only a few stored vectors, and with Dong Liu (1989) made
L-BFGS the standard method for large smooth problems.

## The idea

### Newton's method

[Gradient descent](gradient-methods.md) uses only the slope of $f$ at the current point. Newton's
method also uses its curvature. Around the current point $\mathbf x_k$, approximate $f$ by the first
three terms of its Taylor series, a quadratic function of the displacement $\mathbf p$:

$$
m(\mathbf p) = f(\mathbf x_k) + \nabla f(\mathbf x_k)^{\mathsf T}\mathbf p + \tfrac12\mathbf p^{\mathsf T}\nabla^2 f(\mathbf x_k)\,\mathbf p .
$$

Here $\nabla^2 f$ is the **Hessian**, the symmetric matrix of second partial derivatives. The idea
is to jump straight to the minimum of this model. A minimum requires the gradient of $m$ with
respect to $\mathbf p$ to vanish:

$$
\nabla m(\mathbf p) = \nabla f(\mathbf x_k) + \nabla^2 f(\mathbf x_k)\,\mathbf p = \mathbf 0 .
$$

This is a linear system for the **Newton step** $\mathbf p_k$:

$$
\nabla^2 f(\mathbf x_k)\,\mathbf p_k = -\nabla f(\mathbf x_k), \qquad \mathbf x_{k+1} = \mathbf x_k + \mathbf p_k .
$$

The stationary point of $m$ is a minimum only if the Hessian is **positive definite** (the model
curves upward in every direction). There is another way to read the same formula: a minimum of
$f$ is a root of the system $\nabla f(\mathbf x) = \mathbf 0$, and this is exactly
[Newton's method for systems](../root_finding/nonlinear-systems.md#newtons-method) applied to
it, with the Hessian as the Jacobian of the gradient.

**Strengths.** If $f$ is itself a quadratic, the model is exact and the method reaches the minimum
in one step, whatever the condition number: the Hessian rescales the valley that made gradient
descent zigzag. Near any minimum with a positive definite Hessian it converges quadratically, as in
the [root-finding case](../root_finding/newton.md#newtons-method).

**Weaknesses.** Far from a minimum there is no such guarantee.

- Where the Hessian is indefinite the model has no minimum, and the step goes to a saddle point or
  a maximum of the model instead.
- Even where it is positive definite, the model may be a poor approximation at the distance of
  the step, which can overshoot and increase $f$.
- Each iteration needs the $n \times n$ Hessian and the solution of a linear system, $O(n^3)$ work.

Practical Newton methods therefore modify the Hessian to make it positive definite and add a line
search or a trust region. The library's `NewtonRaphsonMethod` is the pure method, best used close
to a minimum.

### BFGS

**Quasi-Newton** methods keep the speed of Newton's method without computing second derivatives.
They maintain an approximation of the Hessian built up from the gradients seen so far, just as
[Broyden's method](../root_finding/nonlinear-systems.md#broydens-method) builds up a Jacobian.
BFGS, named after Broyden, Fletcher, Goldfarb and Shanno, is the most successful of them.

It is convenient to approximate the *inverse* Hessian directly, by a matrix $H_k$, because then
the Newton-like step needs no linear solve:

$$
\mathbf p_k = -H_k\nabla f(\mathbf x_k), \qquad \mathbf x_{k+1} = \mathbf x_k + \alpha_k\mathbf p_k,
$$

with the step length $\alpha_k$ chosen by a line search.

**What a step teaches.** After the step we know two vectors: the displacement
$\mathbf s_k = \mathbf x_{k+1} - \mathbf x_k$ and the change in the gradient
$\mathbf y_k = \nabla f(\mathbf x_{k+1}) - \nabla f(\mathbf x_k)$. For a quadratic with Hessian $A$ the gradient
is $A\mathbf x + \mathbf b$, so these are related exactly by $\mathbf y_k = A\mathbf s_k$, or
$A^{-1}\mathbf y_k = \mathbf s_k$. We ask the new approximation to agree with this observation, the
**secant equation**

$$
H_{k+1}\mathbf y_k = \mathbf s_k .
$$

**The update.** The secant equation gives $n$ conditions on a symmetric matrix with $n(n+1)/2$
entries, so many matrices satisfy it. BFGS chooses the symmetric one closest to $H_k$ (in a
suitably weighted norm), on the principle that the old information should be kept unless it
conflicts with the new. The result is

$$
H_{k+1} = (I - \rho_k\mathbf s_k\mathbf y_k^{\mathsf T})\,H_k\,(I - \rho_k\mathbf y_k\mathbf s_k^{\mathsf T}) + \rho_k\mathbf s_k\mathbf s_k^{\mathsf T}, \qquad \rho_k = \frac1{\mathbf y_k^{\mathsf T}\mathbf s_k}.
$$

It is easy to check that this satisfies the secant equation. The rightmost bracket applied to
$\mathbf y_k$ gives $\mathbf y_k - \rho_k\mathbf y_k(\mathbf s_k^{\mathsf T}\mathbf y_k) = \mathbf y_k - \mathbf y_k = \mathbf 0$, so the
whole first term vanishes, and the second gives
$\rho_k\mathbf s_k(\mathbf s_k^{\mathsf T}\mathbf y_k) = \mathbf s_k$. The update changes $H_k$ by a matrix of rank
two, and costs $O(n^2)$ operations.

**Staying positive definite.** If $H_k$ is positive definite and $\mathbf y_k^{\mathsf T}\mathbf s_k > 0$, then
$H_{k+1}$ is positive definite too, and so $\mathbf p = -H\nabla f$ is always a descent direction.
The condition $\mathbf y_k^{\mathsf T}\mathbf s_k > 0$ says that the slope along the step has increased, that
is, the function curves upward along it. The
[Wolfe curvature condition](one-dimensional.md#line-searches-and-the-wolfe-conditions)
guarantees exactly this, which is why BFGS is paired with a Wolfe line search.

**Starting.** The first approximation is $H_0 = I$, which makes the first step a gradient descent
step. After that step it is rescaled by $\mathbf y^{\mathsf T}\mathbf s / \mathbf y^{\mathsf T}\mathbf y$, an estimate of the
inverse curvature along the step, so that later steps have about the right length.

BFGS converges superlinearly, and it corrects poor approximations of the Hessian quickly; it is
remarkably robust.

### L-BFGS

BFGS stores the dense $n \times n$ matrix $H_k$, which is impossible for a million variables.
**Limited-memory BFGS** avoids storing it at all.

The observation is that the method never needs $H_k$ itself, only the product $H_k\mathbf g$ with
the current gradient. And $H_k$ is nothing more than $H_0$ (a multiple of the identity) with $k$
updates applied, each defined by one pair of vectors $(\mathbf s_i, \mathbf y_i)$. The product can
therefore be computed from the stored pairs alone, by unrolling the update formula. The procedure
is called the **two-loop recursion**: one loop runs backwards through the pairs applying the
right-hand brackets of the updates, and a second runs forwards applying the left-hand ones. With
$m$ pairs it costs $O(mn)$ operations.

L-BFGS keeps only the most recent $m$ pairs (typically 3 to 20) and discards older ones, so its
memory is $2mn$ numbers instead of $n^2$. Old curvature information is also the least relevant,
since it was gathered far from the current point. The price is that convergence is linear
instead of superlinear, but it is fast in practice, and on large problems L-BFGS is usually the
fastest method that needs only gradients.

### Comparison

Iterations / evaluations of $f$ to reach $f < 10^{-10}$ (each method evaluates $\nabla f$ about as often as $f$,
and Newton's method also evaluates the Hessian at every iteration; gradient descent is limited to
20 000 iterations):

{{#include ../generated/optimization/gradient_methods.md}}

- On the Rosenbrock function, Newton's method needs 7 iterations and BFGS 38. Gradient descent, with
  a fixed step of 0.002, needs over 14 000 for $n = 2$ and fails within 20 000 for $n = 10$.
- On the quadratic with condition number $10^3$ Newton's method is exact in one step (the second
  confirms it). BFGS and conjugate gradients would terminate in $n = 100$ steps with exact line searches;
  with the inexact Wolfe searches used here they need a few hundred.
- L-BFGS with 10 pairs matches BFGS while storing 20 vectors instead of a $100 \times 100$ matrix.

The figure in the [chapter introduction](index.md) shows the convergence histories on the
Rosenbrock function: Newton's quadratic convergence, after a first step that *increases* $f$, and
the superlinear finish of BFGS.

## Algorithm

**Newton** (`NewtonRaphsonMethod`): repeat: evaluate $\nabla f$ and $\nabla^2 f$; solve $\nabla^2 f\,\mathbf p = -\nabla f$; set
$\mathbf x \leftarrow \mathbf x + \mathbf p$ (projected onto the bounds); stop when $f$ changes by less than the tolerance.

**BFGS** (`Bfgs`):

1. $H \leftarrow I$; compute $\mathbf g = \nabla f(\mathbf x)$.
2. $\mathbf d = -H\mathbf g$ (restart with $H = I$ if this is not a descent direction).
3. Strong Wolfe line search along $\mathbf d$ with $c_2 = 0.9$ and initial step 1.
4. Form $\mathbf s$, $\mathbf y$; if $\mathbf y^{\mathsf T}\mathbf s > 0$, update $H$ (scaling it first, after the first step).
5. Stop when $\|\mathbf g\|_\infty$ is below the tolerance.

**L-BFGS** (`Lbfgs`): the same, with $\mathbf d$ computed by the two-loop recursion from the stored pairs
and $H_0 = (\mathbf y^{\mathsf T}\mathbf s / \mathbf y^{\mathsf T}\mathbf y)\,I$ for the newest pair.

## Implementation

The Newton step:

```rust,ignore
{{#include ../../../crates/optimization/src/newton_raphson_method.rs:newton_step}}
```

The BFGS update, expanded to avoid forming matrix products:

```rust,ignore
{{#include ../../../crates/optimization/src/quasi_newton.rs:bfgs_update}}
```

The L-BFGS two-loop recursion:

```rust,ignore
{{#include ../../../crates/optimization/src/quasi_newton.rs:two_loop}}
```

- `NewtonRaphsonMethod` solves the Newton system rather than inverting the Hessian. A singular
  Hessian is an error. Earlier versions panicked there, had no constructor (so they could not be
  created outside the crate), and looped forever without an iteration limit; the default limit is
  now 100.
- `Bfgs` and `Lbfgs` stop on the gradient norm (default $10^{-8}$) and do not support bounds; setting
  bounds is an error.

## In practice

```rust,ignore
{{#include ../../../crates/optimization/examples/optimization_quasi_newton.rs}}
```

**Strengths**

- Newton: quadratic convergence, insensitive to conditioning, few iterations.
- BFGS: needs only gradients, superlinear, robust; the best general-purpose local method for
  moderate $n$.
- L-BFGS: the same in $O(mn)$ memory, for problems with up to millions of variables.

**Weaknesses**

- Newton needs the Hessian and an $O(n^3)$ solve, and the pure method may diverge or find a saddle.
- BFGS needs $O(n^2)$ memory and work per iteration.
- All find local minima only, and need a smooth $f$ (noise in the gradient corrupts the updates).

## Exercises

1. Show that Newton's method is invariant under linear changes of variables $\mathbf x = B\mathbf z$, while
   gradient descent is not.
2. Verify that the BFGS update satisfies the secant equation $H_{k+1}\mathbf y_k = \mathbf s_k$.
3. Show that if $H_k$ is positive definite and $\mathbf y_k^{\mathsf T}\mathbf s_k > 0$ then $H_{k+1}$ is positive definite.
4. Implement the two-loop recursion and check that it gives the same $H\mathbf g$ as the dense BFGS
   matrix when $m$ is at least the number of iterations.
5. Run L-BFGS on the $n$-dimensional Rosenbrock function for $n = 10, 100, 1000$ with $m = 3, 10, 30$ and
   compare iterations and time.

## Further reading

- W. C. Davidon, "Variable metric method for minimization", *SIAM Journal on Optimization* 1 (1991)
  1–17 (the 1959 report).
- R. Fletcher, *Practical Methods of Optimization*, 2nd ed., Wiley, 1987.
- D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale optimization",
  *Mathematical Programming* 45 (1989) 503–528.
- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006, chapters 3, 6 and 7.
- J. E. Dennis and R. B. Schnabel, *Numerical Methods for Unconstrained Optimization and Nonlinear
  Equations*, SIAM, 1996.
