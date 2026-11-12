# Nonlinear systems: Newton's and Broyden's methods

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Solve $n$ nonlinear equations in $n$ unknowns, $\mathbf F(\mathbf x) = \mathbf 0$ with $\mathbf F: \mathbb R^n \to \mathbb R^n$ |
| **Devised by** | Newton's method for systems: Thomas Simpson (1740), Leonid Kantorovich (1948, convergence theory); line searches: Larry Armijo (1966), John Dennis and Robert Schnabel (1983); quasi-Newton updates: Charles Broyden (1965) |
| **Accuracy** | To the requested residual, near a root with a non-singular Jacobian; convergence from a distant start is not guaranteed |
| **Cost** | Newton: a Jacobian ($n$ extra evaluations if by differences) and an $O(n^3)$ solve per iteration, quadratic convergence. Broyden: one evaluation and an $O(n^3)$ solve per iteration, superlinear convergence |
| **API** | [`newton_system`](../../api/lib_jsl_root_finding/nonlinear_systems/fn.newton_system.html), [`broyden_system`](../../api/lib_jsl_root_finding/nonlinear_systems/fn.broyden_system.html), [`SystemOptions`](../../api/lib_jsl_root_finding/nonlinear_systems/struct.SystemOptions.html), [`SystemRootResult`](../../api/lib_jsl_root_finding/nonlinear_systems/struct.SystemRootResult.html) |

</div>

## Overview

Equilibria of mechanical structures and chemical reactions, the operating point of an electrical
circuit, the discretised equations of a nonlinear differential equation, the implicit stages of an ODE
solver: all are systems of nonlinear equations. In more than one dimension there is no bracket to
trap a root, and the workhorse is **Newton's method**, which linearises the system around the
current point and solves the linear system for a correction.

Two refinements make it practical. A **line search** shortens steps that would make things worse,
so that the method converges from much farther away. **Quasi-Newton** methods, of which **Broyden's**
is the classic, avoid computing the Jacobian at every step by updating an approximation of it from
the steps already taken, which saves most of the function evaluations when the Jacobian must be
formed by finite differences.

## History

Thomas Simpson (1740) applied Newton's method to two equations in two unknowns, using fluxions
(derivatives). Its convergence in general spaces was established by Leonid Kantorovich in 1948, whose
theorem guarantees convergence and a unique root in a ball from bounds on $\mathbf F$, its Jacobian and the
Jacobian's variation at the starting point.

The practical difficulty, that Newton's method diverges from poor starting points, was attacked in
the 1960s and 1970s by **globalisation**: line searches (with the sufficient-decrease condition of
Larry Armijo, 1966) and trust regions (Kenneth Levenberg, 1944, Donald Marquardt, 1963, and Michael
Powell's dogleg, 1970). Charles Broyden (1965), working at the English Electric Company on the
equations of nuclear reactors and aircraft design, proposed updating an approximate Jacobian by the
smallest change consistent with the latest step. His idea started the family of quasi-Newton
methods, which in optimisation became BFGS. John Dennis and Jorge Moré (1974) characterised exactly
when such methods converge superlinearly.

## The idea

### Newton's method

The problem is to find a vector $\mathbf x = (x_1, \ldots, x_n)$ at which $n$ functions vanish
together: $F_1(\mathbf x) = 0, \ldots, F_n(\mathbf x) = 0$, written $\mathbf F(\mathbf x) = \mathbf 0$. For
example, the intersection of the circle $x^2 + y^2 = 4$ with the hyperbola $xy = 1$ is a root of

$$
\mathbf F(x, y) = \begin{pmatrix} x^2 + y^2 - 4 \\ xy - 1 \end{pmatrix}.
$$

[Newton's method in one variable](newton.md) replaced $f$ by its tangent line. The analogue in $n$
variables replaces each $F_i$ by its first-order Taylor expansion about the current point
$\mathbf x_k$. For a small displacement $\mathbf p$,

$$
\mathbf F(\mathbf x_k + \mathbf p) \approx \mathbf F(\mathbf x_k) + J(\mathbf x_k)\,\mathbf p, \qquad J_{ij} = \frac{\partial F_i}{\partial x_j}.
$$

The matrix $J$ of all first partial derivatives is the **Jacobian**; it plays the role of $f'$. Row
$i$ is the gradient of $F_i$. For the example above,

$$
J = \begin{pmatrix} 2x & 2y \\ y & x \end{pmatrix}.
$$

The right side of the approximation is a *linear* function of $\mathbf p$, and we know how to find
where a linear function vanishes. Setting it to zero gives the **Newton step**:

1. solve the linear system $J(\mathbf x_k)\,\mathbf p = -\mathbf F(\mathbf x_k)$ for $\mathbf p$;
2. set $\mathbf x_{k+1} = \mathbf x_k + \mathbf p$.

In one dimension step 1 is the division $p = -f/f'$. In $n$ dimensions it is a linear solve by
[Gaussian elimination](../linear_algebra/lu.md). Each iteration of a nonlinear solver therefore
contains a linear solver.

As in one dimension, the convergence is quadratic near a root at which $J$ is non-singular (the
analogue of $f'(r) \ne 0$).

**Where the Jacobian comes from.** It can be supplied as a function, or formed by
[forward differences](../derivatives/finite-differences.md): perturbing $x_j$ by
$h_j = \sqrt{\varepsilon}\max(|x_j|, 1)$ and differencing $\mathbf F$ gives column $j$. That costs one extra
evaluation of $\mathbf F$ per column, $n$ in all. The small errors of a difference Jacobian barely
affect the convergence, because the Jacobian only determines the *step*; the root itself is
defined by $\mathbf F$, which is evaluated exactly.

### Line search: global convergence

Far from a root the linear model is a poor description of $\mathbf F$, and the full Newton step can
land somewhere worse than where it started. The iteration can then diverge, as it did for
$\arctan$ in [one dimension](newton.md#when-newtons-method-fails).

The remedy needs a way to say what "worse" means. With several equations there is no single
function value to watch, so we combine them into one number, the **merit function**

$$
\phi(\mathbf x) = \tfrac12\|\mathbf F(\mathbf x)\|^2 = \tfrac12\sum_i F_i(\mathbf x)^2 .
$$

It is zero at a root and positive elsewhere, so a step is an improvement if it reduces $\phi$.

**The Newton direction always points downhill.** By the chain rule, the gradient of $\phi$ is
$\nabla\phi = J^{\mathsf T}\mathbf F$. The rate of change of $\phi$ along the Newton direction $\mathbf p$
is the dot product of the two, and since $J\mathbf p = -\mathbf F$,

$$
\nabla\phi \cdot \mathbf p = \mathbf F^{\mathsf T} J \mathbf p = -\mathbf F^{\mathsf T}\mathbf F = -\|\mathbf F\|^2 < 0 .
$$

The slope is negative: $\phi$ decreases, at least initially, as we move along $\mathbf p$. So even
when the full step is too long, a *short enough* step in the same direction makes progress.

**Backtracking.** The line search takes $\mathbf x_{k+1} = \mathbf x_k + \lambda\mathbf p$ and looks for
an acceptable $\lambda \in (0, 1]$.

1. Try the full step, $\lambda = 1$.
2. Accept it if $\phi$ has decreased by at least $10^{-4}\lambda\,|\nabla\phi\cdot\mathbf p|$. This is
   the **Armijo condition**: the decrease must be at least a small fraction of what the initial
   slope predicts, which rules out steps that reduce $\phi$ only negligibly.
3. Otherwise choose a smaller $\lambda$. The values of $\phi$ already computed along the line are
   fitted by a quadratic (the first time) or a cubic (afterwards), and $\lambda$ is moved to the
   minimum of the fit. Go to 2.

Near the root the full step always passes the test, so the line search costs nothing there and
the quadratic convergence is untouched.

**What it cannot do.** The line search guarantees that $\phi$ keeps decreasing, but $\phi$ may have
local minima that are not roots. At such a point $\nabla\phi = J^{\mathsf T}\mathbf F = 0$ with
$\mathbf F \ne 0$, which requires the Jacobian to be singular. The solvers then stop and report
`converged = false`. On $x^2 + 1 = 0$, for example, which has no real root, they stop at the
minimum $x = 0$ of $\tfrac12(x^2 + 1)^2$.

### Broyden's method

When the Jacobian is formed by differences, it costs $n$ evaluations of $\mathbf F$ per iteration,
which is most of the work for large $n$. In one dimension the
[secant method](newton.md#the-secant-method) avoids the derivative by using the slope between
the last two points. Broyden's method is its generalisation.

It keeps a matrix $B_k$ that approximates the Jacobian and uses it in place of $J$ in the Newton
step. After each step it has two new pieces of information: the step taken,
$\mathbf s = \mathbf x_{k+1} - \mathbf x_k$, and the resulting change in the function,
$\mathbf y = \mathbf F(\mathbf x_{k+1}) - \mathbf F(\mathbf x_k)$. A good approximation of the Jacobian should
reproduce this observation, so the new matrix is required to satisfy the **secant equation**

$$
B_{k+1}\mathbf s = \mathbf y .
$$

In one dimension this determines the slope $B = y/s$ completely. In $n$ dimensions it is only $n$
equations for the $n^2$ entries of $B_{k+1}$: it says how $B_{k+1}$ acts on the one direction
$\mathbf s$ and nothing about the directions perpendicular to it. Broyden's choice is to leave the
action on those directions exactly as it was:

$$
B_{k+1} = B_k + \frac{(\mathbf y - B_k \mathbf s)\,\mathbf s^{\mathsf T}}{\mathbf s^{\mathsf T}\mathbf s}.
$$

Check the two properties. Applied to $\mathbf s$, the correction term gives
$(\mathbf y - B_k\mathbf s)(\mathbf s^{\mathsf T}\mathbf s)/(\mathbf s^{\mathsf T}\mathbf s) = \mathbf y - B_k\mathbf s$, so
$B_{k+1}\mathbf s = \mathbf y$. Applied to any $\mathbf w$ perpendicular to $\mathbf s$, the correction
vanishes because $\mathbf s^{\mathsf T}\mathbf w = 0$, so $B_{k+1}\mathbf w = B_k\mathbf w$. Among all
matrices satisfying the secant equation, this is the one closest to $B_k$ (in the Frobenius norm,
the square root of the sum of squared entries): it changes $B$ as little as the new information
allows.

Broyden's method converges **superlinearly**: faster than any fixed linear rate, though not
quadratically. It needs one evaluation of $\mathbf F$ per iteration (plus any used by the line
search), instead of $n + 1$. The library forms $B_0$ by differences, and forms it afresh whenever
the updated matrix stops giving a descent direction.

### Comparison

Iterations and evaluations of $\mathbf F$ (including those for difference Jacobians and line searches) to a
residual of $10^{-12}$:

{{#include ../generated/root_finding/systems.md}}

- With an exact Jacobian, Newton's method needs the fewest evaluations.
- With difference Jacobians, the cost per iteration is $n + 1$ evaluations or more; Broyden's method,
  with a few more iterations at about one evaluation each, is much cheaper for the 40 unknowns of the
  Bratu problem (45 evaluations against 124), and modestly cheaper for small systems.
- The Rosenbrock system shows the price of the safeguard. Its second equation is linear, so plain
  Newton reaches the root in two steps, through an intermediate point where $\|\mathbf F\|$ is ten times
  larger. The line search rejects that point and creeps along the curved valley instead, taking 14
  iterations. Safety is usually worth this occasional cost.

A nonlinear system can have several solutions, and which one is found depends on the starting point.
The **Bratu problem** $u'' + e^u = 0$, $u(0) = u(1) = 0$, discretised by
[finite differences](../integration/finite-differences.md), has two:

![The two solutions of the Bratu problem](../generated/root_finding/bratu.svg)

## Algorithm

**Newton** (`newton_system`), from $\mathbf x_0$:

1. If $\|\mathbf F(\mathbf x)\|_\infty \le$ the function tolerance, stop: converged.
2. Form $J(\mathbf x)$ (analytically, or by forward differences) and solve $J\mathbf p = -\mathbf F$ by Gaussian
   elimination with partial pivoting; stop if $J$ is singular.
3. Line search along $\mathbf p$, limiting the step to $100\max(\|\mathbf x\|, n)$.
4. Stop if the relative step is below the step tolerance or the line search fails; converged if the
   residual is below $\max(\text{tolerance}, 10^{-8})$.

**Broyden** (`broyden_system`): the same, with $J$ replaced by $B$, which is updated after each
accepted step and re-formed by differences when a step fails.

## Implementation

The Newton step and its line search:

```rust,ignore
{{#include ../../../crates/root_finding/src/nonlinear_systems.rs:newton_step}}
```

Broyden's update:

```rust,ignore
{{#include ../../../crates/root_finding/src/nonlinear_systems.rs:broyden_update}}
```

- The Jacobian is stored densely and solved by Gaussian elimination: fine for hundreds of unknowns.
  Large sparse systems from PDEs need sparse or iterative linear solvers (Newton-Krylov methods),
  which this crate does not provide.
- Always check `converged` in the result: the solvers return the best point found even when they fail.
- The design follows `newt` and `broydn` of *Numerical Recipes*.

## In practice

```rust,ignore
{{#include ../../../crates/root_finding/examples/root_finding_systems.rs}}
```

**Strengths**

- Quadratic (Newton) or superlinear (Broyden) convergence near a root.
- The line search gives convergence from much farther away than plain iteration.
- Broyden's method needs few function evaluations when no Jacobian is available.

**Weaknesses**

- Convergence to a root is still not guaranteed: the iteration can stop at a local minimum of $\|\mathbf F\|$.
- Dense $O(n^3)$ linear algebra per iteration, and $O(n^2)$ memory.
- Which of several roots is found depends on the starting point.

## Exercises

1. Show that the Newton direction is a descent direction for $\tfrac12\|\mathbf F\|^2$, and that it is not, in
   general, the steepest descent direction.
2. Verify that Broyden's update satisfies the secant equation and changes $B$ only on the span of
   $\mathbf s$: $B_{k+1}\mathbf v = B_k\mathbf v$ whenever $\mathbf v^{\mathsf T}\mathbf s = 0$.
3. Solve the Bratu problem with $e^u$ replaced by $\lambda e^u$ for increasing $\lambda$. Near
   $\lambda \approx 3.51$ the two solutions merge and disappear. What happens to the Jacobian there?
4. Write the intersection of the circle $x^2 + y^2 = 4$ and the hyperbola $xy = 1$ as a system and find all
   four solutions by choosing starting points.
5. Compare the evaluations used by Newton with difference Jacobians and by Broyden for the Bratu
   problem with $n = 10, 20, 40, 80$. How does the ratio grow with $n$?

## Further reading

- C. G. Broyden, "A class of methods for solving nonlinear simultaneous equations", *Mathematics of
  Computation* 19 (1965) 577–593.
- J. E. Dennis and J. J. Moré, "A characterization of superlinear convergence and its application to
  quasi-Newton methods", *Mathematics of Computation* 28 (1974) 549–560.
- J. E. Dennis and R. B. Schnabel, *Numerical Methods for Unconstrained Optimization and Nonlinear
  Equations*, Prentice-Hall, 1983 (SIAM reprint, 1996).
- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006, chapter 11.
- C. T. Kelley, *Iterative Methods for Linear and Nonlinear Equations*, SIAM, 1995.
