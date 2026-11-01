# One-dimensional minimisation and line searches

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Minimise a function of one variable on an interval; or, inside a multidimensional method, choose how far to move along a search direction |
| **Devised by** | Golden section search: Jack Kiefer (1953, as the limit of his Fibonacci search). Line search conditions: Larry Armijo (1966), Philip Wolfe (1969) |
| **Accuracy** | Golden section: the interval shrinks by $1/\varphi \approx 0.618$ per evaluation, down to a limit of about $\sqrt{\varepsilon}$ relative. Line search: a step satisfying the strong Wolfe conditions, which is enough for the global convergence of descent methods |
| **Cost** | Golden section: one evaluation per iteration, $\log_{1.618}(\text{width}/\text{tol})$ in all. Line search: typically one to three evaluations of $f$ and $\nabla f$ |
| **API** | [`GoldenSectionSearch`](../../api/lib_jsl_optimization/golden_section_search/struct.GoldenSectionSearch.html), [`strong_wolfe`](../../api/lib_jsl_optimization/line_search/fn.strong_wolfe.html) |

</div>

## Overview

Minimising a function of one variable is the simplest optimisation problem, and the most often
solved: every gradient-based method in this chapter reduces each iteration to a one-dimensional
problem, choosing the step length $\alpha$ along a search direction $\mathbf d$ by approximately minimising
$\varphi(\alpha) = f(\mathbf x + \alpha\mathbf d)$.

Two kinds of method are described here. **Golden section search** finds the minimum of a unimodal
function on an interval to any accuracy, using only comparisons of function values. It is the
optimisation analogue of [bisection](../root_finding/bracketing.md#bisection): slow but certain.
A **line search** inside a larger method does not need the exact minimum along the line, only a
step that makes enough progress; the **Wolfe conditions** say what "enough" means, and a good line
search finds such a step in one or two evaluations.

## History

Jack Kiefer (1953) asked how to locate the maximum of a unimodal function with a fixed number of
evaluations, and found the optimal strategy, **Fibonacci search**, whose interval reductions are
ratios of Fibonacci numbers. When the number of evaluations is not fixed in advance, the limit of
these ratios, the golden ratio, gives golden section search, almost as good and simpler. Richard
Brent (1973) combined it with parabolic interpolation, as his root finder combines bisection with
interpolation.

Inexact line searches arose with the gradient methods of the 1960s. Larry Armijo (1966) proposed
accepting any step with a sufficient decrease in $f$, found by backtracking. Philip Wolfe (1969,
1971) added a condition on the slope that rules out steps too short to make progress, and proved
that the two together guarantee convergence of a wide class of methods; Guus Zoutendijk proved a
similar theorem. Jorge Moré and David Thuente (1994) designed the line search used in most modern
software, and Nocedal and Wright's textbook version of it is the one used here.

## The idea

### Golden section search

Suppose $f$ is **unimodal** on $[a, b]$: decreasing, then increasing. Evaluate it at two interior points
$c < d$. If $f(c) < f(d)$, the minimum cannot lie in $(d, b]$, so the interval shrinks to $[a, d]$;
otherwise it shrinks to $[c, b]$. Unlike bisection, one interior point is left inside the new
interval, already evaluated. Placing the points at fractions $1 - 1/\varphi$ and $1/\varphi$ of the
interval, with $\varphi = (1 + \sqrt5)/2$ the golden ratio, makes the surviving point fall exactly where
one of the next pair must be, because $1/\varphi^2 = 1 - 1/\varphi$. Every further iteration then costs a
single evaluation and reduces the interval by the factor $1/\varphi \approx 0.618$.

Since the method uses only comparisons, it needs no smoothness: it works as well for $|x - 0.3|$ as
for $(x - 0.3)^2$. Its accuracy has a limit, however. Near a smooth minimum
$f(x) \approx f(x^*) + \tfrac12 f''(x^*)(x - x^*)^2$, and once $\tfrac12 f''(x - x^*)^2$ is smaller than the rounding error
$\varepsilon|f(x^*)|$, the function values can no longer tell which point is lower. The minimum can be
located only to about $\sqrt{\varepsilon}$ relative, $10^{-8}$ in double precision, not $10^{-16}$:

{{#include ../generated/optimization/golden_section.md}}

The first two functions have $f(x^*) = 0$, so no rounding hides the minimum and the tolerance
can be met. For $-xe^{-5x}$ (minimum $-0.0736$ at $x = 0.2$), asking for $10^{-12}$ gives a worse answer
than asking for $10^{-8}$: below the $\sqrt{\varepsilon}$ limit the comparisons are decided by rounding.
The same limit applies to every minimiser, and is why optimisation tolerances on $\mathbf x$ are
usually not set below about $10^{-8}$.

### Line searches and the Wolfe conditions

Given a point $\mathbf x$ and a **descent direction** $\mathbf d$ (one with $\nabla f \cdot \mathbf d < 0$), a step $\alpha > 0$ is
acceptable if it satisfies the **strong Wolfe conditions**

$$
f(\mathbf x + \alpha\mathbf d) \le f(\mathbf x) + c_1 \alpha\,\nabla f(\mathbf x)\cdot\mathbf d,
\qquad
|\nabla f(\mathbf x + \alpha\mathbf d)\cdot\mathbf d| \le c_2\,|\nabla f(\mathbf x)\cdot\mathbf d|,
$$

with $0 < c_1 < c_2 < 1$. The first, **sufficient decrease**, says that $f$ must fall by at least a
fraction $c_1$ of what the tangent line predicts, ruling out steps that are too long. The second, the
**curvature condition**, says that the slope along the line must have flattened by a factor $c_2$,
ruling out steps that are too short. Usual values are $c_1 = 10^{-4}$ and $c_2 = 0.9$ for Newton and
quasi-Newton methods, whose natural step $\alpha = 1$ is usually acceptable, and $c_2 = 0.1$ for conjugate
gradients, which need a more nearly exact minimum along the line.

![A line search along steepest descent](../generated/optimization/line_search.svg)

The figure shows $\varphi(\alpha)$ along the steepest descent direction from $(-1.2, 1)$ on the Rosenbrock
function. With $c_2 = 0.9$ the first trial step, $\alpha = 0.02$, is already acceptable; with $c_2 = 0.1$
the search continues to a step near the minimum along the line.

The conditions matter because of **Zoutendijk's theorem**: if every step satisfies the Wolfe
conditions and the directions do not become orthogonal to the gradient, then $\nabla f(\mathbf x_k) \to 0$,
whatever the starting point.

## Algorithm

**Golden section search** on $[a, b]$ with tolerance $\delta$:

1. Set $c = b - (b - a)/\varphi$, $d = a + (b - a)/\varphi$ and evaluate $f$ at both.
2. While $b - a > \delta$: if $f(c) < f(d)$, set $b \leftarrow d$, $d \leftarrow c$ and evaluate a new $c$; otherwise
   set $a \leftarrow c$, $c \leftarrow d$ and evaluate a new $d$.
3. Return the midpoint of $[a, b]$.

**Strong Wolfe line search** (Nocedal and Wright, algorithms 3.5 and 3.6): try the initial step;
double it while the conditions fail and $f$ keeps decreasing; as soon as an interval is known to
contain acceptable steps (because $f$ rose, or the slope turned positive), **zoom** into it by
safeguarded cubic interpolation of $\varphi$ from its values and slopes at the ends.

## Implementation

```rust,ignore
{{#include ../../../crates/optimization/src/golden_section_search.rs:golden}}
```

The bracketing phase of the line search:

```rust,ignore
{{#include ../../../crates/optimization/src/line_search.rs:wolfe_bracket}}
```

- `GoldenSectionSearch` takes its interval from `bounds`, which are required; the initial
  parameters only fix the dimension, which must be 1. Its `iterations` counts evaluations.
- An earlier version evaluated both interior points in every iteration, doubling the cost of the
  method, and panicked when no bounds were set.
- `strong_wolfe` returns `None` if $\mathbf d$ is not a descent direction. If it cannot meet the conditions
  to rounding precision it returns the best decreasing point found, so the calling method can still
  make progress.

## In practice

```rust,ignore
{{#include ../../../crates/optimization/examples/optimization_one_dimensional.rs}}
```

**Strengths**

- Golden section search needs no derivatives or smoothness, and its cost is known in advance.
- The Wolfe line search makes gradient methods globally convergent at a cost of about one
  evaluation per iteration.

**Weaknesses**

- Golden section search converges only linearly, and finds a local minimum if $f$ is not unimodal.
- No method can locate a smooth minimum more precisely than about $\sqrt{\varepsilon}$ relative.

## Exercises

1. Show that the golden section points satisfy $1/\varphi^2 = 1 - 1/\varphi$, and that this is what lets one
   point be reused.
2. How many evaluations does golden section search need to reduce $[0, 1]$ to $10^{-6}$? Compare with
   bisection for a root.
3. Fit a parabola through three points $(a, f_a)$, $(b, f_b)$, $(c, f_c)$ and find its minimum. Under
   what condition does this step make progress, and why does Brent's method safeguard it?
4. Show that for $0 < c_1 < c_2 < 1$ and $f$ smooth and bounded below along the ray, there is an
   interval of steps satisfying the strong Wolfe conditions.
5. Run gradient descent on the Rosenbrock function with exact line searches (golden section on
   each line). Compare the number of evaluations with the Wolfe line search.

## Further reading

- J. Kiefer, "Sequential minimax search for a maximum", *Proceedings of the American Mathematical
  Society* 4 (1953) 502–506.
- P. Wolfe, "Convergence conditions for ascent methods", *SIAM Review* 11 (1969) 226–235.
- J. J. Moré and D. J. Thuente, "Line search algorithms with guaranteed sufficient decrease", *ACM
  Transactions on Mathematical Software* 20 (1994) 286–307.
- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006, chapter 3.
- R. P. Brent, *Algorithms for Minimization without Derivatives*, Prentice-Hall, 1973, chapter 5.
