# Linear programming: the simplex method

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Minimise a linear function $\mathbf c^{\mathsf T}\mathbf x$ subject to linear equalities and inequalities and bounds on the variables |
| **Devised by** | Leonid Kantorovich (1939) and Tjalling Koopmans (1940s), formulation; George Dantzig (1947), the simplex method; John von Neumann (1947), duality; Robert Bland (1977), the anti-cycling rule |
| **Accuracy** | Exact in exact arithmetic: the method finds an optimal vertex, or proves the problem infeasible or unbounded, in finitely many steps |
| **Cost** | Each pivot costs $O(mn)$ for $m$ constraints and $n$ variables; the number of pivots is usually a small multiple of $m$, but exponential in the worst case |
| **API** | [`LinearProgram`](../../api/lib_jsl_optimization/linear_programming/struct.LinearProgram.html), [`LpSolution`](../../api/lib_jsl_optimization/linear_programming/struct.LpSolution.html), [`LpStatus`](../../api/lib_jsl_optimization/linear_programming/enum.LpStatus.html) |

</div>

## Overview

A **linear program** (LP) optimises a linear objective subject to linear constraints:

$$
\min_{\mathbf x}\ \mathbf c^{\mathsf T}\mathbf x \quad\text{subject to}\quad A_{\text{ub}}\mathbf x \le \mathbf b_{\text{ub}}, \quad A_{\text{eq}}\mathbf x = \mathbf b_{\text{eq}}, \quad \mathbf l \le \mathbf x \le \mathbf u .
$$

Linear programming is the workhorse of operations research: production planning, transportation
and logistics, blending, scheduling, network flows, and the relaxations used to solve integer
programs. It is solved routinely with millions of variables.

The constraints define a convex **polyhedron**, and a linear function on a polyhedron attains its
minimum (if it has one) at a **vertex**. The **simplex method** moves from vertex to adjacent vertex,
improving the objective at each step, until no neighbour is better.

## History

Leonid Kantorovich formulated linear programming in 1939 to plan production at a Leningrad
plywood trust, and Tjalling Koopmans developed it for transportation problems during the Second
World War; they shared the 1975 Nobel Prize in economics. George Dantzig, planning for the US Air
Force, invented the simplex method in 1947. On hearing of it John von Neumann immediately described
**duality**, drawing on his theory of games. The method was one of the first great applications of
computers.

Its efficiency puzzled theorists for decades. Victor Klee and George Minty (1972) built problems on
which it visits all $2^n$ vertices, but in practice it takes a small multiple of $m$ pivots, a
discrepancy explained by Daniel Spielman and Shang-Hua Teng's **smoothed analysis** (2001). Leonid
Khachiyan's ellipsoid method (1979) proved that LP is solvable in polynomial time, and Narendra
Karmarkar's interior-point method (1984) made polynomial-time methods practical. E. M. L. Beale (1955)
had shown that the textbook simplex method can **cycle** forever on degenerate problems; Robert
Bland's rule (1977) is the simplest guaranteed cure.

## The idea

### The simplex method

Adding a non-negative **slack** variable to each inequality, shifting or splitting variables to
make them all non-negative, and so on, puts the LP in **standard form**: $\min\mathbf c^{\mathsf T}\mathbf x$ subject to $A\mathbf x = \mathbf b$,
$\mathbf x \ge \mathbf 0$, with $A$ of size $m \times N$. A vertex is a **basic feasible solution**: choose $m$ **basic**
columns forming an invertible matrix $B$, set the other variables to zero, and solve $B\mathbf x_B = \mathbf b$; it is
feasible if $\mathbf x_B \ge \mathbf 0$.

From a basic feasible solution, the **reduced cost** of a non-basic variable $j$,
$\bar c_j = c_j - \mathbf c_B^{\mathsf T}B^{-1}A_j$, is the rate at which the objective changes as $x_j$ increases from 0.
If every $\bar c_j \ge 0$, no neighbour is better and the vertex is optimal. Otherwise a variable with
$\bar c_j < 0$ **enters** the basis. It increases until a basic variable reaches zero (the **ratio
test**), and that variable **leaves**. If none ever reaches zero, the objective decreases without
bound. The update of the representation is a **pivot**, one step of Gauss-Jordan elimination on the
**tableau** $[A\ |\ \mathbf b]$.

![A linear program in two variables](../generated/optimization/linear_program.svg)

In the figure (the "Wyndor Glass" problem of Hillier and Lieberman's textbook), the feasible region
is the polygon, the dashed lines are contours of the objective, and the optimum is the vertex
$(2, 6)$ where the contour last touches the region. From the origin, the textbook rule (enter the most
negative reduced cost) moves to $(0, 6)$ and then to the optimum; the library, with Bland's rule
and its phase one, takes three pivots in all.

### Two phases

A first basic feasible solution is not always obvious. **Phase one** adds an **artificial** variable
to each constraint, starting from the obvious basis of artificials, and minimises their sum. If the
minimum is positive, the LP is **infeasible**; if zero, the final basis (after driving out any
artificials left at zero) is feasible for the original problem, and **phase two** optimises the true
objective from it.

### Degeneracy and Bland's rule

When a basic variable is zero (a **degenerate** vertex, where more than $n$ constraints meet), a pivot
can change the basis without moving, and the textbook rule of entering the most negative reduced
cost can return to an earlier basis and **cycle**. **Bland's rule**, choosing the entering and
leaving variables of lowest index among the candidates, provably never cycles. It is slower on
average than smarter rules, but simple and safe; this library uses it.

The library's results on a few programs:

{{#include ../generated/optimization/linear_programs.md}}

Beale's example is a famous degenerate problem on which the textbook rule cycles; with Bland's rule
it is solved in six pivots, to its optimum $-1/20$. The last program shows how an absolute value is
minimised by linear programming, with a free variable split into two non-negative ones.

### Duality

Every LP has a **dual**: for $\min\mathbf c^{\mathsf T}\mathbf x$ subject to $A\mathbf x \ge \mathbf b$, $\mathbf x \ge 0$, it is $\max\mathbf b^{\mathsf T}\mathbf y$ subject to
$A^{\mathsf T}\mathbf y \le \mathbf c$, $\mathbf y \ge 0$. Any feasible dual value bounds the primal optimum, and at the optimum they are
equal (**strong duality**). The optimal dual variables are the **shadow prices** of the constraints,
the same as the Lagrange multipliers of [constrained optimisation](constrained.md#optimality-conditions).

## Algorithm

1. Convert to standard form: shift variables with finite lower bounds, negate those with only
   upper bounds, split free ones into two; add slack variables and rows for finite upper bounds;
   make right-hand sides non-negative.
2. Phase one: minimise the sum of the artificial variables. If it is positive, report infeasible.
3. Remove artificial variables from the basis (or drop redundant rows).
4. Phase two: pivot by Bland's rule until all reduced costs are non-negative (optimal) or a column
   has no positive entry (unbounded).
5. Map the standard-form solution back to the original variables.

## Implementation

```rust,ignore
{{#include ../../../crates/optimization/src/linear_programming.rs:simplex_iteration}}
```

- Programs are built with `LinearProgram::new(c)` and the `with_less_equal`, `with_greater_equal`,
  `with_equal`, `with_bounds` and `with_free_variable` methods; variables default to $x \ge 0$. To
  maximise, negate the objective.
- Infeasible and unbounded programs are reported through `LpSolution::status`, not as errors.
- The dense tableau suits problems of up to a few hundred variables and constraints. Large, sparse
  problems need the revised simplex method with LU factorisation, or interior-point methods.
- A safety limit of 50 000 pivots, which Bland's rule makes unreachable in exact arithmetic, used to
  return the current basis as optimal when hit; it is now an error.

## In practice

```rust,ignore
{{#include ../../../crates/optimization/examples/optimization_linear_programming.rs}}
```

**Strengths**

- Finds the global optimum, or proves infeasibility or unboundedness, in finitely many steps.
- Bland's rule makes it immune to cycling.

**Weaknesses**

- Dense tableau: memory and time grow as $mN$ per pivot.
- Bland's rule takes more pivots than the rules used in production solvers.
- Exponential worst case (Klee-Minty), though rarely seen in practice.

## Exercises

1. Solve the Wyndor problem by hand with the simplex tableau, starting from the origin.
2. Write the dual of the Wyndor problem and verify strong duality.
3. Construct the Klee-Minty cube for $n = 3$ and count the pivots taken by the most-negative-reduced-cost
   rule and by Bland's rule.
4. Formulate the problem of fitting a line minimising the largest absolute residual (the minimax or
   Chebyshev fit) as an LP, and solve it for a few points.
5. The transportation problem in the table has integer data. Why is its optimal solution integral?

## Further reading

- G. B. Dantzig, *Linear Programming and Extensions*, Princeton University Press, 1963.
- V. Chvátal, *Linear Programming*, W. H. Freeman, 1983. Contains the version of Beale's example used
  here.
- R. G. Bland, "New finite pivoting rules for the simplex method", *Mathematics of Operations
  Research* 2 (1977) 103–107.
- F. S. Hillier and G. J. Lieberman, *Introduction to Operations Research*, McGraw-Hill (many editions).
- D. A. Spielman and S.-H. Teng, "Smoothed analysis of algorithms: why the simplex algorithm usually
  takes polynomial time", *Journal of the ACM* 51 (2004) 385–463.
