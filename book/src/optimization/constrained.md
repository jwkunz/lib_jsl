# Constrained optimisation: SQP and the augmented Lagrangian method

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Minimise a smooth $f(\mathbf x)$ subject to nonlinear equality constraints $h_i(\mathbf x) = 0$ and inequality constraints $g_j(\mathbf x) \le 0$ |
| **Devised by** | Optimality conditions: Joseph-Louis Lagrange (1788, equalities), William Karush (1939), Harold Kuhn and Albert Tucker (1951). SQP: Robert Wilson (1963), Shih-Ping Han (1977), Michael Powell (1978). Augmented Lagrangian: Magnus Hestenes and Michael Powell (1969), extended to inequalities by R. Tyrrell Rockafellar (1973) |
| **Accuracy** | To the tolerance on the constraint violation and the optimality conditions; SQP converges superlinearly, the augmented Lagrangian method linearly in its outer iterations |
| **Cost** | SQP: per iteration, gradients of $f$ and all constraints, a dense quadratic program and an $O(n^3)$ factorisation. Augmented Lagrangian: an unconstrained minimisation (L-BFGS) per outer iteration |
| **Read first** | [Background: what a minimum looks like](index.md#background-what-a-minimum-looks-like), [Newton and quasi-Newton methods: Newton, BFGS and L-BFGS](newton-quasi-newton.md) |
| **API** | [`sqp`](../../api/lib_jsl_optimization/constrained/fn.sqp.html), [`augmented_lagrangian`](../../api/lib_jsl_optimization/constrained/fn.augmented_lagrangian.html), [`ConstrainedProblem`](../../api/lib_jsl_optimization/constrained/trait.ConstrainedProblem.html), [`ConstrainedResult`](../../api/lib_jsl_optimization/constrained/struct.ConstrainedResult.html) |

</div>

## Overview

Real problems come with constraints: a design must meet its specifications, a portfolio must be
fully invested, concentrations cannot be negative. At a constrained minimum the gradient of $f$ need
not vanish: it is balanced by the gradients of the constraints that are **active** (holding with
equality), weighted by **Lagrange multipliers** that measure how much the optimum would improve if
each constraint were relaxed.

**Sequential quadratic programming** (SQP) applies Newton's method to these optimality conditions:
each iteration minimises a quadratic model of the Lagrangian subject to linearised constraints. It
is the most effective general method for small and medium problems. The **augmented Lagrangian**
method converts the constrained problem into a sequence of unconstrained ones, each solved by any
unconstrained method, and updates multiplier estimates between them. It is simpler and more
forgiving, at the price of more function evaluations.

## History

Lagrange introduced his multipliers in the *Mécanique analytique* (1788) for mechanical systems with
equality constraints. The conditions for inequalities were found by William Karush in his 1939
master's thesis at the University of Chicago, which was not published, and again by Harold Kuhn and
Albert Tucker in 1951; they are now the **KKT conditions**.

Robert Wilson's 1963 Harvard thesis proposed solving a sequence of quadratic programs, and Shih-Ping
Han (1977) and Michael Powell (1978) made SQP practical with quasi-Newton approximations and merit
functions, establishing the method that underlies codes such as SNOPT and MATLAB's `fmincon`.
Penalty methods, adding $\rho\|\mathbf h\|^2$ to $f$ and letting $\rho \to \infty$, were older (Courant, 1943) but become
badly conditioned as $\rho$ grows. Magnus Hestenes and Michael Powell (1969) independently showed that
adding the multiplier term avoids this: the **method of multipliers**, later analysed by Rockafellar
and Dimitri Bertsekas, and the basis of codes such as LANCELOT.

## The idea

### Optimality conditions

The problem is

$$
\min_{\mathbf x} f(\mathbf x) \quad\text{subject to}\quad h_i(\mathbf x) = 0, \qquad g_j(\mathbf x) \le 0,
$$

with **equality constraints** $h_i$ and **inequality constraints** $g_j$. A point satisfying all
of them is **feasible**. Before designing an algorithm we need to know how to recognise a
solution. Without constraints the test is $\nabla f = \mathbf 0$. With constraints it is more subtle,
and is best built up in three steps.

**One equality constraint.** The feasible points lie on the surface $h(\mathbf x) = 0$, and the
gradient $\nabla h$ is perpendicular to that surface. At a constrained minimum, $f$ must not
decrease in any direction *along the surface*. That does not require $\nabla f = \mathbf 0$; it
requires only that $\nabla f$ have no component along the surface, that is, that $\nabla f$ be
perpendicular to it too. Two vectors perpendicular to the same surface are parallel:

$$
\nabla f + \lambda\,\nabla h = \mathbf 0
$$

for some number $\lambda$, the **Lagrange multiplier**. If this failed, sliding along the surface
in the direction of the leftover component of $-\nabla f$ would reduce $f$.

**One inequality constraint.** There are two cases. If the minimum lies strictly inside the
region, $g < 0$, the constraint plays no part (it is **inactive**) and $\nabla f = \mathbf 0$ as
usual. If the minimum lies on the boundary, $g = 0$, the constraint is **active** and behaves
like an equality, $\nabla f + \mu\nabla g = \mathbf 0$, with one extra requirement. Moving into the
interior (against $\nabla g$) must not decrease $f$ either, so $-\nabla f$ must point *out* of the
region, the same way as $\nabla g$. That forces $\mu \ge 0$. Both cases are covered by a single
statement: $\nabla f + \mu\nabla g = \mathbf 0$ with $\mu \ge 0$ and $\mu\,g = 0$, where the last
equation says that either the multiplier or the constraint is zero.

**The general case.** With several constraints the gradient of $f$ is balanced by a combination
of the constraint gradients. It is convenient to collect everything into the **Lagrangian**

$$
L(\mathbf x, \boldsymbol\lambda, \boldsymbol\mu) = f(\mathbf x) + \sum_i\lambda_i h_i(\mathbf x) + \sum_j\mu_j g_j(\mathbf x).
$$

Under mild conditions on the constraints, a local minimum $\mathbf x^*$ has multipliers satisfying
the **Karush-Kuhn-Tucker (KKT) conditions**

$$
\nabla f + \sum_i\lambda_i\nabla h_i + \sum_j\mu_j\nabla g_j = \mathbf 0, \qquad
h_i = 0,\quad g_j \le 0,\quad \mu_j \ge 0,\quad \mu_j g_j = 0 .
$$

The first equation says that the gradient of $L$ with respect to $\mathbf x$ vanishes
(**stationarity**). The next two are feasibility. The last, **complementarity**, says that an
inactive constraint ($g_j < 0$) has $\mu_j = 0$: only active constraints exert a "force" on the
solution.

The multipliers are not just bookkeeping. $\mu_j$ is the rate at which the optimal value would
fall if the constraint $g_j \le 0$ were relaxed to $g_j \le \epsilon$: it is the price of that
constraint. A large multiplier marks a constraint that is costing a great deal.

### Sequential quadratic programming

The KKT conditions are a system of nonlinear equations (and inequalities) in $\mathbf x$ and the
multipliers. SQP solves them by a Newton-like iteration, in which each step is itself a simpler
optimisation problem.

**The subproblem.** At the current point, replace $f$ by a quadratic model and each constraint by
its linearisation, and solve for the step $\mathbf d$:

$$
\min_{\mathbf d}\ \tfrac12\mathbf d^{\mathsf T}B\mathbf d + \nabla f^{\mathsf T}\mathbf d \quad\text{subject to}\quad
h_i + \nabla h_i^{\mathsf T}\mathbf d = 0, \quad g_j + \nabla g_j^{\mathsf T}\mathbf d \le 0 .
$$

A problem with a quadratic objective and linear constraints is a **quadratic program** (QP), and
it can be solved exactly in a finite number of steps. Its solution gives the step $\mathbf d$, and its
own multipliers are the new estimates of $\boldsymbol\lambda$ and $\boldsymbol\mu$. The matrix $B$ approximates the
Hessian of the *Lagrangian*, not of $f$ alone, because the curvature of the constraints matters as
much as that of the objective. With $B$ the exact Hessian, the iteration is Newton's method
applied to the KKT conditions.

The library's implementation makes three further choices.

- **The Hessian approximation.** $B$ is maintained by a
  [BFGS update](newton-quasi-newton.md#bfgs), so no second derivatives are needed. The Lagrangian
  can have negative curvature in some directions even at a solution, which would break the
  ordinary BFGS update; Powell's **damping** modifies the update so that $B$ stays positive
  definite regardless.
- **Solving the QP.** With $B$ positive definite the QP is solved through its **dual**. The step
  can be written $\mathbf d = -B^{-1}(\nabla f + A^{\mathsf T}\mathbf y)$, where the rows of $A$ are the
  constraint gradients and $\mathbf y$ holds the multipliers. The multipliers minimise a convex
  quadratic subject only to the simple bounds $y_j \ge 0$ for the inequalities, a much easier
  problem, which an active-set method solves.
- **Accepting the step.** A step may reduce $f$ while violating the constraints more, or the
  reverse, so progress is measured by the **$\ell_1$ merit function**

$$
f(\mathbf x) + \nu\Big(\sum_i|h_i(\mathbf x)| + \sum_j\max(g_j(\mathbf x), 0)\Big),
$$

  which adds to $f$ a penalty $\nu$ times the total constraint violation. A line search along
  $\mathbf d$ reduces this function. With $\nu$ larger than the largest multiplier, its minimum is the
  constrained minimum. Because violations are penalised and not forbidden, SQP can start from an
  infeasible point.

### The augmented Lagrangian method

The second method reduces the constrained problem to a sequence of unconstrained ones. Consider
equality constraints first.

**The penalty idea, and its flaw.** The simplest approach adds a penalty for violating the
constraints and minimises $f(\mathbf x) + \tfrac{\rho}{2}\|\mathbf h(\mathbf x)\|^2$ without constraints. The
minimiser of this function always violates the constraints slightly, by an amount proportional
to $1/\rho$, because a small violation costs little and can buy a reduction in $f$. Exact
feasibility needs $\rho \to \infty$, and the penalised function then has a valley with extremely
steep walls, which is ill-conditioned and hard to minimise.

**Adding the multiplier term.** The **augmented Lagrangian** adds the penalty to the Lagrangian
instead of to $f$:

$$
L_\rho(\mathbf x, \boldsymbol\lambda) = f(\mathbf x) + \boldsymbol\lambda^{\mathsf T}\mathbf h(\mathbf x) + \tfrac{\rho}{2}\|\mathbf h(\mathbf x)\|^2 .
$$

If $\boldsymbol\lambda$ were the exact multiplier vector, the minimiser of $L_\rho$ over $\mathbf x$ would be the
constrained solution itself, for any sufficiently large *finite* $\rho$. The multiplier term
supplies the force that holds the solution on the constraints, so the penalty no longer has to.

**Estimating the multipliers.** The exact multipliers are unknown, but each minimisation improves
the estimate. At the minimiser of $L_\rho$ its gradient vanishes:

$$
\nabla f + \sum_i\big(\lambda_i + \rho\,h_i(\mathbf x)\big)\nabla h_i = \mathbf 0 .
$$

Compare this with the KKT stationarity condition: the quantity $\lambda_i + \rho h_i$ stands
exactly where the true multiplier should. This suggests the update
$\boldsymbol\lambda \leftarrow \boldsymbol\lambda + \rho\,\mathbf h(\mathbf x)$.

**The method.** Each outer iteration minimises $L_\rho$ over $\mathbf x$ with an unconstrained method
(here [L-BFGS](newton-quasi-newton.md#l-bfgs)) and then updates the multipliers by the rule
above. The penalty $\rho$ is increased tenfold whenever the violation fails to fall by a factor
of four.

**Inequalities** are handled by Rockafellar's form of the augmented Lagrangian, which adds
$\tfrac1{2\rho}\sum_j(\max(0, \mu_j + \rho g_j)^2 - \mu_j^2)$ and updates
$\mu_j \leftarrow \max(0, \mu_j + \rho g_j)$. The $\max$ implements the two cases of the optimality
conditions: a constraint that is comfortably satisfied gets a zero multiplier and drops out,
while an active one is treated like an equality.

### An example

Minimise $(x - 2)^2 + (y - 1)^2$ subject to $y \ge x^2$ and $x + y \le 2$, from the infeasible point
$(-1.5, 0.5)$. The objective is the squared distance from $(2, 1)$, so the unconstrained minimum is
$(2, 1)$, which violates both constraints. The solution is $(1, 1)$, where both are active, with
multipliers $\mu_1 = \mu_2 = 2/3$:

![A constrained problem](../generated/optimization/constrained.svg)

<p class="caption" id="fig-5-7"><strong>Figure 5.7.</strong> A constrained problem.</p>

The KKT conditions can be checked by hand. In the standard form the constraints are
$g_1 = x^2 - y \le 0$ and $g_2 = x + y - 2 \le 0$. At $(1, 1)$ the gradients are
$\nabla f = (-2, 0)$, $\nabla g_1 = (2, -1)$ and $\nabla g_2 = (1, 1)$, and indeed

$$
\nabla f + \tfrac23\nabla g_1 + \tfrac23\nabla g_2 = \big(-2 + \tfrac43 + \tfrac23,\ \ 0 - \tfrac23 + \tfrac23\big) = (0, 0).
$$

<p class="caption" id="tab-5-6"><strong>Table 5.6.</strong> Constrained optimisers on test problems.</p>

{{#include ../generated/optimization/constrained.md}}

SQP reaches the solution in a few iterations; the augmented Lagrangian method needs more outer
iterations and many more evaluations, each outer iteration being a full minimisation. On Hock and
Schittkowski's problem 71, a standard test with four variables, one equality, one nonlinear
inequality and eight bounds, the derivatives are computed by central differences, which dominate
the evaluation counts. Both methods find $f^* = 17.0140173$ and the active constraints, the product
constraint and the bound $x_1 \ge 1$, with the published multipliers.

## Algorithm

**Sequential quadratic programming** (`sqp`). Input: the objective, the constraints and a
starting point $\mathbf x$, which need not be feasible. Start with $B = I$. Repeat:

1. **Evaluate** $f$, the constraints, and the gradients of all of them at $\mathbf x$.
2. **Solve the quadratic program** for the step $\mathbf d$ and the multiplier estimates $\mathbf y$,
   through its dual.
3. **Test.** Stop if the constraint violation is below the tolerance and either the step or the
   KKT residual (the gradient of the Lagrangian) is too.
4. **Update the penalty.** Raise $\nu$ if necessary so that it exceeds the largest multiplier in
   magnitude; this keeps $\mathbf d$ a descent direction for the merit function.
5. **Line search.** Starting from a full step, shorten the step along $\mathbf d$ until the merit
   function decreases sufficiently, and move there.
6. **Update $B$** by the damped BFGS formula, using the step and the change in the gradient of
   the Lagrangian.

Each iteration costs one evaluation of all the gradients and the solution of one dense QP.

**Augmented Lagrangian** (`augmented_lagrangian`). Input as above. Start with all multipliers
zero and $\rho = 10$. Repeat:

1. **Inner minimisation.** Minimise $L_\rho(\mathbf x, \boldsymbol\lambda, \boldsymbol\mu)$ over $\mathbf x$ with L-BFGS,
   starting from the previous solution.
2. **Update the multipliers**: $\boldsymbol\lambda \leftarrow \boldsymbol\lambda + \rho\,\mathbf h(\mathbf x)$ and
   $\mu_j \leftarrow \max(0, \mu_j + \rho\,g_j(\mathbf x))$.
3. **Test.** Measure the violation: $|h_i|$ for the equalities, and for each inequality either
   the amount by which it is violated or, if it is satisfied with a nonzero multiplier, the
   failure of complementarity. Stop if the largest is below the tolerance.
4. **Update the penalty.** If the violation did not fall by a factor of four since the last
   outer iteration, multiply $\rho$ by 10.

Each outer iteration is a complete unconstrained minimisation, which is why the method needs
many more evaluations than SQP.

## Implementation

The SQP subproblem through its dual:

```rust,ignore
{{#include ../../../crates/optimization/src/constrained.rs:sqp_subproblem}}
```

The augmented Lagrangian outer iteration:

```rust,ignore
{{#include ../../../crates/optimization/src/constrained.rs:multiplier_update}}
```

- Problems implement the `ConstrainedProblem` trait: only the objective is required; constraints
  default to none and all derivatives to central differences.
- The QP is dense and solved by Gaussian elimination: suitable for tens to a few hundred variables
  and constraints.
- The result reports the multipliers; positive inequality multipliers identify the active
  constraints.

## In practice

```rust,ignore
{{#include ../../../crates/optimization/examples/optimization_constrained.rs}}
```

**Strengths**

- SQP: superlinear convergence, few evaluations, starts from infeasible points.
- Augmented Lagrangian: simple, reuses any unconstrained solver, tolerant of degenerate constraints.
- Both return Lagrange multipliers, which are sensitivities of the optimum to the constraints.

**Weaknesses**

- Local methods: with non-convex constraints they may find a local solution, or report
  infeasibility for a feasible problem.
- SQP's dense QP limits it to moderate sizes; inconsistent linearised constraints make the QP fail.
- The augmented Lagrangian method needs many evaluations.

## Exercises

1. Verify the KKT conditions at $(1, 1)$ for the example, and show that $\mu_1 = \mu_2 = 2/3$.
2. Solve $\min x + y$ subject to $x^2 + y^2 = 2$ by Lagrange multipliers. What is the sign of $\lambda$, and
   what does it mean?
3. Show that the quadratic penalty method $\min f + \tfrac\rho2\|\mathbf h\|^2$ has a solution that violates the
   constraint by $O(1/\rho)$, and that the Hessian has condition number $O(\rho)$.
4. Derive the dual of the equality-constrained QP $\min\tfrac12\mathbf d^{\mathsf T}B\mathbf d + \mathbf g^{\mathsf T}\mathbf d$ subject to $A\mathbf d = -\mathbf c$.
5. Solve the box problem of the example program with the constraint $x + y + z \le 2.5$ added. Which
   constraints are active?

<details>
<summary>Hints and answers</summary>

1. At $(1, 1)$: $\nabla f + \mu_1\nabla g_1 + \mu_2\nabla g_2 = (-2, 0) + \mu_1(2, -1) + \mu_2(1, 1) = 0$ gives $2\mu_1 + \mu_2 = 2$ and $-\mu_1 + \mu_2 = 0$, so $\mu_1 = \mu_2 = 2/3 \ge 0$. Both constraints are active ($g_1 = g_2 = 0$), so complementarity holds.
2. With $L = x + y + \lambda(x^2 + y^2 - 2)$, stationarity gives $1 + 2\lambda x = 0$ and $1 + 2\lambda y = 0$, so $x = y = \pm1$. The minimum is $(-1, -1)$ with $f = -2$ and $\lambda = \tfrac12 > 0$. With this sign convention $\lambda = -\mathrm dp^*/\mathrm dc$ for the constraint $x^2 + y^2 = c$: the rate at which the optimal value falls as the circle is enlarged ($f^* = -\sqrt{2c}$, so $-\mathrm df^*/\mathrm dc = 1/\sqrt{2c} = \tfrac12$ at $c = 2$).
3. At the penalty minimiser $\nabla f + \rho\,\mathbf h\,\nabla\mathbf h = 0$, so $\mathbf h = -\boldsymbol\lambda^*/\rho + o(1/\rho)$, where $\boldsymbol\lambda^*$ is the multiplier: the violation is $O(1/\rho)$. The Hessian is $\nabla^2 L + \rho\nabla\mathbf h\nabla\mathbf h^{\mathsf T}$: it has eigenvalues of size $\rho$ along the constraint normals and of size $O(1)$ in the tangent directions, so its condition number is $O(\rho)$.
4. With $L = \tfrac12\mathbf d^{\mathsf T}B\mathbf d + \mathbf g^{\mathsf T}\mathbf d + \boldsymbol\lambda^{\mathsf T}(A\mathbf d + \mathbf c)$, stationarity gives $\mathbf d = -B^{-1}(\mathbf g + A^{\mathsf T}\boldsymbol\lambda)$. Substituting, the dual is $\max_{\boldsymbol\lambda}\; -\tfrac12(\mathbf g + A^{\mathsf T}\boldsymbol\lambda)^{\mathsf T}B^{-1}(\mathbf g + A^{\mathsf T}\boldsymbol\lambda) + \mathbf c^{\mathsf T}\boldsymbol\lambda$, an unconstrained concave quadratic.
5. The problem as stated is infeasible: since $(x + y + z)^2 \ge 3(xy + yz + zx) = 9$, the surface-area constraint forces $x + y + z \ge 3$, and the unit cube (sum 3) is already the smallest. Try a bound above 3 (say $3.3$) to see an active inequality, or a bound on a single side such as $x \le 0.8$.

</details>

## Further reading

- J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006, chapters 12, 17
  and 18.
- M. J. D. Powell, "A fast algorithm for nonlinearly constrained optimization calculations", in
  *Numerical Analysis*, Lecture Notes in Mathematics 630, Springer, 1978, 144–157.
- D. P. Bertsekas, *Constrained Optimization and Lagrange Multiplier Methods*, Academic Press, 1982.
- W. Hock and K. Schittkowski, *Test Examples for Nonlinear Programming Codes*, Lecture Notes in
  Economics and Mathematical Systems 187, Springer, 1981.
- H. W. Kuhn, "Nonlinear programming: a historical view", *SIAM-AMS Proceedings* 9 (1976) 1–26.
