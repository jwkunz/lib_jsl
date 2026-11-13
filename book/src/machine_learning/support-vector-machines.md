# Support vector machines and kernels

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Separate classes by the hyperplane with the widest margin, in a feature space defined implicitly by a kernel; fit regression functions with an ε-insensitive loss |
| **Devised by** | James Mercer (1909), positive-definite kernels; Vladimir Vapnik and Alexey Chervonenkis (1963–1974), maximal margins and learning theory; Bernhard Boser, Isabelle Guyon and Vapnik (1992), the kernel trick; Corinna Cortes and Vapnik (1995), soft margins; Harris Drucker et al. (1997), support vector regression; John Platt (1998), sequential minimal optimisation; Rong-En Fan, Pai-Hsuen Chen and Chih-Jen Lin (2005), second-order working sets |
| **Accuracy** | SMO stops when the maximal KKT violation is below $10^{-3}$ (LIBSVM's default) |
| **Cost** | Training between $O(n^2)$ and $O(n^3)$ kernel evaluations; prediction $O(n_{\text{SV}}p)$ per point |
| **API** | [`SupportVectorMachine`](../../api/lib_jsl_machine_learning/supervised/classification/support_vector_machine/struct.SupportVectorMachine.html) (`new(kernel, c)`, `decision_function`, `support_vector_count`), [`SupportVectorRegression`](../../api/lib_jsl_machine_learning/supervised/regression/support_vector_regression/struct.SupportVectorRegression.html) (`new(kernel, c, epsilon)`), [`Kernel`](../../api/lib_jsl_machine_learning/kernels/enum.Kernel.html) (`Linear`, `Polynomial`, `Rbf`, `Sigmoid`, `gram`, `rbf_scaled`) |

</div>

## Overview

Among all hyperplanes that separate two classes, the **support vector machine** (SVM) picks the one farthest from the nearest
points, the **maximal margin**. The solution depends only on those nearest points, the **support vectors**. Allowing some points
inside the margin at a price $C$ (the **soft margin**) handles overlapping classes. And because the optimisation only involves dot
products between points, replacing them by a **kernel** $k(\mathbf x, \mathbf x')$ fits a linear separator in a high- or infinite-dimensional feature space
without ever computing it, giving curved boundaries in the original space. Before deep learning, kernel SVMs were the
most accurate general-purpose classifiers, and they remain strong on small and medium data sets.

## History

James Mercer (1909) characterised the kernels that are inner products in some feature space, and Nachman Aronszajn (1950)
developed the theory of reproducing kernel Hilbert spaces. Vladimir Vapnik and Alexey Chervonenkis, in Moscow from 1963,
studied optimal separating hyperplanes and founded statistical learning theory. Bernhard Boser, Isabelle Guyon and Vapnik
(1992) at Bell Labs combined maximal margins with kernels, and Corinna Cortes and Vapnik (1995) added the soft margin. Harris
Drucker, Christopher Burges, Linda Kaufman, Alex Smola and Vapnik (1997) adapted the method to regression. John Platt's
sequential minimal optimisation (1998) made training practical; Rong-En Fan, Pai-Hsuen Chen and Chih-Jen Lin (2005) improved
its working-set selection in LIBSVM, the most widely used implementation.

## The idea

### The maximal-margin hyperplane

**The margin.** Take two classes with labels $y_i = +1$ and $y_i = -1$, and a linear classifier
that predicts the sign of $f(\mathbf x) = \mathbf w^{\mathsf T}\mathbf x + b$. The boundary $f(\mathbf x) = 0$ is a
hyperplane (a line in two dimensions). If the classes can be separated at all, many hyperplanes
separate them. The support vector machine chooses the one that is as far as possible from the
nearest training points, on the reasoning that a boundary with room to spare on both sides is
least likely to misclassify new points.

To make this precise, note that the distance of a point $\mathbf x$ from the hyperplane is
$|f(\mathbf x)|/\lVert\mathbf w\rVert$. Multiplying $\mathbf w$ and $b$ by a constant does not move the
hyperplane, so we may fix the scale by requiring that the closest points have $|f| = 1$. Every
training point then satisfies $y_if(\mathbf x_i) \ge 1$, the closest points lie on the two parallel
hyperplanes $f = \pm1$, and the gap between those, the **margin**, has width $2/\lVert\mathbf w\rVert$.
Maximising the margin is therefore the same as minimising $\lVert\mathbf w\rVert$.

**The soft margin.** Real classes overlap, and no hyperplane separates them. The **soft-margin**
SVM allows points to violate the margin, at a price:

$$
\min_{\mathbf w, b, \boldsymbol\xi}\ \tfrac12\lVert\mathbf w\rVert^2 + C\sum_i\xi_i \quad\text{subject to}\quad y_if(\mathbf x_i) \ge 1 - \xi_i,\ \xi_i \ge 0 .
$$

The **slack** $\xi_i$ measures how far point $i$ intrudes into the margin: zero if it is on the
correct side with room to spare, between 0 and 1 if it is inside the margin but correctly
classified, and more than 1 if it is on the wrong side. The parameter $C$ sets the price of
violations. Small $C$ gives a wide margin with many violations; large $C$ gives a narrow margin
that fits the training data closely.

At the optimum each slack is as small as its constraint allows,
$\xi_i = \max(0, 1 - y_if(\mathbf x_i))$. This function of the classifier's output is called the
**hinge loss**, and substituting it shows that the SVM minimises the total hinge loss plus an
$\ell_2$ penalty on the weights, with strength $1/(2C)$. In that form it resembles
[penalised logistic regression](logistic-regression.md#separation-and-the-penalty), with a
different loss. The difference matters: the hinge loss is exactly zero for points beyond the
margin, so those points have no influence on the solution at all.

![Linear SVM with margins](../generated/machine_learning/svm_margin.svg)

<p class="caption" id="fig-11-9"><strong>Figure 11.9.</strong> Linear SVM with margins.</p>

Only the six points on or inside the margins (highlighted) determine the boundary; the others could move freely without
changing it. These are the **support vectors**.

### The dual problem and kernels

**The dual.** The problem above is a quadratic program with linear constraints, and
[Lagrange duality](../optimization/constrained.md#optimality-conditions) transforms it into an
equivalent problem in the Lagrange multipliers $\alpha_i$, one for each training point's
constraint:

$$
\min_{\boldsymbol\alpha}\ \tfrac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,\mathbf x_i^{\mathsf T}\mathbf x_j - \sum_i\alpha_i \quad\text{subject to}\quad 0 \le \alpha_i \le C,\ \sum_i\alpha_iy_i = 0 .
$$

The solution of the original problem is recovered as $\mathbf w = \sum_i\alpha_iy_i\mathbf x_i$, so the
classifier is

$$
f(\mathbf x) = \sum_i\alpha_iy_i\,\mathbf x_i^{\mathsf T}\mathbf x + b .
$$

Two features of the dual are important. First, by the complementarity condition, a multiplier is
nonzero only if its constraint is active: $\alpha_i = 0$ for every point outside the margin. Only
the support vectors have $\alpha_i > 0$, so the sum for $f$ has few terms. Second, the training
points appear, in both the problem and the classifier, *only* through dot products
$\mathbf x_i^{\mathsf T}\mathbf x_j$.

**The kernel trick.** A linear boundary is often not enough. One remedy is to compute new
features from the old, for instance all products of pairs of features, and fit a linear
classifier in the larger feature space; a hyperplane there is a curved surface in the original
space. The feature space may be huge. But since the SVM needs only dot products, it never needs
the new features themselves, only a function

$$
k(\mathbf x, \mathbf x') = \text{the dot product of the feature vectors of } \mathbf x \text{ and } \mathbf x',
$$

called a **kernel**, and for many useful feature maps the kernel can be computed directly from
$\mathbf x$ and $\mathbf x'$ at trivial cost. In two dimensions, for example,
$(\mathbf x^{\mathsf T}\mathbf x')^2 = x_1^2x_1'^2 + 2x_1x_2x_1'x_2' + x_2^2x_2'^2$ is the dot product of the
feature vectors $(x_1^2, \sqrt2x_1x_2, x_2^2)$: one multiplication and a square replace the
explicit computation of three quadratic features. Replacing every dot product by $k$ gives the
kernel SVM, with $f(\mathbf x) = \sum_i\alpha_iy_ik(\mathbf x_i, \mathbf x) + b$. Any function that is a
**positive-definite kernel** corresponds to a dot product in some feature space and can be used:

- **Linear** $\mathbf x^{\mathsf T}\mathbf x'$: the original machine.
- **Polynomial** $(\gamma\mathbf x^{\mathsf T}\mathbf x' + c_0)^d$: the feature space of all monomials up to degree $d$.
- **Radial basis function (RBF)** $e^{-\gamma\lVert\mathbf x - \mathbf x'\rVert^2}$: a feature space of infinite dimension. The
  classifier $f$ becomes a sum of Gaussian bumps centred on the support vectors.
- **Sigmoid** $\tanh(\gamma\mathbf x^{\mathsf T}\mathbf x' + c_0)$: not positive definite for all parameters.

### The kernel width

The RBF parameter $\gamma$ sets the reach of each support vector, $1/\sqrt\gamma$:

| | | |
|:---:|:---:|:---:|
| ![γ = 0.3](../generated/machine_learning/svm_rbf_0.3.svg) | ![γ = 3](../generated/machine_learning/svm_rbf_3.svg) | ![γ = 100](../generated/machine_learning/svm_rbf_100.svg) |

A small $\gamma$ gives a smooth, nearly linear boundary; $\gamma = 3$ follows the moons; $\gamma = 100$ draws an island around nearly every point and makes
almost all of them support vectors. $C$ and $\gamma$ interact and must be chosen together, usually by cross-validation over a grid:

<p class="caption" id="tab-11-7"><strong>Table 11.7.</strong> Columns: γ = 0.1, γ = 1, γ = 10, γ = 100.</p>

{{#include ../generated/machine_learning/svm_grid.md}}

### Sequential minimal optimisation

The dual is a quadratic program with $n$ variables, one per training point. Each variable is
confined to the interval $[0, C]$ (a "box"), and there is one equality constraint. A
general-purpose solver would need the $n \times n$ matrix of kernel values, which is too large
for big data sets. **Sequential minimal optimisation** (SMO) exploits the simple structure of
the constraints.

It would be natural to optimise one multiplier at a time, as in
[coordinate descent](linear-regression.md#coordinate-descent), but the constraint
$\sum\alpha_iy_i = 0$ forbids it: with all the others fixed, a single multiplier cannot move. The
smallest set that can move is **two**. With two multipliers free, the equality constraint
confines them to a line segment inside the box, and the objective along that segment is a
quadratic in one variable. Its minimum is given by a formula, clipped to the ends of the segment.
SMO repeats this two-variable step until the optimality conditions hold.

The choice of the pair decides the speed. Following LIBSVM, the library chooses the first
variable as the one that most violates the optimality conditions, and the second to maximise
the guaranteed decrease of the objective (second-order selection). It keeps the gradient
$Q\boldsymbol\alpha - \mathbf e$ of the objective up to date at a cost of $O(n)$ per step, and stops when
the largest violation falls below the tolerance.

More than two classes are handled **one versus one**: a separate machine is trained for each
pair of classes, and a point is assigned to the class that wins the most pairwise contests.

### Support vector regression

**ε-insensitive regression** ignores errors smaller than $\varepsilon$ and penalises larger ones linearly, $\max(0, |y - f(\mathbf x)| - \varepsilon)$: the fit lies inside a tube
of half-width $\varepsilon$ where possible, and only points on or outside the tube are support vectors. The library solves the dual
$\min\tfrac12\boldsymbol\beta^{\mathsf T}(K + 1)\boldsymbol\beta - \mathbf y^{\mathsf T}\boldsymbol\beta + \varepsilon\lVert\boldsymbol\beta\rVert_1$ subject to $|\beta_i| \le C$ by coordinate descent, each coordinate having a closed-form soft-threshold-and-clip
update. Adding 1 to the kernel folds the intercept into the kernel (penalising it slightly) and removes the equality constraint.

![Support vector regression](../generated/machine_learning/svr.svg)

<p class="caption" id="fig-11-10"><strong>Figure 11.10.</strong> Support vector regression.</p>

## Algorithm

**Sequential minimal optimisation** for two classes. Write the dual objective as
$\tfrac12\boldsymbol\alpha^{\mathsf T}Q\boldsymbol\alpha - \mathbf e^{\mathsf T}\boldsymbol\alpha$ with $Q_{ij} = y_iy_jK_{ij}$, and let
$G = Q\boldsymbol\alpha - \mathbf e$ be its gradient.

1. **Initialise** $\boldsymbol\alpha = \mathbf 0$, so that $G = -\mathbf e$.
2. **Select the first variable.** Let $I_{\text{up}}$ be the indices whose $\alpha$ can still move
   in the direction that increases $y_t\alpha_t$, and $I_{\text{low}}$ those that can move the other
   way. Choose $i$ in $I_{\text{up}}$ maximising $-y_tG_t$: the variable that most violates the
   optimality conditions.
3. **Select the second variable.** Among $t$ in $I_{\text{low}}$ with
   $b_{it} = -y_iG_i + y_tG_t > 0$, choose $j$ minimising $-b_{it}^2/a_{it}$, where
   $a_{it} = K_{ii} + K_{tt} - 2K_{it}$. That quantity is the decrease in the objective that the
   pair $(i, t)$ would achieve, so this picks the most profitable partner.
4. **Test.** The gap between the largest value of $-y_tG_t$ over $I_{\text{up}}$ and the smallest
   over $I_{\text{low}}$ measures how far the solution is from optimal. If it is below the tolerance
   $\varepsilon$, stop.
5. **Update the pair.** Move $\alpha_i$ and $\alpha_j$ to the minimum of the objective along the
   line allowed by the equality constraint, and clip the result so that both stay in $[0, C]$.
6. **Update the gradient** for the change in the two multipliers, which needs columns $i$ and
   $j$ of the kernel matrix: $O(n)$ operations. Go to 2.

When the loop ends, the bias $b$ is the average of $y_tG_t$ over the **free** support vectors
(those with $0 < \alpha_t < C$), which lie exactly on the margin.

**Prediction**: evaluate $f(\mathbf x) = \sum_i\alpha_iy_ik(\mathbf x_i, \mathbf x) + b$ over the support
vectors only, and return its sign.

**More than two classes**: train one machine for every pair of classes, on the points of those
two classes; to predict, let every machine vote and return the class with the most votes.

## Implementation

```rust,ignore
{{#include ../../../crates/machine_learning/src/supervised/classification/support_vector_machine.rs:smo}}
```

```rust,ignore
{{#include ../../../crates/machine_learning/src/supervised/regression/support_vector_regression.rs:svr_dual}}
```

- The full kernel matrix is computed for each binary problem, $O(n^2)$ memory: practical up to a few thousand points.
- `Kernel::rbf_scaled` sets $\gamma = 1/(p \operatorname{var}(X))$, scikit-learn's `"scale"` heuristic.
- `decision_function` (two classes only) is positive for class 1.

## In practice

```rust,ignore
{{#include ../../../crates/machine_learning/examples/ml_support_vector_machines.rs}}
```

```text
linear                 accuracy 0.537 with 200 support vectors
polynomial, degree 2   accuracy 0.997 with 49 support vectors
RBF, γ = 2             accuracy 0.995 with 43 support vectors
SVR: R² 0.983 with 50 of 150 points as support vectors
```

No line separates concentric circles; a degree-2 polynomial kernel, whose feature space contains $x_1^2 + x_2^2$, separates them almost
perfectly.

**Strengths**

- Flexible nonlinear boundaries through kernels, with a convex training problem and a sparse solution.
- Effective in high dimensions and on small data sets.

**Weaknesses**

- $O(n^2)$ memory and up to $O(n^3)$ time: unsuitable for very large data sets.
- No probabilities; $C$ and the kernel parameters need tuning; features must be scaled.

## Exercises

1. Show that the margin of the canonical hyperplane is $2/\lVert\mathbf w\rVert$.
2. Derive the dual of the soft-margin problem with Lagrange multipliers, and show that $\alpha_i = C$ for points strictly inside the margin.
3. Show that $(\mathbf x^{\mathsf T}\mathbf x' + 1)^2$ is the inner product of explicit feature vectors in $\mathbb R^6$ for $\mathbf x \in \mathbb R^2$.
4. Show that the RBF kernel matrix is positive definite for distinct points.
5. Derive the two-variable SMO update for $y_i \ne y_j$ and its clipping bounds.

<details>
<summary>Hints and answers</summary>

1. The hyperplanes $\mathbf w^{\mathsf T}\mathbf x + b = \pm1$ are the margin boundaries; the distance between them is $2/\lVert\mathbf w\rVert$.
2. The Lagrangian gives $\mathbf w = \sum\alpha_iy_i\mathbf x_i$, $\sum\alpha_iy_i = 0$ and $0 \le \alpha_i \le C$. Complementarity then implies $\alpha_i = C$ for points with margin violation (strictly inside the margin or misclassified).
3. $(\mathbf x^{\mathsf T}\mathbf x' + 1)^2 = \varphi(\mathbf x)^{\mathsf T}\varphi(\mathbf x')$ with $\varphi(\mathbf x) = (1, \sqrt2x_1, \sqrt2x_2, x_1^2, x_2^2, \sqrt2x_1x_2)$.
4. Bochner's theorem: the Gaussian kernel is the Fourier transform of a positive measure with full support, so for distinct points $\sum c_ic_jk(\mathbf x_i, \mathbf x_j) = \int|\sum c_ie^{i\boldsymbol\omega\cdot\mathbf x_i}|^2\mathrm d\mu > 0$ unless all $c_i = 0$.
5. Fix all multipliers but $\alpha_i, \alpha_j$. The constraint $\sum\alpha_ky_k = 0$ keeps $\alpha_i - \alpha_j$ constant when $y_i \ne y_j$; optimise the resulting quadratic in one variable, and clip to $L = \max(0, \alpha_j - \alpha_i)$, $H = \min(C, C + \alpha_j - \alpha_i)$.

</details>

## Further reading

- C. Cortes and V. Vapnik, "Support-vector networks", *Machine Learning* 20 (1995) 273–297.
- J. C. Platt, "Sequential minimal optimization: a fast algorithm for training support vector machines", Microsoft Research
  technical report MSR-TR-98-14, 1998.
- R.-E. Fan, P.-H. Chen and C.-J. Lin, "Working set selection using second order information for training support vector machines",
  *Journal of Machine Learning Research* 6 (2005) 1889–1918.
- C.-C. Chang and C.-J. Lin, "LIBSVM: a library for support vector machines", *ACM Transactions on Intelligent Systems and Technology*
  2 (2011) 27:1–27:27.
- B. Schölkopf and A. J. Smola, *Learning with Kernels*, MIT Press, 2002.
