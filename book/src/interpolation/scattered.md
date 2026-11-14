# Scattered data interpolation

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Interpolate values $f_i$ given at arbitrary, irregularly placed points $\mathbf p_i$ in two or three dimensions |
| **Devised by** | Inverse distance weighting: Donald Shepard (1968). Radial basis functions: Rolland Hardy's multiquadrics (1971), Jean Duchon's thin-plate splines (1976); theory by Charles Micchelli (1986) |
| **Accuracy** | Shepard: low, with flat spots at the data points. Radial basis functions: high for smooth functions, depending on the kernel and the point spacing |
| **Cost** | Shepard: $O(n)$ per evaluation, no set-up. RBF: an $O(n^3)$ dense solve, then $O(n)$ per evaluation |
| **Read first** | [Polynomial interpolation](polynomial.md), [Gaussian elimination and LU factorisation](../linear_algebra/lu.md) |
| **API** | [`RadialBasisInterpolator`](../../api/lib_jsl_interpolation/two_d/radial_basis_functions/struct.RadialBasisInterpolator.html), [`RadialKernel`](../../api/lib_jsl_interpolation/two_d/radial_basis_functions/enum.RadialKernel.html), [`Shepards`](../../api/lib_jsl_interpolation/three_d/shepards/struct.Shepards.html) |

</div>

## Overview

Measurements are often taken where they can be: weather stations, boreholes, sensors on a structure,
samples of an expensive simulation. Such **scattered data** have no grid, so the tensor-product
methods of the [previous entry](grids.md) do not apply. Two families of methods build interpolants
directly from the points and their distances.

**Shepard's method** (inverse distance weighting) takes a weighted average of the data, with weights
that grow without bound near each data point. It is simple and never overshoots, but crude.
**Radial basis functions** write the interpolant as a sum of identical bumps $\varphi(\|\mathbf p - \mathbf p_i\|)$
centred at the data points, plus a low-degree polynomial, with coefficients chosen so that it passes
through the data. They are accurate, smooth, work in any dimension, and underlie methods in machine
learning, meshless PDE solvers and computer graphics.

## History

Donald Shepard, a student at Harvard's Laboratory for Computer Graphics, proposed inverse distance
weighting in 1968 for mapping geographic data, and it remains a default in geographic information
systems. Rolland Hardy (1971), a geodesist at Iowa State University, introduced **multiquadrics**,
$\sqrt{1 + (\varepsilon r)^2}$, to interpolate topographic surfaces. Jean Duchon (1976) derived the **thin-plate
spline** $r^2 \ln r$ as the surface that interpolates the data with minimum bending energy, the
two-dimensional analogue of the [natural cubic spline](splines.md#cubic-splines). Charles Micchelli
(1986) proved that the interpolation matrices of these kernels are invertible for any distinct points,
which established radial basis functions as a sound general method.

## The idea

### Shepard's inverse distance weighting

The data are values $f_i$ at points $\mathbf p_i$. A natural guess for the value at a new point
$\mathbf p$ is an average of the data in which nearby points count for more. Shepard weighted each
point by an inverse power of its distance:

$$
s(\mathbf p) = \frac{\sum_i w_i(\mathbf p)\, f_i}{\sum_i w_i(\mathbf p)}, \qquad w_i(\mathbf p) = \|\mathbf p - \mathbf p_i\|^{-\mu}.
$$

The exponent $\mu > 0$ is chosen by the user; $\mu = 2$ is common. Three properties follow directly
from the formula.

- **It interpolates.** As $\mathbf p \to \mathbf p_i$ the weight $w_i$ grows without bound and dominates
  both sums, so $s \to f_i$. (This is the same argument as for the
  [barycentric formula](rational.md#barycentric-rational-functions).)
- **It never overshoots.** The weights are positive, and dividing by their sum makes them add up
  to 1. So $s(\mathbf p)$ is a weighted average of the data and lies between the smallest and the
  largest $f_i$.
- **It is flat at the data points.** For $\mu > 1$ the gradient of $s$ vanishes at every data
  point. The surface has a little plateau at each point and bumps in between, which is a poor
  model of a smooth function: a smooth function generally has a nonzero slope at the places where
  it happens to have been measured.

Larger $\mu$ makes the weights fall off faster and so localises the influence of each point.

### Radial basis functions

Radial basis functions (RBFs) take a different approach. Choose a function $\varphi(r)$ of one
variable, the **kernel**, and centre a copy of it on every data point: the function
$\varphi(\|\mathbf p - \mathbf p_i\|)$ depends only on the distance from $\mathbf p_i$ (it is *radial*). The
interpolant is a weighted sum of these $n$ functions plus a plane, written here for two dimensions
with $\mathbf p = (x, y)$:

$$
s(\mathbf p) = \sum_{i=1}^n w_i\,\varphi\big(\|\mathbf p - \mathbf p_i\|\big) + a + b\,x + c\,y.
$$

Unlike Shepard's weights, the coefficients $w_i$ are not given by a formula. They are unknowns,
found by requiring that the interpolant pass through the data.

**Setting up the equations.** There are $n + 3$ unknowns: $w_1, \ldots, w_n$ and $a, b, c$. The
interpolation conditions $s(\mathbf p_j) = f_j$ give $n$ equations, each linear in the unknowns. The
remaining three are the **side conditions**

$$
\sum_i w_i = 0, \qquad \sum_i w_i x_i = 0, \qquad \sum_i w_i y_i = 0,
$$

which say that the weights carry no constant or linear trend; that part of the data is left to the
plane $a + bx + cy$. Together they form a symmetric linear system:

$$
\begin{pmatrix} \Phi & P \\ P^{\mathsf T} & 0 \end{pmatrix}
\begin{pmatrix} \mathbf w \\ \mathbf c \end{pmatrix} =
\begin{pmatrix} \mathbf f \\ \mathbf 0 \end{pmatrix},
\qquad \Phi_{ij} = \varphi\big(\|\mathbf p_i - \mathbf p_j\|\big), \quad P_i = (1, x_i, y_i).
$$

The top block row is the interpolation conditions: $\Phi\mathbf w$ is the sum of kernels evaluated at
the data points, and $P\mathbf c$ with $\mathbf c = (a, b, c)$ is the plane. The bottom block row is the
side conditions. The matrix is dense, so solving the system costs $O(n^3)$ operations, which limits
the method to a few thousand points.

**Why the plane?** It has two purposes. It makes the interpolant reproduce planes exactly with any
kernel: if the data lie on a plane, the solution is $\mathbf w = 0$. And for some kernels it is
needed for the system to have a solution at all. The **cubic** kernel $r^3$ and the
**thin-plate spline** $r^2 \ln r$ are only *conditionally* positive definite, which means the system
is guaranteed to be solvable only together with the side conditions.

**Choosing a kernel.** The thin-plate spline is the two-dimensional analogue of the
[natural cubic spline](splines.md#cubic-splines): it minimises the bending energy of a thin plate
pinned at the data. It has no parameter and is a robust default.

Other kernels contain a **shape parameter** $\varepsilon$ that sets their width: the Gaussian
$e^{-(\varepsilon r)^2}$, the multiquadric $\sqrt{1 + (\varepsilon r)^2}$ and the inverse multiquadric
$1/\sqrt{1 + (\varepsilon r)^2}$. They can be very accurate for smooth functions, but they pose a
dilemma. As $\varepsilon \to 0$ the kernels become wide and flat and the interpolant becomes more
accurate. But wide, flat kernels centred at different points look almost alike, so the rows of
$\Phi$ become almost equal and the matrix becomes catastrophically ill-conditioned: rounding errors
destroy the solution before the theoretical accuracy is reached. Choosing $\varepsilon$ is an art. A common rule is to make the
kernel width $1/\varepsilon$ comparable to the typical distance between points.

### Comparison

Franke's function, a standard test surface with two peaks and a dip, sampled at 100 Halton points in
the unit square:

<p class="caption" id="tab-3-11"><strong>Table 3.11.</strong> Accuracy of scattered-data interpolation methods.</p>

{{#include ../generated/interpolation/scattered_table.md}}

![A slice through the interpolants](../generated/interpolation/scattered_slice.svg)

<p class="caption" id="fig-3-12"><strong>Figure 3.12.</strong> A slice through the interpolants.</p>

All the radial basis functions reach errors of a few thousandths; Shepard's method is ten times worse
and visibly bumpy, with flat spots wherever the slice passes near a data point.

## Algorithm

**Shepard.** There is no set-up. For each evaluation point $\mathbf p$:

1. loop over the data points, computing the distance $r_i = \|\mathbf p - \mathbf p_i\|$;
2. if some $r_i$ is zero, return $f_i$ at once;
3. otherwise accumulate $\sum_i r_i^{-\mu}f_i$ and $\sum_i r_i^{-\mu}$, and return their ratio.

Cost: $O(n)$ per point.

**Radial basis functions.**

1. *Set-up.* Fill the $n \times n$ block $\Phi$ with the kernel evaluated at every pair of data
   points, and the block $P$ with the rows $(1, x_i, y_i)$. Solve the resulting
   $(n + 3) \times (n + 3)$ system by [LU factorisation](../linear_algebra/lu.md) for the weights
   $w_i$ and the coefficients $a, b, c$. Report an error if the matrix is singular. Cost:
   $O(n^3)$, once.
2. *Evaluation.* Sum the $n$ kernels centred on the data points, weighted by the $w_i$, and add
   the plane $a + bx + cy$. Cost: $O(n)$ per point.

## Implementation

```rust,ignore
{{#include ../../../crates/interpolation/src/two_d/radial_basis_functions.rs:rbf_system}}
```

- The RBF system is dense: fine for a few thousand points, too slow beyond. Large problems use compactly
  supported kernels (sparse systems) or fast multipole methods.
- Duplicate points make the RBF system singular; collinear points make the affine part undetermined.
  Both are reported as errors.
- Shepard's method is implemented in three dimensions; two-dimensional data can use a constant third
  coordinate, as in the examples.

## In practice

```rust,ignore
{{#include ../../../crates/interpolation/examples/interpolation_scattered.rs}}
```

**Strengths**

- Any point locations, any dimension, no mesh.
- RBFs are smooth and accurate; the thin-plate spline needs no parameter.
- Shepard's method is trivial to implement and never overshoots.

**Weaknesses**

- RBFs cost $O(n^3)$ to set up, and shape-parameter kernels can be ill-conditioned.
- Shepard's method is inaccurate, with flat spots at the data.
- Extrapolation far from the data is unreliable for every method.

## Exercises

1. Show that Shepard's interpolant with $\mu = 2$ has zero gradient at each data point.
2. Solve the RBF system for three points and show why the side conditions are needed for the
   thin-plate spline.
3. Vary $\varepsilon$ for the Gaussian kernel on Franke's function from 0.5 to 20. Plot the error and the
   condition number of $\Phi$. Where is the best $\varepsilon$?
4. Compare RBF interpolation on 100 random points with 100 Halton points. Which gives the smaller
   maximum error, and why?
5. Interpolate the rainfall data of the example with Shepard's method for $\mu = 1, 2, 8$ and describe how
   the surface changes.

<details>
<summary>Hints and answers</summary>

1. Near a data point $\mathbf x_i$ the weights are $|\mathbf x - \mathbf x_i|^{-\mu}$, so $u(\mathbf x) = f_i + O(|\mathbf x - \mathbf x_i|^{\mu})$. For $\mu > 1$ this has zero gradient at $\mathbf x_i$, giving flat spots at every data point.
2. For three points the TPS system is $3 + 3$ equations in $3 + 3$ unknowns; the kernel $r^2\ln r$ is only conditionally positive definite, so the polynomial part and the side conditions $\sum \lambda_i p(\mathbf x_i) = 0$ are needed to make the system nonsingular and to reproduce linear functions.
3. Small $\varepsilon$ (flat Gaussians) gives small error but a hopelessly ill-conditioned $\Phi$; large $\varepsilon$ gives a well-conditioned but inaccurate fit. The best $\varepsilon$ lies in between, at a condition number far above 1.
4. Halton points usually give the smaller maximum error because they have a smaller fill distance (no large gaps), which governs the interpolation error.
5. With $\mu = 1$ the surface has cusps at the data; $\mu = 2$ is smoother with flat spots; $\mu = 8$ approaches a piecewise-constant nearest-neighbour surface.

</details>

## Further reading

- D. Shepard, "A two-dimensional interpolation function for irregularly-spaced data", *Proceedings of
  the 1968 ACM National Conference*, 517–524.
- R. L. Hardy, "Multiquadric equations of topography and other irregular surfaces", *Journal of
  Geophysical Research* 76 (1971) 1905–1915.
- J. Duchon, "Interpolation des fonctions de deux variables suivant le principe de la flexion des plaques
  minces", *RAIRO Analyse Numérique* 10 (1976) 5–12.
- C. A. Micchelli, "Interpolation of scattered data: distance matrices and conditionally positive
  definite functions", *Constructive Approximation* 2 (1986) 11–22.
- G. E. Fasshauer, *Meshfree Approximation Methods with MATLAB*, World Scientific, 2007.
- R. Franke, "Scattered data interpolation: tests of some methods", *Mathematics of Computation* 38
  (1982) 181–200. The source of the test function.
