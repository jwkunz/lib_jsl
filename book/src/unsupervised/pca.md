# Principal component analysis and kernel PCA

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Find the orthogonal directions of greatest variance and project data onto the first few; extend this to nonlinear structure through a kernel |
| **Devised by** | Karl Pearson (1901), lines of closest fit; Harold Hotelling (1933), principal components; Kari Karhunen (1946) and Michel Loève (1955), the continuous expansion; Bernhard Schölkopf, Alexander Smola and Klaus-Robert Müller (1998), kernel PCA |
| **Accuracy** | Exact up to the eigensolver; the variances agree with NumPy's SVD to $10^{-13}$ (table below) |
| **Cost** | PCA: $O(np^2 + p^3)$ via the covariance matrix. Kernel PCA: $O(n^2p + n^3)$ via the Gram matrix |
| **API** | [`Pca`](../../api/lib_jsl_machine_learning/unsupervised/reduction/principal_component_analysis/struct.Pca.html) (`new(components)`, `components`, `explained_variance`, `explained_variance_ratio`, `inverse_transform`), [`KernelPca`](../../api/lib_jsl_machine_learning/unsupervised/reduction/kernel_pca/struct.KernelPca.html) (`new(kernel, components)`, `eigenvalues`); both implement [`Transformer`](../../api/lib_jsl_machine_learning/common/trait.Transformer.html) |

</div>

## Overview

Measured features are often redundant: height and weight, or the prices of related stocks, move together. **Principal component
analysis** (PCA) rotates the data to new axes, the **principal components**, ordered so that the first captures as much variance as
possible, the second as much of the rest, and so on. Keeping the first few gives a lower-dimensional summary that loses the least
variance, for visualisation, compression, denoising or as input to another model. **Kernel PCA** does the same in the feature space of
a kernel, where curved structure in the original space can become linear.

## History

Karl Pearson (1901) sought the line and plane of closest fit to a cloud of points. Harold Hotelling (1933), analysing
psychological test scores, introduced the name principal components and the eigenvector formulation. The same expansion for random processes
is the Karhunen-Loève transform (Kari Karhunen, 1946; Michel Loève), and in other fields PCA is known as empirical orthogonal
functions (meteorology) or proper orthogonal decomposition (fluid mechanics). Bernhard Schölkopf, Alexander Smola and
Klaus-Robert Müller (1998) expressed PCA through inner products only, so that a kernel could replace them.

## The idea

### Directions of greatest variance

PCA looks for the directions along which the data vary most. The assumption behind it is that
directions of large variation carry the structure of the data, and directions of small variation
are mostly noise.

**Setting up.** Subtract the mean from every point, $\widetilde{\mathbf x}_i = \mathbf x_i - \bar{\mathbf x}$, so
that the cloud of points is centred on the origin, and form the sample **covariance matrix**

$$
C = \frac{1}{n - 1}\sum_i\widetilde{\mathbf x}_i\widetilde{\mathbf x}_i^{\mathsf T} .
$$

Its diagonal entries are the variances of the individual features, and its off-diagonal entries
say how pairs of features vary together.

**The first direction.** Project the points onto a unit vector $\mathbf u$: the coordinate of point
$i$ along it is $\mathbf u^{\mathsf T}\widetilde{\mathbf x}_i$. The variance of these coordinates is

$$
\frac{1}{n - 1}\sum_i(\mathbf u^{\mathsf T}\widetilde{\mathbf x}_i)^2
= \mathbf u^{\mathsf T}\Big(\frac{1}{n - 1}\sum_i\widetilde{\mathbf x}_i\widetilde{\mathbf x}_i^{\mathsf T}\Big)\mathbf u = \mathbf u^{\mathsf T}C\mathbf u .
$$

We want the unit vector that maximises $\mathbf u^{\mathsf T}C\mathbf u$. Since $C$ is symmetric it has
orthonormal eigenvectors $\mathbf u_1, \mathbf u_2, \ldots$ with eigenvalues
$\lambda_1 \ge \lambda_2 \ge \cdots \ge 0$. Writing $\mathbf u = \sum_ja_j\mathbf u_j$ with $\sum_ja_j^2 = 1$
gives $\mathbf u^{\mathsf T}C\mathbf u = \sum_j\lambda_ja_j^2$, a weighted average of the eigenvalues. It is
largest when all the weight is on the largest eigenvalue: $\mathbf u = \mathbf u_1$, and the variance
along it is $\lambda_1$. (The expression $\mathbf u^{\mathsf T}C\mathbf u$ for unit $\mathbf u$ is the
[Rayleigh quotient](../linear_algebra/eigen-iterations.md#power-iteration-inverse-iteration-and-the-rayleigh-quotient)
of $C$.)

**The other directions.** The second direction maximises the variance among unit vectors
orthogonal to the first. By the same argument it is $\mathbf u_2$, with variance $\lambda_2$, and so
on. The **principal axes** are therefore the eigenvectors of the covariance matrix, ordered by
eigenvalue:

$$
C = U\Lambda U^{\mathsf T}, \qquad \text{scores } \mathbf z_i = U_q^{\mathsf T}\widetilde{\mathbf x}_i,
$$

where $U_q$ holds the first $q$ eigenvectors as columns. The **scores** $\mathbf z_i$ are the
coordinates of the points along the first $q$ axes, a description of each point by $q$ numbers
instead of $p$. The variance of the $j$-th score is $\lambda_j$, and different scores are
uncorrelated, because in the eigenvector basis the covariance matrix is diagonal.

![Principal axes](../generated/unsupervised/pca_axes.svg)

<p class="caption" id="fig-12-7"><strong>Figure 12.7.</strong> Principal axes.</p>

### Best low-rank reconstruction

The same axes solve a second problem, which gives another reason to use them. Suppose we keep
only the $q$ scores of each point and want to recover the point as well as possible. The
reconstruction from the scores is

$$
\hat{\mathbf x}_i = \bar{\mathbf x} + U_q\mathbf z_i,
$$

the projection of the point onto the $q$-dimensional subspace spanned by the axes (shifted back
by the mean). Among *all* $q$-dimensional subspaces, this one minimises the sum of squared
distances from the points to their projections. This is the
[Eckart-Young theorem](../linear_algebra/svd.md#low-rank-approximation) applied to the matrix of
centred data. Maximising the variance kept and minimising the error discarded are the same
thing, because by Pythagoras the squared length of each centred point is the sum of the two.

The error is exactly the variance in the directions thrown away: the mean squared reconstruction
error equals the sum of the discarded eigenvalues (times $(n - 1)/n$, and divided by $p$ for a
per-feature mean):

<p class="caption" id="tab-12-11"><strong>Table 12.11.</strong> Columns: components kept, variance explained, reconstruction mean squared error, discarded eigenvalues ÷ features.</p>

{{#include ../generated/unsupervised/pca_reconstruction.md}}

The last two columns agree to every printed digit.

### How many components?

The **explained variance ratio** $\lambda_j/\sum_k\lambda_k$ shows how much each component matters. For data generated from three hidden factors in 12
dimensions plus noise:

![Scree plot](../generated/unsupervised/pca_scree.svg)

<p class="caption" id="fig-12-8"><strong>Figure 12.8.</strong> Scree plot.</p>

The first three components carry 98.8% of the variance, and the remaining nine share the noise equally. A **scree plot** like this,
or a target fraction such as 95%, is the usual way to choose $q$. PCA depends on the units: a feature measured in millimetres
dominates one in metres, so features on different scales should be standardised first, which amounts to PCA of the correlation matrix.

### Testing against NumPy

On 200 points in five dimensions driven by two factors, the library's variances agree with those from NumPy's SVD of the centred
data to $8 \times 10^{-14}$ relative, and the two leading axes (up to sign) to $7 \times 10^{-16}$; see the [reference table](hierarchical.md#checking-against-scipy). The trailing axes span
nearly equal noise variances and are only defined up to rotation, so they are not compared.

### Kernel PCA

PCA only finds linear structure. Two concentric rings have no direction of high variance that separates them. Kernel PCA maps
each point to features $\varphi(\mathbf x)$ and performs PCA there, using only inner products $k(\mathbf x, \mathbf x') = \varphi(\mathbf x)^{\mathsf T}\varphi(\mathbf x')$ ([kernels](../machine_learning/support-vector-machines.md#the-dual-problem-and-kernels)). Every principal axis is a combination of the
mapped training points, $\mathbf v = \sum_i\alpha_i\varphi(\mathbf x_i)$, and the coefficients solve an eigenproblem of the **centred Gram matrix**

$$
\widetilde K = K - \mathbf 1_nK - K\mathbf 1_n + \mathbf 1_nK\mathbf 1_n, \qquad \widetilde K\boldsymbol\alpha = \lambda\boldsymbol\alpha,
$$

where $\mathbf 1_n$ is the matrix with every entry $1/n$, which centres the features. The scores of a new point are $\sum_i\alpha_i\widetilde k(\mathbf x, \mathbf x_i)$, with $\boldsymbol\alpha$ scaled to $1/\sqrt\lambda$ so that the axis has
unit length in feature space.

| | |
|:---:|:---:|
| ![Linear PCA of rings](../generated/unsupervised/kpca_linear.svg) | ![RBF kernel PCA of rings](../generated/unsupervised/kpca_rbf.svg) |

Linear PCA only rotates the rings. With an RBF kernel, the first two components describe the angle around the rings (a degenerate
pair, like a sine and a cosine), and the third separates the inner ring from the outer one. The eigenvalues come in such pairs:

![Kernel PCA eigenvalues](../generated/unsupervised/kpca_eigenvalues.svg)

<p class="caption" id="fig-12-9"><strong>Figure 12.9.</strong> Kernel PCA eigenvalues.</p>

The centred Gram matrix always has the eigenvalue zero (the constant vector) and, for smooth kernels, many tiny ones.

## Algorithm

**PCA.**

1. **Centre.** Subtract from each column of the data matrix its mean, and store the means.
2. **Covariance.** Form $C = \widetilde X^{\mathsf T}\widetilde X/(n - 1)$.
3. **Eigendecomposition.** Compute the eigenvalues and eigenvectors of $C$ with the
   [symmetric eigensolver](../linear_algebra/eigensolvers.md).
4. **Select.** Keep the eigenvectors of the $q$ largest eigenvalues as the principal axes. An
   eigenvector is defined only up to sign, so each is given the sign that makes its largest
   entry positive, which makes the result reproducible.
5. **Transform** a point: subtract the stored means and take the dot product with each axis.
   **Reconstruct** it: multiply the scores by the axes and add the means back.

Forming $C$ costs $O(np^2)$ and its eigendecomposition $O(p^3)$.

**Kernel PCA.**

1. **Gram matrix.** Compute $K_{ij} = k(\mathbf x_i, \mathbf x_j)$ for all pairs of training points.
2. **Centre it.** Subtract from each entry its row mean and its column mean, and add the
   overall mean (the formula for $\widetilde K$ above). Store the row means and the overall
   mean.
3. **Eigendecomposition.** Compute the eigenvalues and eigenvectors of $\widetilde K$, and keep
   the $q$ largest. Scale each eigenvector by $1/\sqrt\lambda$.
4. **Transform a new point**: compute its kernel values with all the training points, centre
   that row with the stored means, and take its dot product with each scaled eigenvector.

The eigendecomposition of the $n \times n$ Gram matrix costs $O(n^3)$: kernel PCA scales with the
number of points, not the number of features.

## Implementation

```rust,ignore
{{#include ../../../crates/machine_learning/src/unsupervised/reduction/principal_component_analysis.rs:pca_fit}}
```

```rust,ignore
{{#include ../../../crates/machine_learning/src/unsupervised/reduction/kernel_pca.rs:kpca_fit}}
```

- PCA works through the $p \times p$ covariance matrix, which is efficient when $n \gg p$. For wide data ($p > n$), or for the best accuracy, the SVD of
  the centred data matrix is preferable: forming $C$ squares the condition number.
- Kernel PCA stores the training data and needs a cross-kernel row per new point. With the linear kernel its scores equal `Pca`'s up
  to sign: both are $\sqrt\lambda\,\mathbf v$ for the eigenpairs of $\widetilde X\widetilde X^{\mathsf T}$.

## In practice

```rust,ignore
{{#include ../../../crates/machine_learning/examples/ml_pca.rs}}
```

```text
explained variance ratio [0.6842, 0.3046, 0.0042, 0.0035, 0.0035]
two components: reconstruction RMS error 0.228 (noise level 0.3)
first axis [0.776, 0.247, -0.012, 0.519, -0.261]
kernel PCA eigenvalues [30.85, 30.23, 18.68]; component 3 means: outer 0.304, inner -0.304
```

Two components explain 99% of the variance. The reconstruction error is below the noise level, because the part of the noise
that lies in the two-dimensional subspace is kept.

**Strengths**

- Fast, exact and parameter-free; uncorrelated scores; the optimal linear compression.
- Kernel PCA finds nonlinear structure with the same linear algebra.

**Weaknesses**

- Linear PCA misses curved structure; the components need not be interpretable or relevant to a later prediction task.
- Sensitive to feature scales and outliers.
- Kernel PCA costs $O(n^3)$, keeps the training set, and has no simple inverse mapping back to the inputs.

## Exercises

1. Show that the first principal axis maximises $\mathbf u^{\mathsf T}C\mathbf u$ subject to $\lVert\mathbf u\rVert = 1$, using a Lagrange multiplier.
2. Show that the principal axes are the right singular vectors of the centred data matrix, and relate the singular values to the eigenvalues of $C$.
3. Prove that the reconstruction error with $q$ components is $\frac{n - 1}{n}\sum_{j > q}\lambda_j$ per point.
4. Show that kernel PCA with the linear kernel reproduces the PCA scores exactly, up to sign.
5. Derive the double-centring formula from $\widetilde\varphi(\mathbf x) = \varphi(\mathbf x) - \frac1n\sum_j\varphi(\mathbf x_j)$.

## Further reading

- I. T. Jolliffe, *Principal Component Analysis*, 2nd ed., Springer, 2002.
- H. Hotelling, "Analysis of a complex of statistical variables into principal components", *Journal of Educational Psychology* 24 (1933) 417–441, 498–520.
- B. Schölkopf, A. Smola and K.-R. Müller, "Nonlinear component analysis as a kernel eigenvalue problem", *Neural Computation* 10 (1998) 1299–1319.
- I. T. Jolliffe and J. Cadima, "Principal component analysis: a review and recent developments", *Philosophical Transactions of the Royal Society A* 374 (2016) 20150202.
