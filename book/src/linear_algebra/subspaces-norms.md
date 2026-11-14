# Matrix functions, subspaces and norms

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Compute the principal square root, logarithm and fractional powers of a matrix; the pseudoinverse, numerical rank and orthonormal bases of the range and null space; matrix norms and condition numbers |
| **Devised by** | Eliakim Moore (1920) and Roger Penrose (1955), the pseudoinverse; Åke Björck and Sven Hammarling (1983), the Schur square root; Charles Kenney and Alan Laub (1989), inverse scaling and squaring; Sheung Hun Cheng, Nicholas Higham, Kenney and Laub (2001), its Padé form; Gene Golub and William Kahan (1965), numerical rank from the SVD |
| **Accuracy** | Agrees with SciPy and NumPy to $5 \times 10^{-15}$ relative on well-conditioned matrices |
| **Cost** | `sqrtm`: a Schur form and $n^3/3$ operations; `logm`: a Schur form, $k$ triangular square roots and 8 triangular solves; the SVD-based functions: one SVD |
| **Read first** | [The singular value decomposition](svd.md), [Matrix functions and matrix equations](matrix-equations.md) |
| **API** | [`sqrtm`](../../api/lib_jsl_linalg/fn.sqrtm.html), [`logm`](../../api/lib_jsl_linalg/fn.logm.html), [`fractional_matrix_power`](../../api/lib_jsl_linalg/fn.fractional_matrix_power.html), [`pinv`](../../api/lib_jsl_linalg/fn.pinv.html), [`matrix_rank`](../../api/lib_jsl_linalg/fn.matrix_rank.html), [`null_space`](../../api/lib_jsl_linalg/fn.null_space.html), [`orth`](../../api/lib_jsl_linalg/fn.orth.html), [`norm`](../../api/lib_jsl_linalg/fn.norm.html), [`cond`](../../api/lib_jsl_linalg/fn.cond.html) |

</div>

## Overview

Two groups of tools complete the dense linear algebra:

- **Matrix functions beyond the exponential.** The square root $A^{1/2}$ appears in covariance factorisations and in symmetric orthogonalisation, the
  logarithm in recovering a generator $\log A$ from a transition matrix $A = e^{Bt}$, and fractional powers in interpolating between transformations.
- **The four fundamental subspaces and the size of a matrix.** The pseudoinverse solves least-squares problems of any rank. The rank and the
  bases of the range and null space describe what a matrix can and cannot reach. Norms and condition numbers measure its size and how much it
  can amplify errors.

The functions follow SciPy (`sqrtm`, `logm`, `fractional_matrix_power`, `pinv`, `null_space`, `orth`) and NumPy (`matrix_rank`, `norm`, `cond`).

## History

Eliakim Moore (1920) defined a generalised inverse for singular matrices, and Roger Penrose (1955) characterised it by four equations. Gene Golub and
William Kahan's SVD algorithm (1965) made it computable, and made the numerical rank precise: the number of singular values above a
tolerance. Åke Björck and Sven Hammarling (1983) computed matrix square roots from the Schur form, and Nicholas Higham (1987) adapted their method to real arithmetic.
For the logarithm, Charles Kenney and Alan Laub (1989) proposed **inverse scaling and squaring**: take square roots until the matrix is
close to the identity, then use a series. Sheung Hun Cheng, Higham, Kenney and Laub (2001) chose the Padé approximants for it, and
Awad Al-Mohy and Higham (2012) refined the choice of the number of square roots.

## The idea

### Square roots and logarithms through the Schur form

Every square matrix has a complex Schur form $A = ZTZ^{\mathsf H}$ with $T$ upper triangular and $Z$ unitary, and every primary matrix function satisfies $f(A) = Zf(T)Z^{\mathsf H}$. Functions of
triangular matrices are triangular, so they can be computed entry by entry.

For the **square root**, $R^2 = T$ with $R$ upper triangular gives $r_{ii} = \sqrt{t_{ii}}$, the principal root with positive real part, and for $i < j$

$$
r_{ij} = \frac{t_{ij} - \sum_{i<k<j} r_{ik}r_{kj}}{r_{ii} + r_{jj}} ,
$$

computed column by column. The denominators are non-zero as long as no two eigenvalues have square roots summing to zero, which the
principal branch guarantees unless $A$ has a repeated zero eigenvalue.

For the **logarithm**, power series converge only near the identity. **Inverse scaling and squaring** uses $\log A = 2^k\log A^{1/2^k}$. Take $k$ square roots of $T$ until
$\|T^{1/2^k} - I\|_1 \le 0.25$, then evaluate $\log(I + X)$ by the $[8/8]$ Padé approximant. Written as 8-point Gauss-Legendre quadrature of $\log(I + X) = \int_0^1 X(I + tX)^{-1}\,dt$, it is

$$
\log(I + X) \approx \sum_{j=1}^8 w_j\,X(I + t_jX)^{-1}.
$$

The approximation error for $\|X\| \le 0.25$ is below $10^{-16}$. **Fractional powers** follow as $A^p = \exp(p\log A)$.

### The pseudoinverse and the fundamental subspaces

With the SVD $A = U\Sigma V^{\mathsf H}$ and $r$ non-zero singular values:

- the **range** of $A$ is spanned by the first $r$ columns of $U$ (`orth`);
- the **null space** by the last $n - r$ columns of $V$ (`null_space`);
- the **pseudoinverse** is $A^+ = V\Sigma^+U^{\mathsf H}$, with $\Sigma^+$ inverting the non-zero singular values.

$A^+\mathbf b$ is the least-squares solution of $A\mathbf x = \mathbf b$ of smallest norm. It satisfies Penrose's four equations: $AA^+A = A$, $A^+AA^+ = A^+$, and both $AA^+$ and $A^+A$ are Hermitian.

In floating point no singular value of a rank-deficient matrix is exactly zero, so "non-zero" means "above a tolerance", $\sigma_i > \text{rtol} \cdot \sigma_1$, with
$\text{rtol} = \max(m, n)\varepsilon$ by default. The choice matters when the data are noisy. A rank-3 matrix perturbed by noise of size $10^{-8}$ has seven singular values near
$10^{-8}$, and inverting them amplifies the noise by $10^8$:

<p class="caption" id="tab-7-12"><strong>Table 7.12.</strong> The pseudoinverse solution for several truncation tolerances.</p>

{{#include ../generated/linear_algebra/pinv_truncation.md}}

With the tolerance above the noise, the pseudoinverse recovers the minimum-norm solution of the noise-free problem. Below it, the
solution is ruined. Choosing the tolerance from the known accuracy of the data is a form of regularisation.

### Norms and condition numbers

The norms of a matrix measure how much it can stretch a vector:

| norm | value | from |
|---|---|---|
| $\|A\|_1$ | largest column sum of $|a_{ij}|$ | the entries |
| $\|A\|_\infty$ | largest row sum | the entries |
| $\|A\|_2$ | $\sigma_1$, the largest singular value | the SVD |
| $\|A\|_F$ | $\sqrt{\sum|a_{ij}|^2} = \sqrt{\sum\sigma_i^2}$ | the entries |
| nuclear | $\sum\sigma_i$ | the SVD |

The **condition number** $\kappa(A) = \|A\|\|A^{-1}\|$ bounds the relative change of the solution of $A\mathbf x = \mathbf b$ per relative change of the data. In the 2-norm it is $\sigma_1/\sigma_n$, which also
applies to rectangular matrices and least squares. Roughly, $\log_{10}\kappa$ digits are lost when solving with $A$, as the
[chapter introduction](index.md#stability-and-conditioning) explains.

## Algorithm

**`logm(A)`**: Schur form $A = ZTZ^{\mathsf H}$; while $\|T - I\|_1 > 0.25$, set $T \leftarrow T^{1/2}$ (triangular square root) and $k \leftarrow k + 1$; $X = T - I$; $L = 2^k\sum_j w_jX(I + t_jX)^{-1}$
with Gauss-Legendre nodes $t_j$ and weights $w_j$ on $[0, 1]$; return $ZLZ^{\mathsf H}$.

**`pinv(A, rtol)`**: the SVD; $r$ = the number of $\sigma_i > \text{rtol} \cdot \sigma_1$; $A^+ = \sum_{i \le r}\mathbf v_i\mathbf u_i^{\mathsf H}/\sigma_i$.

## Implementation

```rust,ignore
{{#include ../../../crates/linalg/src/logsqrt.rs:logm}}
```

```rust,ignore
{{#include ../../../crates/linalg/src/subspaces.rs:pinv}}
```

- The matrix functions return complex matrices, even for real input whose result is real. Take the real parts in that case.
- Defaults follow SciPy and NumPy: `pinv`, `null_space` and `orth` truncate at $\max(m, n)\varepsilon\,\sigma_1$; `matrix_rank` at $\sigma_1\max(m, n)\varepsilon$.
- `cond` in the 1, ∞ and Frobenius norms inverts the matrix, and is infinite for a singular one.

## In practice

Every function is compared with SciPy 1.17.1 and NumPy 2 (references from `book/generator/data/make_subspaces_refs.py`). The null
space and range are compared through their projectors $QQ^{\mathsf H}$, which do not depend on the choice of basis:

<p class="caption" id="tab-7-13"><strong>Table 7.13.</strong> Matrix functions, subspaces and norms against SciPy and NumPy.</p>

{{#include ../generated/linear_algebra/subspaces_scipy.md}}

**Strengths**

- One method, the Schur form, for every matrix function, real or complex. The square root and logarithm agree with SciPy to rounding.
- SVD-based subspaces and pseudoinverses that are reliable for rank-deficient and rectangular matrices, with an explicit truncation tolerance.

**Weaknesses**

- No `funm` for general functions and no real-arithmetic Schur method, so real results are computed in complex arithmetic.
- `logm` takes a fixed number of Padé terms. Al-Mohy and Higham's algorithm chooses the number of square roots and the degree adaptively, and is faster.
- The subspace functions cost a full SVD. For large sparse matrices, iterative methods are needed.

## Exercises

1. Show that $f(A) = Zf(T)Z^{\mathsf H}$ for any polynomial $f$, and hence for any function defined by a power series.
2. Derive the recurrence for $r_{ij}$ from $(R^2)_{ij} = t_{ij}$.
3. Show that $\log(I + X) = \int_0^1 X(I + tX)^{-1}dt$, and why Gauss-Legendre quadrature of it gives a Padé approximant.
4. Verify Penrose's four equations for $A^+ = V\Sigma^+U^{\mathsf H}$.
5. Show that $\|A\|_2 \le \|A\|_F \le \sqrt r\,\|A\|_2$ for a matrix of rank $r$.

<details>
<summary>Hints and answers</summary>

1. $A^k = ZT^kZ^{\mathsf H}$ because $Z^{\mathsf H}Z = I$, and sums and limits preserve the identity.
2. $(R^2)_{ij} = \sum_{i \le k \le j}r_{ik}r_{kj} = r_{ii}r_{ij} + r_{ij}r_{jj} + \sum_{i<k<j}r_{ik}r_{kj}$; solve for $r_{ij}$.
3. Differentiate $\log(I + tX)$ with respect to $t$; $X$ and $(I + tX)^{-1}$ commute. An $m$-point Gauss rule applied to $X(I + tX)^{-1}$ gives a rational function of degree $[m/m]$ that matches the series to order $2m$, which is the Padé approximant.
4. $\Sigma\Sigma^+\Sigma = \Sigma$ and $\Sigma^+\Sigma\Sigma^+ = \Sigma^+$, and $\Sigma\Sigma^+$ and $\Sigma^+\Sigma$ are diagonal and real. The unitary factors carry these over.
5. $\|A\|_F^2 = \sum_{i \le r}\sigma_i^2$ lies between $\sigma_1^2$ and $r\sigma_1^2$.

</details>

## Further reading

- N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008, chapters 6 (square root) and 11 (logarithm).
- S. H. Cheng, N. J. Higham, C. S. Kenney and A. J. Laub, "Approximating the logarithm of a matrix to specified accuracy", *SIAM Journal on Matrix Analysis and Applications* 22 (2001) 1112–1125.
- Å. Björck and S. Hammarling, "A Schur method for the square root of a matrix", *Linear Algebra and its Applications* 52/53 (1983) 127–140.
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013, sections 2.3 (norms), 5.5 (pseudoinverse and rank).
- R. Penrose, "A generalized inverse for matrices", *Proceedings of the Cambridge Philosophical Society* 51 (1955) 406–413.
