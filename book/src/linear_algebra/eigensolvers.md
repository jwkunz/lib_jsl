# Symmetric and general eigensolvers

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Compute all eigenvalues, and optionally eigenvectors, of a dense Hermitian (real symmetric) or general square matrix, real or complex |
| **Devised by** | Householder tridiagonalisation: Alston Householder (1958), Wallace Givens (1954, by rotations); the implicit QL algorithm: Hilary Bowdler, Roger Martin, Christian Reinsch and James Wilkinson (1968); Hessenberg reduction and the implicit double-shift QR step: John Francis (1961); balancing: Beresford Parlett and Christian Reinsch (1969) |
| **Accuracy** | Backward stable: the eigenvalues of $A + \delta A$ with $\|\delta A\| \approx \varepsilon\|A\|$. For Hermitian matrices this bounds the eigenvalue errors by $\varepsilon\|A\|$; for non-normal matrices the errors can be much larger |
| **Cost** | Hermitian: $\tfrac43n^3$ flops for eigenvalues, about $9n^3$ with vectors. General: about $10n^3$ for eigenvalues, $25n^3$ with vectors |
| **API** | [`Eigh`](../../api/lib_jsl_linalg/trait.Eigh.html), [`EigValsh`](../../api/lib_jsl_linalg/trait.EigValsh.html), [`Eig`](../../api/lib_jsl_linalg/trait.Eig.html), [`EigVals`](../../api/lib_jsl_linalg/trait.EigVals.html) |

</div>

## Overview

The QR algorithm of the [previous entry](eigen-iterations.md) is the core of every dense eigensolver, but
applied to a full matrix it costs $O(n^3)$ per step. Production codes add three ingredients. A one-time
**reduction** by orthogonal similarity to tridiagonal form (Hermitian) or upper Hessenberg form (general)
makes each QR step $O(n)$ or $O(n^2)$, because the QR algorithm preserves these forms. **Implicit** shifted steps
apply the shift without ever forming $A - \mu I$, and for real matrices use complex conjugate pairs of shifts while
staying in real arithmetic. **Balancing** rescales a general matrix to reduce its norm before iterating.

Symmetric and general problems behave very differently. Hermitian matrices have real eigenvalues and an
orthonormal basis of eigenvectors, and their eigenvalues are perfectly conditioned: a perturbation of size
$\delta$ moves them by at most $\delta$. General matrices can have complex eigenvalues, may lack a full set of eigenvectors,
and their eigenvalues can be extraordinarily sensitive to perturbations.

## History

Wallace Givens (1954) reduced symmetric matrices to tridiagonal form by rotations, and Alston Householder
(1958) by reflections, at half the cost. The tridiagonal QL algorithm, the QR algorithm run from the bottom
up, was published with ALGOL procedures by Hilary Bowdler, Roger Martin, Christian Reinsch and James Wilkinson
in 1968, in the series that became the *Handbook for Automatic Computation* (1971) and then the EISPACK library.
John Francis's 1961 papers already contained the Hessenberg reduction and the implicit double-shift step for
real matrices. Beresford Parlett and Christian Reinsch (1969) introduced balancing. LAPACK (1992) blocked the
reductions; Karen Braman, Ralph Byers and Roy Mathias (2002) introduced the small-bulge multishift QR algorithm
with aggressive early deflation, which makes LAPACK much faster on large general matrices.

## The idea

### Hermitian matrices: tridiagonalisation and QL

Householder reflections applied from both sides, $Q^{\mathsf H}AQ = T$, zero everything outside the three central
diagonals, at a cost of $\tfrac43n^3$ flops. For a complex Hermitian matrix, a diagonal unitary scaling then makes the
off-diagonal of $T$ real and non-negative, so the rest of the work is in real arithmetic. The **implicit QL algorithm**
diagonalises the real symmetric tridiagonal matrix by Givens rotations, chasing each shifted step down (or up) the
diagonal at a cost of $O(n)$ per step, with about two steps per eigenvalue. The eigenvectors are $Q$ (accumulated in blocks,
in the compact WY form) times the product of the rotations, formed with the matrix-multiply kernel.

### General matrices: Hessenberg reduction and Francis's algorithm

A general matrix cannot be reduced to tridiagonal form by similarity, but it can be reduced to **upper Hessenberg**
form, zero below the first subdiagonal, which the QR algorithm also preserves. For a real matrix with complex conjugate
eigenvalues, a single real shift cannot converge to a complex eigenvalue, and complex shifts would force complex
arithmetic. **Francis's double-shift step** uses the pair of eigenvalues of the trailing $2 \times 2$ block as shifts together:
it computes the first column of $(H - \mu_1I)(H - \mu_2I)$, which is real, introduces a small "bulge" with a $3 \times 3$ reflection,
and chases it down the matrix to restore Hessenberg form. By the implicit Q theorem the result equals two explicit
shifted QR steps. The matrix converges to **real Schur form**, quasi-triangular with $1 \times 1$ blocks for real eigenvalues and
$2 \times 2$ blocks for complex pairs; the library then triangularises each block to obtain the complex Schur form
$A = ZTZ^{\mathsf H}$. Complex matrices use single shifts and Givens rotations. The eigenvectors of the triangular $T$ follow by back
substitution, and are mapped back through $Z$.

### Balancing

The rounding errors of the QR algorithm are proportional to $\|A\|$. **Balancing** replaces $A$ by $D^{-1}AD$ for a diagonal $D$
chosen so that each row and the corresponding column have similar norms. With the entries of $D$ powers of 2 the
eigenvalues are unchanged, not even rounded, and the norm can fall by many orders of magnitude. It matters most for
badly scaled matrices: without it, this library's solver returned 18 zeros and a wild complex pair for the companion
matrix of Wilkinson's polynomial ([polynomial roots](../root_finding/polynomial-roots.md#balancing)); it is
now always applied, as in LAPACK's `geev`.

### Accuracy and conditioning

Both solvers are backward stable:

{{#include ../generated/linear_algebra/eigen_accuracy.md}}

For Hermitian matrices, backward stability guarantees accurate eigenvalues, because eigenvalues of Hermitian matrices
move by no more than the perturbation (Weyl's theorem). For **non-normal** matrices there is no such guarantee: the
sensitivity of an eigenvalue is measured by the angle between its left and right eigenvectors, and can be enormous. The
Grcar matrix, a Toeplitz matrix with $-1$ on the subdiagonal and $1$ on the diagonal and three superdiagonals, is the classic
example:

![Eigenvalues of the Grcar matrix](../generated/linear_algebra/grcar.svg)

Perturbations of size $10^{-10}$ move its eigenvalues by up to $0.4$, and even the computed eigenvalues of the unperturbed matrix
are those of a matrix within rounding error of it. The eigenvalues of its symmetric part, on the other hand, do not move
visibly. For non-normal matrices the **pseudospectra**, the sets of eigenvalues of all nearby matrices, are often more
meaningful than the eigenvalues themselves.

### Speed

![Eigensolver speed](../generated/linear_algebra/eigen_speed.svg)

The reduction to tridiagonal form makes the production symmetric solver roughly two orders of magnitude faster than the
reference QR algorithm on the full matrix at $n = 128$. Against OpenBLAS's LAPACK, the symmetric solver is about as fast, and the general
solver is as fast for small matrices but 2.5 times slower at $n = 256$, for want of multishift QR with aggressive early
deflation ([issue #9](https://github.com/jwkunz/lib_jsl/issues/9)).

## Algorithm

**`eigh`**: tridiagonalise by Householder reflections; scale the off-diagonal to real; run implicit QL with Wilkinson-like
shifts, accumulating rotations if vectors are wanted; sort ascending; form the eigenvectors $Q D Z$.

**`eig`**: balance; reduce to Hessenberg form; run Francis double-shift QR (real) or single-shift QR (complex) to Schur form,
deflating negligible subdiagonal entries; triangularise $2 \times 2$ blocks; back-substitute for the eigenvectors of $T$; map back
through $Z$ and the balancing; normalise.

## Implementation

The implicit QL algorithm for the tridiagonal matrix (after Numerical Recipes' `tqli`):

```rust,ignore
{{#include ../../../crates/linalg/src/eigh.rs:tql}}
```

Balancing:

```rust,ignore
{{#include ../../../crates/linalg/src/eig.rs:balance}}
```

- `eigh(uplo)` reads one triangle and returns ascending eigenvalues and orthonormal eigenvectors; `eigvalsh` skips the vectors.
- `eig` returns complex eigenvalues (in Schur order) and unit-norm complex eigenvectors for real or complex input; `eigvals` skips
  the vectors.

## In practice

```rust,ignore
{{#include ../../../crates/linalg_reference/examples/linalg_eigen.rs}}
```

**Strengths**

- Complete, backward-stable dense eigensolvers for real and complex, Hermitian and general matrices.
- The Hermitian solver matches LAPACK's speed.

**Weaknesses**

- The general solver lacks multishift QR and aggressive early deflation, and falls behind LAPACK for large matrices.
- No generalized eigenproblem ($A\mathbf v = \lambda B\mathbf v$; for symmetric positive definite $B$, reduce with the Cholesky factor of $B$), and no
  iterative methods for a few eigenvalues of large sparse matrices.
- Eigenvalues of non-normal matrices may be inaccurate, as for any method.

## Exercises

1. Show that the QR algorithm preserves upper Hessenberg form and symmetric tridiagonal form.
2. Count the flops of Householder tridiagonalisation.
3. Compute the eigenvalues of the companion matrix of $\prod_{k=1}^{20}(x - k)$ with and without balancing, and explain the difference.
4. Compute the condition numbers $1/|\mathbf y^{\mathsf H}\mathbf x|$ (with unit left and right eigenvectors) of the eigenvalues of the Grcar matrix.
5. Solve the generalized problem $K\mathbf v = \omega^2M\mathbf v$ for a mass-spring system with non-uniform masses by transforming with $M^{-1/2}$.

## Further reading

- J. H. Wilkinson, *The Algebraic Eigenvalue Problem*, Oxford University Press, 1965.
- J. H. Wilkinson and C. Reinsch, *Handbook for Automatic Computation, Volume II: Linear Algebra*, Springer, 1971.
- D. S. Watkins, *The Matrix Eigenvalue Problem: GR and Krylov Subspace Methods*, SIAM, 2007.
- K. Braman, R. Byers and R. Mathias, "The multishift QR algorithm. Part II: aggressive early deflation", *SIAM Journal on
  Matrix Analysis and Applications* 23 (2002) 948–973.
- L. N. Trefethen and M. Embree, *Spectra and Pseudospectra*, Princeton University Press, 2005.
