# Dense linear algebra

Linear algebra is the foundation of scientific computing. Nearly every method in this book reduces
to it in the end: [Newton's method](../root_finding/nonlinear-systems.md) solves a linear system at
each step, [least-squares fitting](../optimization/least-squares.md) is a sequence of linear
least-squares problems, [implicit ODE solvers](../integration/stiff-ode.md) and
[finite elements](../integration/finite-elements.md) solve linear systems, and
[Prony's method](../interpolation/trigonometric.md#pronys-method) and
[polynomial root finding](../root_finding/polynomial-roots.md) compute eigenvalues. This part
covers **dense** matrices, stored in full, of up to a few thousand rows.

## The central questions

- **Which problem?** A linear system $A\mathbf x = \mathbf b$; a least-squares problem $\min\|A\mathbf x - \mathbf b\|$; an eigenvalue
  problem $A\mathbf v = \lambda\mathbf v$; or the singular value decomposition $A = U\Sigma V^{\mathsf T}$.
- **What structure?** Symmetric (or Hermitian) positive definite matrices halve the work of a
  linear solve and need no pivoting; symmetric matrices have real eigenvalues and orthogonal
  eigenvectors, computed far more cheaply and reliably than those of a general matrix.
- **How well conditioned?** The **condition number** $\kappa(A) = \|A\|\|A^{-1}\|$ measures how much the solution can
  change relative to a change in the data. No algorithm can do better than lose about $\log_{10}\kappa$
  digits; a good algorithm does not lose more.

## Stability and conditioning

The organising idea of numerical linear algebra is **backward stability**, developed by James
Wilkinson in the 1950s and 1960s. An algorithm is backward stable if the answer it computes is the
exact answer for a slightly perturbed problem: the computed solution $\hat{\mathbf x}$ of $A\mathbf x = \mathbf b$ satisfies
$(A + \delta A)\hat{\mathbf x} = \mathbf b$ with $\|\delta A\|/\|A\|$ a small multiple of the machine precision $\varepsilon$. Combined with the
conditioning of the problem, this bounds the error:

$$
\frac{\|\hat{\mathbf x} - \mathbf x\|}{\|\mathbf x\|} \lesssim \kappa(A)\,\frac{\|\delta A\|}{\|A\|} \approx \kappa(A)\,\varepsilon .
$$

The rule of thumb, **forward error ≈ condition number × backward error**, explains almost
every numerical result in this part. Gaussian elimination with partial pivoting, Householder QR, the
QR algorithm and the SVD are all backward stable (elimination in practice rather than in the worst
case); the classical Gram-Schmidt process, the normal equations for least squares, and the
eigenvalues of $A^{\mathsf T}A$ for the SVD are not, and the [reference crate](#the-reference-crate) implements
them to show the difference.

## Two crates

### The production crate

`lib_jsl_linalg` ([API reference](../../api/lib_jsl_linalg/index.html)) is a pure-Rust replacement
for LAPACK, for real (`f64`) and complex (`Complex<f64>`) matrices. Its traits follow the names of the
`ndarray-linalg` crate (`solve`, `inv`, `det`, `factorize`, `cholesky`, `solve_triangular`, `eigh`,
`eig`, `svd`, `least_squares`), and are implemented for every ndarray two-dimensional array. The
factorisations are organised, as in LAPACK, so that most of their arithmetic is done by a fast
matrix-multiply kernel (from the `matrixmultiply` crate). Against OpenBLAS on one core, most
operations take between 0.4 and 1.6 times as long (the README's benchmark table); the open issues
[#9](https://github.com/jwkunz/lib_jsl/issues/9), [#10](https://github.com/jwkunz/lib_jsl/issues/10) and
[#11](https://github.com/jwkunz/lib_jsl/issues/11) track the remaining gaps.

### The reference crate

`lib_jsl_linalg_reference` ([API reference](../../api/lib_jsl_linalg_reference/index.html)) contains
the textbook algorithms, written to look like the formulas: Gaussian elimination with and without
pivoting, Doolittle LU, Cholesky-Banachiewicz, classical and modified Gram-Schmidt, Householder QR,
power and Rayleigh quotient iteration, cyclic Jacobi, the unshifted and shifted QR algorithm, the
SVD through $A^{\mathsf T}A$, and least squares by the normal equations. They are for study, and each entry in
this part contrasts them with the production algorithm: what pivoting, reflections, shifts,
blocking and bidiagonalisation buy. `cargo bench -p lib_jsl_linalg_reference` races the two.

## Testing against SciPy

The references come from SciPy 1.17.1's `scipy.linalg`, which calls LAPACK (through OpenBLAS), on random matrices written by `book/generator/data/make_linear_algebra_refs.py`. Each entry is the largest difference relative to the largest entry of SciPy's result; factors determined only up to sign are compared in absolute value:

{{#include ../generated/linear_algebra/scipy_reference.md}}

Both crates agree with LAPACK to a few units of rounding error on these well-conditioned matrices. Where their algorithms differ in stability, as in the SVD through $A^{\mathsf T}A$ or the normal equations, the difference shows only on ill-conditioned problems, which the entries of this part examine.

## Choosing a method

| Your problem | Use | Entry |
|---|---|---|
| Solve $A\mathbf x = \mathbf b$, general square $A$ | `solve`, or `factorize` for many right-hand sides | [Gaussian elimination and LU](lu.md) |
| Solve $A\mathbf x = \mathbf b$, symmetric positive definite $A$ | `cholesky` and two triangular solves | [Cholesky factorisation](cholesky.md) |
| Determinant or inverse | `det`, `inv` (prefer solving to inverting) | [Gaussian elimination and LU](lu.md#lu-factorisation) |
| An orthonormal basis, or a rank-revealing factorisation | Householder QR (with column pivoting) | [QR factorisation](qr.md) |
| Overdetermined, underdetermined or rank-deficient $A\mathbf x \approx \mathbf b$ | `least_squares` | [Linear least squares](least-squares.md) |
| All eigenvalues (and vectors) of a symmetric or Hermitian matrix | `eigh`, `eigvalsh` | [Symmetric and general eigensolvers](eigensolvers.md) |
| All eigenvalues (and vectors) of a general matrix | `eig`, `eigvals` | [Symmetric and general eigensolvers](eigensolvers.md#general-matrices-hessenberg-reduction-and-franciss-algorithm) |
| One eigenpair, near a known value | inverse or Rayleigh quotient iteration | [Eigenvalue algorithms](eigen-iterations.md) |
| Singular values, rank, condition number, low-rank approximation | `svd` (`svd_jacobi` for graded matrices) | [The singular value decomposition](svd.md) |
