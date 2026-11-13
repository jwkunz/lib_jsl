# Eigenvalue algorithms: power iteration, Jacobi and the QR algorithm

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Compute eigenvalues $\lambda$ and eigenvectors $\mathbf v$ with $A\mathbf v = \lambda\mathbf v$, one at a time (power and inverse iteration) or all at once (Jacobi, QR algorithm) |
| **Devised by** | Carl Gustav Jacobi (1846), the rotation method; Richard von Mises (1929), power iteration; Helmut Wielandt (1944), inverse iteration; Lord Rayleigh, the quotient, with cubic convergence of the iteration shown by Alexander Ostrowski (1958); Heinz Rutishauser (1958), the LR algorithm; John Francis (1961) and Vera Kublanovskaya (1961), the QR algorithm |
| **Accuracy** | For symmetric matrices: absolute errors of about $\varepsilon\|A\|$ in the eigenvalues; Jacobi also gives small eigenvalues to high relative accuracy |
| **Cost** | Power iteration: $O(n^2)$ per step, $O(1/\log(|\lambda_1|/|\lambda_2|))$ steps. Jacobi: $O(n^3)$ per sweep, 6–10 sweeps. QR algorithm on a full matrix: $O(n^3)$ per step, about two steps per eigenvalue with shifts |
| **API** | Reference: [`power_iteration`](../../api/lib_jsl_linalg_reference/power_iteration/index.html), [`jacobi_eigenvalue`](../../api/lib_jsl_linalg_reference/jacobi_eigenvalue/index.html), [`qr_algorithm`](../../api/lib_jsl_linalg_reference/qr_algorithm/index.html) |

</div>

## Overview

Eigenvalues describe the natural behaviour of linear systems: the frequencies of vibration of a
structure, the growth rates of a population model, the energy levels of a quantum system, the
stability of an equilibrium, the principal axes of a dataset. Unlike linear systems, eigenvalue
problems cannot be solved in a finite number of steps for $n \ge 5$: the eigenvalues are the roots of the
characteristic polynomial, and by the Abel-Ruffini theorem there is no formula. Every eigenvalue
algorithm is therefore iterative.

This entry describes the three classical ideas, as implemented in the reference crate:
**power iteration** and its shifted variants, which find one eigenpair; **Jacobi's method**, which
annihilates off-diagonal entries by rotations; and the **QR algorithm**, the basis of all modern
eigensolvers. The [next entry](eigensolvers.md) describes how production codes make the QR algorithm
fast.

## History

Carl Gustav Jacobi (1846) diagonalised symmetric matrices by plane rotations, to compute the secular
perturbations of the planets; his method was forgotten and rediscovered for computers by Herman
Goldstine, Francis Murray and John von Neumann in the 1950s. Richard von Mises (1929) described power
iteration; Helmut Wielandt (1944) introduced inverse iteration with shifts. Heinz Rutishauser (1958)
discovered that factoring $A = LR$ and multiplying in reverse order converges to a triangular matrix, and in
1961 John Francis in England and Vera Kublanovskaya in the Soviet Union independently replaced $LR$ by the
stable $QR$ factorisation. Francis also introduced the Hessenberg reduction, shifts and the implicit
double-shift step that made the method practical; it was later named one of the ten most important
algorithms of the twentieth century.

## The idea

### Power iteration, inverse iteration and the Rayleigh quotient

An **eigenvector** of $A$ is a nonzero vector $\mathbf v$ that $A$ merely stretches,
$A\mathbf v = \lambda\mathbf v$; the factor $\lambda$ is its **eigenvalue**. For matrices larger than
$4 \times 4$ there is no formula for the eigenvalues (they are roots of a polynomial of degree $n$),
so every eigenvalue algorithm is an iteration. The simplest one shows the principle behind all the
others.

**Power iteration.** Suppose $A$ has eigenvectors $\mathbf v_1, \ldots, \mathbf v_n$ with eigenvalues
ordered by magnitude, $|\lambda_1| > |\lambda_2| \ge \cdots$. Any starting vector is a combination of
the eigenvectors, $\mathbf x = \sum c_i\mathbf v_i$. Each multiplication by $A$ multiplies the $i$-th
component by $\lambda_i$, so after $k$ multiplications

$$
A^k\mathbf x = \sum_i c_i\lambda_i^k\mathbf v_i
= \lambda_1^k\Big(c_1\mathbf v_1 + c_2\Big(\frac{\lambda_2}{\lambda_1}\Big)^{\!k}\mathbf v_2 + \cdots\Big).
$$

Every ratio $(\lambda_i/\lambda_1)^k$ tends to zero, so the direction of $A^k\mathbf x$ tends to that of
$\mathbf v_1$: repeated multiplication amplifies the dominant eigenvector relative to all the others.
To stop the vector overflowing or underflowing, it is rescaled to unit length at each step:

$$
\mathbf x \leftarrow \frac{A\mathbf x}{\|A\mathbf x\|} .
$$

The unwanted components shrink by the factor $|\lambda_2/\lambda_1|$ per step. This is linear
convergence, and it is slow when the two largest eigenvalues are close in magnitude.

**The Rayleigh quotient.** Given an approximate eigenvector $\mathbf x$, the best estimate of the
corresponding eigenvalue is

$$
\rho(\mathbf x) = \frac{\mathbf x^{\mathsf T}A\mathbf x}{\mathbf x^{\mathsf T}\mathbf x} .
$$

If $\mathbf x$ is an exact eigenvector this equals $\lambda$. For symmetric $A$ it has a valuable
property: if $\mathbf x$ differs from an eigenvector by an error of size $\delta$, then $\rho(\mathbf x)$
differs from the eigenvalue by only about $\delta^2$. An eigenvector known to three digits gives an
eigenvalue good to six.

**Inverse iteration.** Power iteration finds only the largest eigenvalue. To find others, apply
it to a different matrix with the same eigenvectors. If $A\mathbf v = \lambda\mathbf v$, then for any number
$\sigma$,

$$
(A - \sigma I)^{-1}\mathbf v = \frac{1}{\lambda - \sigma}\,\mathbf v .
$$

The matrix $(A - \sigma I)^{-1}$ has the same eigenvectors as $A$, with eigenvalues
$1/(\lambda_i - \sigma)$. The largest of these belongs to the $\lambda_i$ *closest* to $\sigma$. So power
iteration on $(A - \sigma I)^{-1}$ converges to the eigenvector whose eigenvalue is nearest the
**shift** $\sigma$, and the rate is $|\lambda_{\text{nearest}} - \sigma|/|\lambda_{\text{next}} - \sigma|$. If
$\sigma$ is a good estimate of an eigenvalue, the numerator is tiny and convergence takes a step or
two. The inverse is never formed: each step solves the linear system
$(A - \sigma I)\mathbf y = \mathbf x$ using one [LU factorisation](lu.md) computed at the start.

**Rayleigh quotient iteration** combines the last two ideas: after every step, replace the shift
by the Rayleigh quotient of the current vector. A better vector gives a better shift, which gives
a much better vector. For symmetric matrices the convergence is cubic, tripling the number of
correct digits per step, at the cost of a new factorisation at each step because the shift
changes.

{{#include ../generated/linear_algebra/power_iteration.md}}

Power iteration slows in proportion to $1/(1 - |\lambda_2/\lambda_1|)$; inverse iteration with a shift near 10 is fast in every case.
Rayleigh quotient iteration needs three to five steps, but it converges to *an* eigenvalue, the one its starting
vector leads it to (here 0.37, in the middle of the spectrum), not necessarily the one wanted.

### The Jacobi method

The methods above find one eigenpair. To find all of them for a symmetric matrix, the goal is an
orthogonal matrix $V$ with $V^{\mathsf T}AV = D$ diagonal: the diagonal of $D$ then holds the
eigenvalues and the columns of $V$ the eigenvectors. Jacobi's method builds $V$ from the simplest
orthogonal matrices, rotations in a coordinate plane.

For a $2 \times 2$ symmetric matrix a single rotation does the job exactly. In an $n \times n$
matrix, a **plane rotation** $J(p, q, \theta)$ acts only on coordinates $p$ and $q$, and the angle can
be chosen so that $J^{\mathsf T}AJ$ has a zero in positions $(p, q)$ and $(q, p)$. The angle satisfies
$\cot 2\theta = (a_{qq} - a_{pp})/(2a_{pq})$, and the smaller of its two solutions is taken, which
keeps $|\theta| \le \pi/4$.

The rotation changes all of rows and columns $p$ and $q$, so a zero created by one rotation is
generally filled in again by later ones. Progress is nevertheless guaranteed. An orthogonal
similarity preserves the sum of squares of all the entries, and the rotation moves $2a_{pq}^2$ of
that sum from the off-diagonal part onto the diagonal. The off-diagonal sum of squares therefore
falls by exactly $2a_{pq}^2$ at each rotation and never increases. Sweeping repeatedly over all
pairs $(p, q)$ drives $A$ to a diagonal matrix of eigenvalues, and the product of all the
rotations is $V$.

Convergence is quadratic once the off-diagonal part is small, and 6 to 10 sweeps suffice whatever the order (table below).
Jacobi's method is slower than the QR algorithm, but simple, naturally parallel, and accurate: it determines
small eigenvalues of graded matrices to high relative accuracy (Demmel and Veselić, 1992), which is why its
one-sided form is used for the [SVD](svd.md#one-sided-jacobi).

### The QR algorithm

The standard method for all the eigenvalues of a matrix looks, at first sight, like a trick.
Factor the current matrix as $A_k = Q_kR_k$ by a [QR factorisation](qr.md), and multiply the
factors back together in the *opposite* order:

$$
A_{k+1} = R_kQ_k .
$$

**It preserves the eigenvalues.** Since $R_k = Q_k^{\mathsf T}A_k$, the new matrix is
$A_{k+1} = Q_k^{\mathsf T}A_kQ_k$. This is a similarity transformation by an orthogonal matrix, so
$A_{k+1}$ has the same eigenvalues as $A_k$, and by induction the same as $A$. Symmetry is
preserved too.

**It converges, because it is power iteration in disguise.** Apply power iteration to $n$
vectors at once, re-orthonormalising them after each multiplication so that they do not all
collapse onto the dominant eigenvector. This is **subspace iteration**: the first vector
converges to the dominant eigenvector, the first two span the dominant two-dimensional
invariant subspace, and so on. It can be shown that the accumulated product $Q_1Q_2\cdots Q_k$ of the QR
algorithm is exactly the orthonormal basis that subspace iteration produces from the identity
matrix. Consequently $A_k$ converges to a diagonal matrix (for symmetric $A$) or an upper triangular one
(in general), with the eigenvalues on the diagonal in order of decreasing magnitude. The entry in
position $(i, j)$ below the diagonal decays like $|\lambda_i/\lambda_j|^k$.

This **unshifted** algorithm inherits the weakness of power iteration: it converges slowly when
eigenvalues are close in magnitude, and not at all for eigenvalues of equal magnitude.

**Shifts make it fast.** The remedy is the same as before. The **shifted** QR algorithm factors
$A_k - \mu_kI = Q_kR_k$ and sets $A_{k+1} = R_kQ_k + \mu_kI$, which is still a similarity
transformation. It turns out that the last row of the iteration behaves like inverse iteration
with shift $\mu_k$. With a shift close to an eigenvalue, the off-diagonal entries of the last row
fall to zero in a few steps, leaving that eigenvalue isolated in the bottom corner. It is then
**deflated**: set aside as converged, while the iteration continues on the leading
$(n - 1) \times (n - 1)$ block.

A good shift is available from the matrix itself. **Wilkinson's shift** is the eigenvalue of the
trailing $2 \times 2$ block that is nearer to its bottom right entry. With it the algorithm converges for every
symmetric matrix, cubically in general:

{{#include ../generated/linear_algebra/jacobi_qr.md}}

The shifted algorithm needs about two steps per eigenvalue, independently of the spectrum. But each step on a full matrix
costs $O(n^3)$, for $O(n^4)$ in all; production codes first reduce the matrix to tridiagonal or Hessenberg form, which makes each
step $O(n)$ or $O(n^2)$ ([next entry](eigensolvers.md)).

## Algorithm

**Power iteration.** Input: $A$, a starting vector $\mathbf x$ of unit length and a tolerance.
Repeat:

1. compute $\mathbf y = A\mathbf x$ and the Rayleigh quotient $\rho = \mathbf x^{\mathsf T}\mathbf y$;
2. if the residual $\|\mathbf y - \rho\mathbf x\|$ is below the tolerance, return $\rho$ and $\mathbf x$;
3. set $\mathbf x = \mathbf y/\|\mathbf y\|$.

Each iteration costs one matrix-vector product.

**Inverse iteration** with shift $\sigma$: factor $A - \sigma I$ once; then run the same loop with
step 3 replaced by "solve $(A - \sigma I)\mathbf y = \mathbf x$ and normalise $\mathbf y$".

**Rayleigh quotient iteration**: as inverse iteration, but with the shift replaced by the
current $\rho$ at every step, so that the matrix must be factorised again each time. As the shift
converges to an eigenvalue the matrix becomes nearly singular. That does no harm: the solution
is then huge, but its *direction* is the eigenvector, and only the direction is kept. If the
factorisation finds the matrix exactly singular, the shift is an eigenvalue to rounding error,
and the iteration stops with the current vector.

**Cyclic Jacobi.** Input: a symmetric $A$. Set $V = I$. Repeat:

1. if the off-diagonal norm $\big(\sum_{p \ne q}a_{pq}^2\big)^{1/2}$ is below the tolerance, stop;
2. for every pair $p < q$ in turn (one **sweep**): compute the rotation angle from $a_{pp}$,
   $a_{qq}$ and $a_{pq}$; apply the rotation to rows and columns $p$ and $q$ of $A$, which sets
   $a_{pq}$ to zero; apply it to columns $p$ and $q$ of $V$.

The diagonal of $A$ then holds the eigenvalues and the columns of $V$ the eigenvectors. A sweep
costs $O(n^3)$.

**Shifted QR.** Input: a symmetric $A$. The **active block** is the leading part of the matrix
not yet deflated; it starts as the whole matrix. While its order exceeds 1:

1. if the off-diagonal entries of its last row are negligible compared with the neighbouring
   diagonal entries, set them to zero, record the last diagonal entry as an eigenvalue, and
   shrink the active block by one (deflation);
2. otherwise compute Wilkinson's shift $\mu$ from the trailing $2 \times 2$ block, factor
   $A - \mu I = QR$ on the active block, and replace the block by $RQ + \mu I$.

## Implementation

Rayleigh quotient iteration:

```rust,ignore
{{#include ../../../crates/linalg_reference/src/power_iteration.rs:rqi}}
```

Cyclic Jacobi:

```rust,ignore
{{#include ../../../crates/linalg_reference/src/jacobi_eigenvalue.rs:jacobi}}
```

The shifted QR algorithm with deflation:

```rust,ignore
{{#include ../../../crates/linalg_reference/src/qr_algorithm.rs:shifted_qr}}
```

- The reference routines are for real symmetric matrices, written for clarity; the production solvers handle general
  and complex matrices.

## In practice

```rust,ignore
{{#include ../../../crates/linalg_reference/examples/linalg_eigen_iterations.rs}}
```

**Strengths**

- Power and inverse iteration need only matrix-vector products or one factorisation, and suit large sparse matrices when
  a few eigenvalues are wanted (the basis of Lanczos and Arnoldi methods).
- Jacobi is simple, parallel and accurate; the shifted QR algorithm converges reliably and fast.

**Weaknesses**

- Power iteration is slow for close eigenvalues; Rayleigh quotient iteration may converge to an unwanted eigenvalue.
- On full matrices, Jacobi and QR cost $O(n^3)$ per sweep or step: they are practical only after reduction, or for small $n$.

## Exercises

1. Show that the Rayleigh quotient of a vector with error $\delta$ in the eigenvector direction has error $O(\delta^2)$ for symmetric $A$.
2. Show that a Jacobi rotation reduces the off-diagonal sum of squares by exactly $2a_{pq}^2$.
3. Show that $A_{k+1} = R_kQ_k$ is orthogonally similar to $A_k$, and that $Q_1\cdots Q_k R_k\cdots R_1 = A^k$.
4. Run unshifted QR on a matrix with eigenvalues $1$ and $-1$. Why does it never converge, and how does a shift fix it?
5. Use inverse iteration with the eigenvalues computed by `eigvalsh` as shifts to compute the eigenvectors, and compare
   with `eigh`.

## Further reading

- B. N. Parlett, *The Symmetric Eigenvalue Problem*, Prentice-Hall, 1980 (SIAM reprint, 1998).
- J. G. F. Francis, "The QR transformation: a unitary analogue to the LR transformation", *The Computer Journal* 4 (1961)
  265–271 and 332–345.
- J. Demmel and K. Veselić, "Jacobi's method is more accurate than QR", *SIAM Journal on Matrix Analysis and
  Applications* 13 (1992) 1204–1245.
- D. S. Watkins, "Francis's algorithm", *American Mathematical Monthly* 118 (2011) 387–403.
- L. N. Trefethen and D. Bau, *Numerical Linear Algebra*, SIAM, 1997, lectures 24–31.
