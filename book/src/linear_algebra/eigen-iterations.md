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

Expanding a starting vector in eigenvectors, $\mathbf x = \sum c_i\mathbf v_i$, repeated multiplication gives
$A^k\mathbf x = \sum c_i\lambda_i^k\mathbf v_i$, dominated by the eigenvalue of largest magnitude. **Power iteration**,
$\mathbf x \leftarrow A\mathbf x/\|A\mathbf x\|$, converges to $\mathbf v_1$ at the rate $|\lambda_2/\lambda_1|$ per step: slowly when the two largest eigenvalues are
close. The best eigenvalue estimate from a vector is the **Rayleigh quotient** $\rho(\mathbf x) = \mathbf x^{\mathsf T}A\mathbf x/\mathbf x^{\mathsf T}\mathbf x$, which for
symmetric $A$ is accurate to the *square* of the error in $\mathbf x$.

**Inverse iteration** applies power iteration to $(A - \sigma I)^{-1}$, whose eigenvalues are $1/(\lambda_i - \sigma)$: it converges
to the eigenvalue nearest the **shift** $\sigma$, at the rate $|\lambda_{\text{nearest}} - \sigma|/|\lambda_{\text{next}} - \sigma|$, which a good shift makes tiny.
**Rayleigh quotient iteration** updates the shift to $\rho(\mathbf x)$ at every step and converges cubically for symmetric
matrices, at the cost of a new factorisation per step:

{{#include ../generated/linear_algebra/power_iteration.md}}

Power iteration slows in proportion to $1/(1 - |\lambda_2/\lambda_1|)$; inverse iteration with a shift near 10 is fast in every case.
Rayleigh quotient iteration needs three to five steps, but it converges to *an* eigenvalue, the one its starting
vector leads it to (here 0.37, in the middle of the spectrum), not necessarily the one wanted.

### The Jacobi method

A plane rotation $J(p, q, \theta)$ can be chosen so that $J^{\mathsf T}AJ$ has a zero in positions $(p, q)$ and $(q, p)$. The angle satisfies
$\cot 2\theta = (a_{qq} - a_{pp})/(2a_{pq})$, and the smaller root keeps $|\theta| \le \pi/4$. Later rotations fill the zero in again, but the
off-diagonal sum of squares falls by exactly $2a_{pq}^2$ at each rotation, so sweeping repeatedly over all pairs drives $A$
to a diagonal matrix of eigenvalues; the product of the rotations holds the eigenvectors. Convergence is
quadratic once the off-diagonal part is small, and 6 to 10 sweeps suffice whatever the order (table below).
Jacobi's method is slower than the QR algorithm, but simple, naturally parallel, and accurate: it determines
small eigenvalues of graded matrices to high relative accuracy (Demmel and Veselić, 1992), which is why its
one-sided form is used for the [SVD](svd.md#one-sided-jacobi).

### The QR algorithm

Factor $A_k = Q_kR_k$ and multiply in the other order:

$$
A_{k+1} = R_kQ_k = Q_k^{\mathsf T}A_kQ_k .
$$

Each iterate is orthogonally similar to $A$, so it has the same eigenvalues. The accumulated product $Q_1\cdots Q_k$ is exactly
what **subspace iteration** (power iteration on $n$ vectors at once, kept orthonormal) would produce, so $A_k$ converges to a
diagonal (for symmetric $A$) or triangular matrix, with entry $(i, j)$ decaying like $|\lambda_i/\lambda_j|^k$. The **unshifted** algorithm
therefore converges slowly when eigenvalues are close in magnitude, and not at all for eigenvalues of equal magnitude.

The **shifted** QR algorithm factors $A_k - \mu_kI$ and adds $\mu_kI$ back. With a shift near an eigenvalue, the last row converges in a few
steps (it is inverse iteration in disguise); that eigenvalue is then **deflated**, and the iteration continues on the leading
$(n - 1) \times (n - 1)$ block. **Wilkinson's shift**, the eigenvalue of the trailing $2 \times 2$ block nearer its corner, converges for every
symmetric matrix, cubically in general:

{{#include ../generated/linear_algebra/jacobi_qr.md}}

The shifted algorithm needs about two steps per eigenvalue, independently of the spectrum. But each step on a full matrix
costs $O(n^3)$, for $O(n^4)$ in all; production codes first reduce the matrix to tridiagonal or Hessenberg form, which makes each
step $O(n)$ or $O(n^2)$ ([next entry](eigensolvers.md)).

## Algorithm

- **Power iteration**: $\mathbf x \leftarrow A\mathbf x$, normalise, until the residual $\|A\mathbf x - \rho(\mathbf x)\mathbf x\|$ is below the tolerance.
- **Rayleigh quotient iteration**: solve $(A - \rho(\mathbf x)I)\mathbf y = \mathbf x$, $\mathbf x \leftarrow \mathbf y/\|\mathbf y\|$.
- **Cyclic Jacobi**: sweep over all pairs $p < q$, rotating to annihilate $a_{pq}$, until the off-diagonal norm is below the tolerance.
- **Shifted QR**: while the active block has order above 1: if its last row is negligible, deflate; otherwise take a QR step
  with Wilkinson's shift.

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
