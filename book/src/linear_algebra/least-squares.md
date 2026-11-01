# Linear least squares

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Find $\mathbf x$ minimising $\|A\mathbf x - \mathbf b\|_2$ for an $m \times n$ matrix $A$: the best fit to an overdetermined system, or the minimum-norm solution of an underdetermined or rank-deficient one |
| **Devised by** | Adrien-Marie Legendre (1805) and Carl Friedrich Gauss (1809), least squares and the normal equations; Gene Golub (1965), solution by Householder QR; Eliakim Moore (1920) and Roger Penrose (1955), the pseudoinverse; Golub and William Kahan (1965), solution by the SVD |
| **Accuracy** | QR: backward stable, error $\propto \kappa(A)\varepsilon$ (plus $\kappa^2\varepsilon$ times the relative residual). Normal equations: error $\propto \kappa(A)^2\varepsilon$ |
| **Cost** | QR: $2mn^2 - \tfrac23n^3$ flops. Normal equations: $mn^2 + \tfrac13n^3$. SVD: several times more |
| **API** | [`LeastSquaresSvd`](../../api/lib_jsl_linalg/trait.LeastSquaresSvd.html) (`least_squares`), [`LeastSquaresResult`](../../api/lib_jsl_linalg/struct.LeastSquaresResult.html); reference: [`least_squares`](../../api/lib_jsl_linalg_reference/least_squares/index.html) (`normal_equations`, `qr_least_squares`) |

</div>

## Overview

When there are more equations than unknowns, as in fitting a model with a few parameters to many
measurements, $A\mathbf x = \mathbf b$ generally has no solution, and the **least-squares** solution minimises the sum of
squared residuals instead. It is the maximum-likelihood estimate when the errors are independent and
Gaussian, and the most widely used method of data fitting. When there are fewer equations than unknowns,
or the columns of $A$ are dependent, there are infinitely many minimisers, and the natural choice is the one
of smallest norm.

There are three standard methods, in increasing order of cost and robustness: the **normal equations**,
**QR factorisation**, and the **singular value decomposition**. The first squares the condition number; the
second does not; the third also handles rank deficiency gracefully.

## History

Adrien-Marie Legendre published the method of least squares in 1805, and Carl Friedrich Gauss in 1809,
claiming priority and justifying it by the normal distribution of errors. Both solved the **normal
equations** $A^{\mathsf T}A\mathbf x = A^{\mathsf T}\mathbf b$ by elimination, and for a century and a half this was the method, done by hand for the
geodetic surveys of Europe. Gene Golub's 1965 paper showed that Householder QR avoids forming $A^{\mathsf T}A$ and
its squared condition number, and made QR the standard. Eliakim Moore (1920) and Roger Penrose (1955)
defined the **pseudoinverse**, which gives the minimum-norm solution, and Golub and William Kahan (1965)
showed how to compute it stably through the SVD.

## The idea

### The normal equations

At the minimum, the residual $\mathbf r = \mathbf b - A\mathbf x$ is orthogonal to the column space of $A$: $A^{\mathsf T}\mathbf r = \mathbf 0$, which gives the
**normal equations**

$$
A^{\mathsf T}A\,\mathbf x = A^{\mathsf T}\mathbf b .
$$

If $A$ has full column rank, $A^{\mathsf T}A$ is symmetric positive definite, and [Cholesky](cholesky.md) solves the system
cheaply. But $\kappa(A^{\mathsf T}A) = \kappa(A)^2$: forming the product loses twice as many digits as the problem requires, and
for $\kappa(A) > 10^8$ the computed $A^{\mathsf T}A$ may not even be positive definite.

### QR factorisation

With $A = QR$ (thin QR, $Q$ of size $m \times n$) and $Q_\perp$ completing it to an orthogonal matrix, orthogonality gives

$$
\|A\mathbf x - \mathbf b\|^2 = \|R\mathbf x - Q^{\mathsf T}\mathbf b\|^2 + \|Q_\perp^{\mathsf T}\mathbf b\|^2,
$$

so the minimiser solves the triangular system $R\mathbf x = Q^{\mathsf T}\mathbf b$, and the minimum residual is $\|Q_\perp^{\mathsf T}\mathbf b\|$. With Householder QR,
$Q^{\mathsf T}\mathbf b$ is computed by applying the reflections to $\mathbf b$, and $Q$ is never formed. The error is proportional to $\kappa(A)\varepsilon$
(with an additional $\kappa^2\varepsilon$ term weighted by the relative size of the residual, which is inherent in the problem).

Fitting a polynomial of degree $d$ in the monomial basis, whose Vandermonde matrix becomes more ill-conditioned as $d$
grows, shows the difference:

![Normal equations against QR](../generated/linear_algebra/least_squares.svg)

The normal equations follow $\kappa^2\varepsilon$ and fail outright (Cholesky finds $A^{\mathsf T}A$ indefinite) beyond degree 13; both QR
methods follow $\kappa\varepsilon$. (Better still, fit in an orthogonal basis such as
[Chebyshev polynomials](../interpolation/polynomial.md#chebyshev-series), whose matrix is well conditioned.)

### Rank deficiency and the minimum-norm solution

If $A$ has rank $r < n$, the minimisers form an affine subspace, and the **pseudoinverse** picks the one of least norm:
with the SVD $A = U\Sigma V^{\mathsf T}$,

$$
\mathbf x = A^+\mathbf b = \sum_{i=1}^{r}\frac{\mathbf u_i^{\mathsf T}\mathbf b}{\sigma_i}\,\mathbf v_i .
$$

Numerically, rank is a matter of degree: singular values below a threshold, here $\max(m, n)\,\varepsilon\,\sigma_1$, are treated as zero.
Including them would divide by numbers that are rounding errors and give a huge, meaningless solution.

The library's `least_squares` combines the methods. It factorises with **pivoted QR** and, if the diagonal of $R$ shows
full column rank, solves by back substitution. Otherwise, or when the system is underdetermined, it uses the SVD
and returns the minimum-norm solution. It reports the numerical rank, the singular values and the residual sum of
squares:

{{#include ../generated/linear_algebra/rank_deficient.md}}

On the full-rank path, the singular values used to be computed as the square roots of the eigenvalues of $R^{\mathsf H}R$,
which reintroduces the squared condition number. For a matrix with singular values $1$, $10^{-5}$ and $10^{-10}$ the
smallest came out as $3.5 \times 10^{-9}$. They now come from the SVD of $R$.

## Algorithm

`least_squares(A, b)`:

1. If $m \ge n$: factor $AP = QR$ with column pivoting, compute $Q^{\mathsf H}\mathbf b$. If every $|r_{ii}| > 10^3\max(m, n)\,\varepsilon|r_{11}|$, solve $R\mathbf z = (Q^{\mathsf H}\mathbf b)_{1:n}$,
   set $\mathbf x = P\mathbf z$, and take the singular values from the SVD of $R$ and the residual from $(Q^{\mathsf H}\mathbf b)_{n+1:m}$.
2. Otherwise compute the thin SVD, determine the rank, and form $\mathbf x = \sum_{i\le r}\sigma_i^{-1}(\mathbf u_i^{\mathsf H}\mathbf b)\mathbf v_i$.

## Implementation

The normal equations (reference):

```rust,ignore
{{#include ../../../crates/linalg_reference/src/least_squares.rs:normal_equations}}
```

The production solver:

```rust,ignore
{{#include ../../../crates/linalg/src/mod.rs:least_squares}}
```

- Works for real and complex matrices.
- For weighted least squares, scale each row of $A$ and $\mathbf b$ by the square root of its weight; for nonlinear models, see
  [nonlinear least squares](../optimization/least-squares.md).

## In practice

```rust,ignore
{{#include ../../../crates/linalg_reference/examples/linalg_least_squares.rs}}
```

**Strengths**

- Backward-stable QR for full-rank problems, the SVD for rank-deficient ones, chosen automatically.
- Reports the rank and singular values, which diagnose ill-conditioning.

**Weaknesses**

- Dense: very large or sparse problems need iterative methods (LSQR, conjugate gradients on the normal equations).
- The rank threshold is a judgement: noisy data may need a larger threshold (truncated SVD) or regularisation.

## Exercises

1. Derive the normal equations by setting the gradient of $\|A\mathbf x - \mathbf b\|^2$ to zero.
2. Show that $\kappa_2(A^{\mathsf T}A) = \kappa_2(A)^2$ using the SVD.
3. Show that the minimum-norm solution is orthogonal to the null space of $A$.
4. Fit a degree-15 polynomial on $[0, 1]$ in the monomial and in the Chebyshev basis, and compare the condition numbers and
   the errors.
5. Add Tikhonov regularisation, $\min\|A\mathbf x - \mathbf b\|^2 + \lambda^2\|\mathbf x\|^2$, by augmenting $A$ with $\lambda I$, and study the effect of $\lambda$ on an ill-conditioned fit.

## Further reading

- G. H. Golub, "Numerical methods for solving linear least squares problems", *Numerische Mathematik* 7 (1965)
  206–216.
- Å. Björck, *Numerical Methods for Least Squares Problems*, SIAM, 1996.
- C. L. Lawson and R. J. Hanson, *Solving Least Squares Problems*, Prentice-Hall, 1974 (SIAM reprint, 1995).
- L. N. Trefethen and D. Bau, *Numerical Linear Algebra*, SIAM, 1997, lectures 11, 18 and 19.
