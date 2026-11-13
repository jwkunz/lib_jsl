# Cholesky factorisation

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Factor a symmetric (Hermitian) positive definite matrix as $A = LL^{\mathsf H}$ with $L$ lower triangular; solve systems with it; test positive definiteness |
| **Devised by** | André-Louis Cholesky (about 1910, published posthumously by Ernest Benoît in 1924); Tadeusz Banachiewicz (1938), the row-oriented form; blocked algorithms: the LAPACK project (1992) |
| **Accuracy** | Backward stable without any pivoting: the computed factor satisfies $LL^{\mathsf H} = A + \delta A$ with $\|\delta A\| \le c\,n\,\varepsilon\|A\|$ |
| **Cost** | $\tfrac13n^3$ flops, half of LU; $2n^2$ per solve |
| **API** | [`Cholesky`](../../api/lib_jsl_linalg/trait.Cholesky.html), [`SolveTriangular`](../../api/lib_jsl_linalg/trait.SolveTriangular.html), [`UPLO`](../../api/lib_jsl_linalg/enum.UPLO.html), [`Diag`](../../api/lib_jsl_linalg/enum.Diag.html); reference: [`cholesky`](../../api/lib_jsl_linalg_reference/cholesky/fn.cholesky.html) |

</div>

## Overview

A matrix is **symmetric positive definite** if $A = A^{\mathsf T}$ and $\mathbf x^{\mathsf T}A\mathbf x > 0$ for every $\mathbf x \ne \mathbf 0$. Such matrices are
everywhere: covariance matrices in statistics, stiffness matrices in structural mechanics and finite
elements, the normal equations of least squares, Gram matrices of kernel methods, the Hessians of
convex functions at a minimum. For them, Gaussian elimination simplifies into the **Cholesky
factorisation** $A = LL^{\mathsf T}$: half the work, half the storage, no pivoting, and guaranteed stability. It also
answers a question in passing: the factorisation succeeds exactly when the matrix is positive definite,
which makes it the cheapest test for that property.

Beyond solving systems, the Cholesky factor generates correlated random vectors ($L\mathbf z$ has covariance $A$
when $\mathbf z$ has independent standard normal entries), computes Mahalanobis distances and log-determinants
($\ln\det A = 2\sum\ln\ell_{ii}$), and whitens data.

## History

André-Louis Cholesky, a French artillery officer and geodesist, devised the method around 1910 to
solve the normal equations of geodetic surveys in Crete and North Africa. He was killed in the First
World War in 1918, and his method was published by his colleague Ernest Benoît in 1924. Tadeusz
Banachiewicz, a Polish astronomer, independently developed it in 1938 in his "Cracovian" matrix
notation, in the row-by-row order that bears his name. Alan Turing and James Wilkinson analysed its
stability in the 1940s and 1960s, showing that it needs no pivoting at all.

## The idea

### The factorisation

We look for a lower triangular matrix $L$ with $A = LL^{\mathsf T}$. For a $2 \times 2$ matrix, writing
out the product shows how the entries of $L$ can be found one at a time:

$$
\begin{pmatrix} a_{11} & a_{21} \\ a_{21} & a_{22} \end{pmatrix} =
\begin{pmatrix} \ell_{11} & 0 \\ \ell_{21} & \ell_{22} \end{pmatrix}
\begin{pmatrix} \ell_{11} & \ell_{21} \\ 0 & \ell_{22} \end{pmatrix} =
\begin{pmatrix} \ell_{11}^2 & \ell_{11}\ell_{21} \\ \ell_{11}\ell_{21} & \ell_{21}^2 + \ell_{22}^2 \end{pmatrix}.
$$

The top left entry gives $\ell_{11} = \sqrt{a_{11}}$. The off-diagonal entry then gives
$\ell_{21} = a_{21}/\ell_{11}$, and the last gives $\ell_{22} = \sqrt{a_{22} - \ell_{21}^2}$. For
$A = \begin{pmatrix} 4 & 2 \\ 2 & 5 \end{pmatrix}$ this yields $\ell_{11} = 2$, $\ell_{21} = 1$ and
$\ell_{22} = \sqrt{5 - 1} = 2$.

The same comparison of entries works for any size. Entry $(i, j)$ of $LL^{\mathsf T}$ is the dot
product of rows $i$ and $j$ of $L$, $\sum_{k \le j}\ell_{ik}\ell_{jk}$ for $i \ge j$. Setting it equal to
$a_{ij}$ and solving for the one entry not yet known gives

$$
\ell_{jj} = \sqrt{a_{jj} - \sum_{k<j}\ell_{jk}^2}, \qquad \ell_{ij} = \frac1{\ell_{jj}}\Big(a_{ij} - \sum_{k<j}\ell_{ik}\ell_{jk}\Big) \quad (i > j).
$$

Working through the matrix row by row (or column by column), every entry on the right has
already been computed when it is needed. The cost is about $\tfrac13n^3$ flops, half that of
[LU](lu.md), because symmetry means only half the matrix has to be processed.

Three properties make this factorisation special.

- **It tests positive definiteness.** The quantity under the square root is positive at every
  step if and only if $A$ is positive definite. (It equals the ratio of successive leading
  principal minors of $A$, the determinants of its top left $j \times j$ blocks, and a symmetric
  matrix is positive definite exactly when all of these are positive.) So the algorithm either
  succeeds, or meets a non-positive number under a square root and reports that $A$ is not
  positive definite.
- **It needs no pivoting.** The diagonal entries of $A = LL^{\mathsf T}$ are
  $a_{ii} = \sum_k\ell_{ik}^2$, so every entry of row $i$ of $L$ satisfies $\ell_{ik}^2 \le a_{ii}$. The
  entries of $L$ cannot grow beyond the size of those of $A$. There is no growth factor to
  worry about, and the algorithm is backward stable as it stands.
- **It solves systems like LU.** $A\mathbf x = \mathbf b$ is solved by forward substitution with
  $L\mathbf y = \mathbf b$ followed by back substitution with $L^{\mathsf T}\mathbf x = \mathbf y$.

For complex Hermitian matrices the same formulas hold with the conjugate transpose $L^{\mathsf H}$
and $|\ell_{jk}|^2$.

{{#include ../generated/linear_algebra/cholesky_cases.md}}

The Hilbert matrix of order 12, with condition number near $10^{16}$, is still factorised: positive
definiteness is decided by the rounded arithmetic, and a matrix closer to singularity may be reported as
indefinite even though it is not.

### The blocked algorithm

As with [LU](lu.md#blocked-and-recursive-factorisation), the formulas update one entry at a time, and run
at the speed of memory. The **right-looking blocked** algorithm processes a block of $b$ columns at a
time: factor the diagonal block $A_{11} = L_{11}L_{11}^{\mathsf T}$ by the unblocked method; solve $L_{21}L_{11}^{\mathsf T} = A_{21}$ for the panel
below it; and update the trailing matrix, $A_{22} \leftarrow A_{22} - L_{21}L_{21}^{\mathsf T}$, a symmetric rank-$b$ update done by the
matrix-multiply kernel, on the lower triangle only. The update holds almost all the arithmetic.

The library's block size is 16. In the [speed figure](lu.md#blocked-and-recursive-factorisation) the
blocked Cholesky reaches about 7 GFLOP/s, a third of the rate of the recursive LU. It is still faster in
time, since it does half the work, but it trails OpenBLAS by up to 1.8 times for small matrices, a gap
tracked in [issue #10](https://github.com/jwkunz/lib_jsl/issues/10).

## Algorithm

**Cholesky-Banachiewicz** (the reference crate). Input: a symmetric (or Hermitian) matrix $A$,
of which only the lower triangle is read. For each row $i = 1, \ldots, n$:

1. for $j = 1, \ldots, i - 1$, compute
   $\ell_{ij} = \big(a_{ij} - \sum_{k<j}\ell_{ik}\ell_{jk}\big)/\ell_{jj}$, a dot product of the parts of
   rows $i$ and $j$ already computed;
2. compute $s = a_{ii} - \sum_{k<i}\ell_{ik}^2$; if $s \le 0$, stop and report that $A$ is not
   positive definite; otherwise $\ell_{ii} = \sqrt s$.

The cost is $\tfrac13n^3$ flops.

**Blocked Cholesky** (the production crate). Copy the given triangle into $L$, and process the
columns in blocks of 16. For each block, with $A_{11}$ the diagonal block, $A_{21}$ the panel
below it and $A_{22}$ the trailing matrix:

1. factor the diagonal block, $A_{11} = L_{11}L_{11}^{\mathsf T}$, by the unblocked method, failing if
   a diagonal quantity is not positive;
2. solve $L_{21}L_{11}^{\mathsf T} = A_{21}$ for the panel $L_{21}$ (a triangular solve for each row);
3. update the lower triangle of the trailing matrix, $A_{22} \leftarrow A_{22} - L_{21}L_{21}^{\mathsf T}$,
   with the matrix-multiply kernel.

**Solving** $A\mathbf x = \mathbf b$ with the factor: forward substitution for $L\mathbf y = \mathbf b$, then back
substitution for $L^{\mathsf T}\mathbf x = \mathbf y$, $2n^2$ flops in all.

## Implementation

The textbook algorithm:

```rust,ignore
{{#include ../../../crates/linalg_reference/src/cholesky.rs:banachiewicz}}
```

The production loop over blocks:

```rust,ignore
{{#include ../../../crates/linalg/src/cholesky.rs:blocked_cholesky}}
```

- `cholesky(UPLO::Lower)` reads the lower triangle and returns $L$; `UPLO::Upper` reads the upper triangle and
  returns $U = L^{\mathsf H}$. The other triangle of the input is ignored.
- `solve_triangular(uplo, diag, b)` solves with a triangular matrix and a vector or matrix right-hand side;
  `Diag::Unit` assumes a unit diagonal.
- A non-positive pivot is reported as `NotPositiveDefinite` with its index.

## In practice

```rust,ignore
{{#include ../../../crates/linalg_reference/examples/linalg_cholesky.rs}}
```

**Strengths**

- Half the cost of LU, no pivoting, unconditionally stable.
- A built-in test of positive definiteness.

**Weaknesses**

- Only for Hermitian positive definite matrices; for symmetric indefinite matrices, the $LDL^{\mathsf T}$ factorisation
  with Bunch-Kaufman pivoting (not provided) is the analogue.

## Exercises

1. Show that the diagonal quantity at step $j$ is the ratio of the $j$-th and $(j-1)$-th leading principal minors.
2. Prove that $|\ell_{ij}| \le \sqrt{a_{ii}}$, and deduce that the factorisation cannot suffer element growth.
3. Generate 10 000 samples of $L\mathbf z$ with $\mathbf z \sim N(0, I)$ and check that their sample covariance approaches $A$.
4. Compute $\ln\det A$ for a $500 \times 500$ positive definite matrix from the Cholesky factor, and compare with the logarithm
   of `det`. Which overflows?
5. Factor the matrix $A_\delta = \begin{pmatrix}1 & 1\\ 1 & 1 + \delta\end{pmatrix}$ for $\delta = 10^{-8}, 10^{-16}, 10^{-17}$ and explain the results.

## Further reading

- C. Brezinski, "The life and work of André Cholesky", *Numerical Algorithms* 43 (2006) 279–288.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002, chapter 10.
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013,
  §4.2.
