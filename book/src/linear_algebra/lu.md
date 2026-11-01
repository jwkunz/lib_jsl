# Gaussian elimination and LU factorisation

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Solve a square linear system $A\mathbf x = \mathbf b$; factor $PA = LU$ to solve many systems, compute the determinant, or invert $A$ |
| **Devised by** | Elimination: Chinese mathematicians of the *Nine Chapters* (around 200 BC), Carl Friedrich Gauss (1809); the factorisation: Myrick Doolittle (1878), Prescott Crout (1941), Alan Turing (1948); error analysis: John von Neumann and Herman Goldstine (1947), James Wilkinson (1961); recursive LU: Sivan Toledo (1997) |
| **Accuracy** | Backward stable in practice with partial pivoting: the computed solution solves a system perturbed by a small multiple of $\varepsilon$ times the growth factor, which is almost always small |
| **Cost** | $\tfrac23n^3$ flops to factor, $2n^2$ per right-hand side, $2n^3$ for the inverse |
| **API** | [`Solve`](../../api/lib_jsl_linalg/trait.Solve.html), [`Factorize`](../../api/lib_jsl_linalg/trait.Factorize.html), [`LuFactorization`](../../api/lib_jsl_linalg/struct.LuFactorization.html), [`Determinant`](../../api/lib_jsl_linalg/trait.Determinant.html), [`Inverse`](../../api/lib_jsl_linalg/trait.Inverse.html), [`SolveTriangular`](../../api/lib_jsl_linalg/trait.SolveTriangular.html); reference: [`gaussian_elimination`](../../api/lib_jsl_linalg_reference/gaussian_elimination/index.html), [`lu`](../../api/lib_jsl_linalg_reference/lu/index.html) |

</div>

## Overview

**Gaussian elimination** solves $A\mathbf x = \mathbf b$ by subtracting multiples of each equation from the ones below it
until the system is upper triangular, then solving that from the bottom up. It is the oldest
algorithm in numerical linear algebra and still the method of choice for dense square systems.
Recorded properly, elimination computes a factorisation $PA = LU$ into a permutation, a unit lower
triangular matrix of multipliers and an upper triangular matrix. The factorisation is the useful
object: it solves further systems with the same matrix in $O(n^2)$, and gives the determinant for free.

Two refinements make it work on computers. **Partial pivoting**, swapping rows to use the largest
available pivot, makes it stable. **Blocking** reorganises the arithmetic around matrix-matrix
products, which run near the peak speed of modern processors. The reference crate implements the
first, the production crate both.

## History

The *Nine Chapters on the Mathematical Art*, compiled in China around 200 BC, solve systems of up to
five equations by elimination on a counting board, column by column, with the *fangcheng* rule.
Gauss described elimination for the normal equations of least squares in his *Theoria motus* (1809) and
used it in 1810 for the orbit of the asteroid Pallas; his notation made it standard in geodesy and
astronomy. Myrick Doolittle of the
US Coast and Geodetic Survey (1878) organised the hand calculation as a factorisation, and Prescott
Crout (1941) gave the version with a unit upper triangle; Alan Turing (1948) described elimination
explicitly as LU factorisation and introduced the condition number.

When computers arrived, John von Neumann and Herman Goldstine (1947) feared that rounding errors would
make elimination useless beyond a few dozen equations. James Wilkinson (1961) showed, by **backward
error analysis**, that with partial pivoting it is stable unless the elements grow, and that growth is
almost never large in practice, though it can be in contrived examples. The LAPACK project (1992)
blocked the algorithm for cache memories, and Sivan Toledo (1997) proposed the recursive formulation
that LAPACK adopted as `getrf2` and that this library uses.

## The idea

### Gaussian elimination and pivoting

At step $k$, row $i > k$ is replaced by row $i$ minus $\ell_{ik} = a_{ik}/a_{kk}$ times row $k$, which zeroes the entries below
the **pivot** $a_{kk}$. After $n - 1$ steps $A$ has become an upper triangular $U$, and **back substitution**,
$x_i = (c_i - \sum_{j>i}u_{ij}x_j)/u_{ii}$ from $i = n$ down, finishes the solve.

Without pivoting a zero pivot stops the algorithm, and a small one ruins it. For the system
$\begin{pmatrix}\varepsilon & 1\\ 1 & 1\end{pmatrix}\mathbf x = \begin{pmatrix}1\\ 2\end{pmatrix}$, whose solution is close to $(1, 1)$, the multiplier $1/\varepsilon$ is huge, the second
row becomes $(0, 1 - 1/\varepsilon)$, the 1 is lost to rounding, and $x_1$ is computed from a catastrophic cancellation:

{{#include ../generated/linear_algebra/small_pivot.md}}

**Partial pivoting** swaps the row with the largest $|a_{ik}|$, $i \ge k$, into the pivot position, so every
multiplier satisfies $|\ell_{ik}| \le 1$. Wilkinson's analysis shows that the computed solution then satisfies
$(A + \delta A)\hat{\mathbf x} = \mathbf b$ with

$$
\frac{\|\delta A\|_\infty}{\|A\|_\infty} \le c\,n^3\rho\,\varepsilon, \qquad \rho = \frac{\max_{i,j,k}|a_{ij}^{(k)}|}{\max_{i,j}|a_{ij}|},
$$

where the **growth factor** $\rho$ measures how large the entries become during elimination. It can be as large
as $2^{n-1}$: Wilkinson's example has 1 on the diagonal and in the last column and $-1$ below the diagonal, so that
partial pivoting never swaps and the last column doubles at every step:

{{#include ../generated/linear_algebra/wilkinson_growth.md}}

Such matrices are almost never met in practice: for random matrices the growth is about $n^{2/3}$, and the
error is at the level of rounding, as the last column shows. **Complete pivoting**, searching the whole
remaining submatrix, bounds the growth far better but costs $O(n^3)$ comparisons and is rarely used.

### LU factorisation

The multipliers $\ell_{ik}$ form a unit lower triangular $L$, and the reduced matrix is $U$: elimination computes
$PA = LU$, with $P$ the permutation of the row swaps. To solve $A\mathbf x = \mathbf b$, permute $\mathbf b$, solve $L\mathbf y = P\mathbf b$ by forward
substitution and $U\mathbf x = \mathbf y$ by back substitution, $2n^2$ flops in all. The **determinant** is
$\det A = \pm\prod_iu_{ii}$, with the sign of the permutation. The **inverse** costs $2n^3$ flops, three times a solve's
factorisation, and is rarely needed: $A^{-1}\mathbf b$ is computed better, and $n$ times faster, by solving.

### Conditioning

Backward stability bounds the residual; the error depends also on the condition number. For the
notoriously ill-conditioned **Hilbert matrices** $H_{ij} = 1/(i + j - 1)$, the error grows exactly as $\kappa\varepsilon$ while the
backward error stays at the level of rounding:

![Hilbert systems](../generated/linear_algebra/hilbert.svg)

Beyond order 13, $\kappa(H)$ exceeds $1/\varepsilon$: the computed inverse used to estimate it is itself meaningless, and
the "solution" has no correct digits, although it still satisfies the system to rounding accuracy.

### Blocked and recursive factorisation

The textbook algorithm performs $\tfrac23n^3$ flops as updates of single rows, each reading and writing
memory: on modern processors it runs at the speed of memory, not of arithmetic. Reorganising the same
operations into multiplications of blocks lets each number fetched from memory be used many times.
**Recursive LU** splits the columns in half:

1. factor the left half $\begin{pmatrix}A_{11}\\ A_{21}\end{pmatrix} = \begin{pmatrix}L_{11}\\ L_{21}\end{pmatrix}U_{11}$ recursively, with pivoting;
2. solve $L_{11}U_{12} = A_{12}$ for the top right block;
3. update the bottom right block, $A_{22} \leftarrow A_{22} - L_{21}U_{12}$, with one matrix multiplication;
4. factor the updated $A_{22}$ recursively.

Nearly all the arithmetic is in step 3, at every level of the recursion, so the factorisation runs at
close to the speed of the matrix-multiply kernel. The library factors narrow panels (32 columns) in
column-major order, where the pivot searches run along contiguous memory, and solves triangular
systems in blocks the same way:

![Factorisation speed](../generated/linear_algebra/factorisation_speed.svg)

On this laptop the reference Doolittle LU stays near 2 GFLOP/s, limited by memory; the recursive LU
reaches about 20 GFLOP/s at $n = 1024$, and LAPACK with OpenBLAS is 1.1 to 1.6 times faster still
([issue #10](https://github.com/jwkunz/lib_jsl/issues/10)).

## Algorithm

**Gaussian elimination with partial pivoting** (reference): for $k = 1, \ldots, n - 1$: find the largest $|a_{ik}|$, $i \ge k$; swap
rows; for each $i > k$, subtract $\ell_{ik}$ times row $k$ from row $i$ (and from $\mathbf b$). Then back substitute.

**Recursive LU** (production): as above, on panels of at most 32 columns factorised in column-major order
with an inner recursion; a singular pivot (below $10^{-3}\varepsilon\max|a_{ij}|$) is reported as an error.

## Implementation

The textbook algorithm:

```rust,ignore
{{#include ../../../crates/linalg_reference/src/gaussian_elimination.rs:partial_pivoting}}
```

Doolittle LU, keeping the multipliers:

```rust,ignore
{{#include ../../../crates/linalg_reference/src/lu.rs:doolittle}}
```

The production recursion:

```rust,ignore
{{#include ../../../crates/linalg/src/lu.rs:recursive_lu}}
```

- All production routines work for `f64` and `Complex<f64>` through the `Scalar` trait.
- `solve` factorises and solves; `factorize` returns an `LuFactorization` to reuse; `solve_t` and `solve_h`
  solve with $A^{\mathsf T}$ and $A^{\mathsf H}$.
- The inverse is computed as $U^{-1}L^{-1}P$, using the triangularity of $L^{-1}$ to save a third of the work, as
  LAPACK's `getri` does.

## In practice

```rust,ignore
{{#include ../../../crates/linalg_reference/examples/linalg_lu.rs}}
```

**Strengths**

- The general-purpose dense solver: stable in practice, $\tfrac23n^3$ flops, near peak speed when blocked.
- The factorisation is reusable, and gives determinants.

**Weaknesses**

- Not guaranteed stable: exponential growth is possible, though almost never met.
- Dense storage and $O(n^3)$ cost: large sparse systems need sparse or iterative methods.
- Ignores structure: use Cholesky for symmetric positive definite matrices, at half the cost.

## Exercises

1. Count the flops of Gaussian elimination and show that the leading term is $\tfrac23n^3$.
2. Show that the inverse of a unit lower triangular matrix is unit lower triangular, and that the
   multipliers of elimination are the entries of $L$.
3. Prove that partial pivoting gives growth factor at most $2^{n-1}$, and that Wilkinson's matrix attains it.
4. Solve $H\mathbf x = \mathbf b$ for the Hilbert matrix of order 10 with $\mathbf b = H\mathbf 1$, perturb one entry of $\mathbf b$ by $10^{-10}$, and
   compare the change in $\mathbf x$ with $\kappa(H)$.
5. Use the factorisation to compute $\det A$ for a random $200 \times 200$ matrix scaled by 10. Why can the determinant
   overflow even when the matrix is well conditioned, and how would you compute $\ln|\det A|$ instead?

## Further reading

- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013,
  chapter 3.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002, chapter 9.
- L. N. Trefethen and D. Bau, *Numerical Linear Algebra*, SIAM, 1997, lectures 20–23.
- S. Toledo, "Locality of reference in LU decomposition with partial pivoting", *SIAM Journal on Matrix
  Analysis and Applications* 18 (1997) 1065–1081.
- J. F. Grcar, "Mathematicians of Gaussian elimination", *Notices of the American Mathematical Society*
  58 (2011) 782–792.
