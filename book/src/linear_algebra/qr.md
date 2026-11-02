# QR factorisation

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Factor an $m \times n$ matrix ($m \ge n$) as $A = QR$, with $Q$ having orthonormal columns and $R$ upper triangular; with column pivoting, $AP = QR$ reveals the numerical rank |
| **Devised by** | Orthogonalisation: Pierre-Simon Laplace (1812), Jørgen Pedersen Gram (1883), Erhard Schmidt (1907); reflections: Alston Householder (1958); rotations: Wallace Givens (1958); column pivoting: Peter Businger and Gene Golub (1965); the stability of modified Gram-Schmidt: Åke Björck (1967) |
| **Accuracy** | Householder: $Q$ orthogonal to machine precision, backward stable. Modified Gram-Schmidt: loss of orthogonality $\approx \kappa\varepsilon$. Classical Gram-Schmidt: $\approx \kappa^2\varepsilon$ |
| **Cost** | $2mn^2 - \tfrac23n^3$ flops for Householder ($2mn^2$ for Gram-Schmidt) |
| **API** | Production: used inside [`LeastSquaresSvd`](../../api/lib_jsl_linalg/trait.LeastSquaresSvd.html) and the SVD (crate-internal). Reference: [`qr`](../../api/lib_jsl_linalg_reference/qr/index.html) (`classical_gram_schmidt`, `modified_gram_schmidt`, `householder`, `orthogonality_error`) |

</div>

## Overview

The **QR factorisation** writes a matrix as an orthogonal factor times a triangular one. Its
columns of $Q$ are an orthonormal basis for the column space of $A$, built so that the first $k$ columns
of $Q$ span the first $k$ columns of $A$. It is the tool of choice for linear least squares, the first step
of the [SVD](svd.md) and of eigenvalue algorithms, and the core of the QR algorithm for eigenvalues.

Orthogonal matrices are the perfectly conditioned transformations: they preserve lengths, so they
never amplify errors. That is the source of QR's stability, but only if the computed $Q$ really is
orthogonal. The three classical algorithms differ precisely there: the **Gram-Schmidt** process
orthogonalises the columns one by one and, in floating point, gradually loses orthogonality;
**Householder reflections** build $Q$ as a product of exactly orthogonal matrices and keep it.

## History

The orthogonalisation of a set of functions or vectors appears in Laplace's work on least squares
(1812); Jørgen Pedersen Gram (1883) used it for series expansions, and Erhard Schmidt (1907) stated
the process for functions in his work on integral equations. For computation, Alston Householder
(1958) introduced the reflections that bear his name and showed how to triangularise a matrix with
them, and Wallace Givens (1958) used plane rotations. Gene Golub (1965) made Householder QR the
standard method for least squares, and with Peter Businger added column pivoting to reveal rank.
Åke Björck (1967) showed that the modified Gram-Schmidt process, though it loses orthogonality, gives
accurate least-squares solutions. Robert Schreiber and Charles Van Loan (1989) devised the compact
WY representation that lets reflections be applied in blocks, at matrix-multiply speed.

## The idea

### Gram-Schmidt orthogonalisation

**Classical Gram-Schmidt** subtracts from each column $\mathbf a_j$ its components along the previous $\mathbf q_i$, computed
from the original column, and normalises:

$$
r_{ij} = \mathbf q_i^{\mathsf T}\mathbf a_j \ (i < j), \qquad \mathbf v_j = \mathbf a_j - \sum_{i<j}r_{ij}\mathbf q_i, \qquad r_{jj} = \|\mathbf v_j\|, \qquad \mathbf q_j = \mathbf v_j/r_{jj} .
$$

**Modified Gram-Schmidt** computes each $r_{ij}$ from the partially orthogonalised vector instead, subtracting
one component at a time. In exact arithmetic the two are identical. In floating point, rounding errors in
the early $\mathbf q_i$ contaminate the later ones, and the computed $Q$ loses orthogonality: by about $\kappa^2\varepsilon$ for classical
Gram-Schmidt and $\kappa\varepsilon$ for modified, where $\kappa$ is the condition number of $A$. Reorthogonalising (running the
process twice) restores orthogonality.

### Householder reflections

A **Householder reflection** $H = I - 2\mathbf v\mathbf v^{\mathsf T}$, with $\|\mathbf v\| = 1$, reflects vectors in the hyperplane orthogonal to $\mathbf v$.
It is symmetric and orthogonal, and for any vector $\mathbf x$ the choice $\mathbf v \propto \mathbf x + \operatorname{sign}(x_1)\|\mathbf x\|\mathbf e_1$ maps $\mathbf x$ onto
$-\operatorname{sign}(x_1)\|\mathbf x\|\mathbf e_1$. The sign is chosen so that the first component of $\mathbf v$ involves no cancellation. Applying
reflections $H_1, H_2, \ldots, H_n$ that zero the subdiagonal of each column in turn gives

$$
H_n\cdots H_2H_1A = R, \qquad Q = H_1H_2\cdots H_n .
$$

Each reflection is applied without forming it, as $H\mathbf y = \mathbf y - 2(\mathbf v^{\mathsf T}\mathbf y)\mathbf v$, and $Q$ is usually kept as the list of
vectors $\mathbf v_k$. Because every factor is orthogonal to working precision, so is the product: the computed $Q$ is
orthogonal to about $\varepsilon$ regardless of the conditioning of $A$.

![Loss of orthogonality](../generated/linear_algebra/orthogonality.svg)

The figure shows the three methods on $60 \times 20$ matrices with singular values spread between 1 and $1/\kappa$. Classical
Gram-Schmidt has lost all orthogonality by $\kappa = 10^8$, modified Gram-Schmidt follows $\kappa\varepsilon$, and Householder stays at
the level of rounding throughout. All three reproduce $A = QR$ to rounding error: the difference is
entirely in $Q$.

### Column pivoting

**QR with column pivoting**, $AP = QR$, chooses at each step the remaining column of largest norm. The diagonal of
$R$ then decreases, $|r_{11}| \ge |r_{22}| \ge \cdots$, and a sharp drop reveals the **numerical rank**: if $A$ is close to a matrix of
rank $r$, then $|r_{r+1,r+1}|$ is small. The column norms are **downdated** after each step rather than recomputed,
with a recomputation when cancellation makes the downdated value unreliable (the safeguard used in LAPACK's
`geqp3`). The production library uses pivoted QR for [least squares](least-squares.md) and to precondition
the [Jacobi SVD](svd.md#one-sided-jacobi).

## Algorithm

**Householder QR**: for $k = 1, \ldots, n$: form $\mathbf v_k$ from column $k$ (rows $k$ to $m$); apply $H_k$ to columns $k$ to $n$; store
$\mathbf v_k$. To form $Q$ explicitly, apply the reflections in reverse order to the first $n$ columns of the identity.

**With column pivoting**: before step $k$, swap the remaining column of largest (downdated) norm into position $k$.

## Implementation

Modified Gram-Schmidt (reference):

```rust,ignore
{{#include ../../../crates/linalg_reference/src/qr.rs:mgs}}
```

Householder QR (reference):

```rust,ignore
{{#include ../../../crates/linalg_reference/src/qr.rs:householder}}
```

Pivoted Householder QR (production), for real and complex matrices:

```rust,ignore
{{#include ../../../crates/linalg/src/qr.rs:householder_pivoted}}
```

- For complex matrices the reflection maps $\mathbf x$ onto $-e^{i\theta}\|\mathbf x\|\mathbf e_1$, where $e^{i\theta}$ is the phase of $x_1$.
- When an explicit orthogonal factor is needed (in the SVD and both eigensolvers), the library accumulates blocks of
  reflections in the **compact WY form** $I - VTV^{\mathsf H}$ and applies each block with three matrix multiplications.
- The production QR is not exposed directly; least squares and the SVD use it. Its unpivoted, blocked
  form for least squares is tracked in [issue #11](https://github.com/jwkunz/lib_jsl/issues/11).

## In practice

```rust,ignore
{{#include ../../../crates/linalg_reference/examples/linalg_qr.rs}}
```

**Strengths**

- Householder QR is backward stable and keeps $Q$ orthogonal for any conditioning.
- Column pivoting reveals numerical rank cheaply.

**Weaknesses**

- Twice the cost of LU for square matrices.
- Gram-Schmidt needs reorthogonalisation to be reliable (but is the natural choice when columns arrive one at a time,
  as in Krylov methods).

## Exercises

1. Show that $H = I - 2\mathbf v\mathbf v^{\mathsf T}$ with $\|\mathbf v\| = 1$ is symmetric, orthogonal and an involution ($H^2 = I$).
2. Show that classical and modified Gram-Schmidt compute the same $Q$ and $R$ in exact arithmetic.
3. Apply classical Gram-Schmidt to $\begin{pmatrix}1 & 1 & 1\\ \varepsilon & 0 & 0\\ 0 & \varepsilon & 0\\ 0 & 0 & \varepsilon\end{pmatrix}$ with $\varepsilon^2$ below rounding (Läuchli's matrix) and
   compute $\mathbf q_2^{\mathsf T}\mathbf q_3$ by hand.
4. Run classical Gram-Schmidt twice ("twice is enough") and measure the loss of orthogonality as in the figure.
5. Show that column pivoting gives $|r_{11}| \ge |r_{22}| \ge \cdots \ge |r_{nn}|$.

## Further reading

- A. S. Householder, "Unitary triangularization of a nonsymmetric matrix", *Journal of the ACM* 5 (1958)
  339–342.
- Å. Björck, "Solving linear least squares problems by Gram-Schmidt orthogonalization", *BIT* 7 (1967)
  1–21.
- R. Schreiber and C. Van Loan, "A storage-efficient WY representation for products of Householder
  transformations", *SIAM Journal on Scientific and Statistical Computing* 10 (1989) 53–57.
- L. N. Trefethen and D. Bau, *Numerical Linear Algebra*, SIAM, 1997, lectures 7–10 and 16.
- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013,
  chapter 5.
