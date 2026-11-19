# Further factorisations: pivoted QR, RQ, Hessenberg, real Schur, LDLᵀ and polar

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Factorise a matrix into structured pieces: $AP = QR$ with column pivoting, $A = RQ$, $A = QHQ^{\mathsf H}$ (Hessenberg), $A = ZTZ^{\mathsf T}$ (real Schur), $A = LDL^{\mathsf H}$ for symmetric indefinite $A$, $A = UP$ (polar); and compute principal angles between subspaces and general matrix functions $f(A)$ |
| **Devised by** | Alston Householder (1958), reflections; Peter Businger and Gene Golub (1965), column pivoting; James Bunch and Linda Kaufman (1977), symmetric indefinite pivoting; Issai Schur (1909); Beresford Parlett (1976), the Schur-Parlett recurrence; Camille Jordan (1875) and Åke Björck and Gene Golub (1973), principal angles; Léon Autonne (1902), the polar decomposition |
| **Accuracy** | Agrees with SciPy (LAPACK) entry by entry to $6 \times 10^{-15}$ relative, factor for factor and pivot for pivot |
| **Cost** | QR, RQ: $2mn^2 - \tfrac23n^3$ operations, plus as much again to form $Q$; Hessenberg: $\tfrac{10}3n^3$, plus $\tfrac43n^3$ for $Q$; real Schur: about $25n^3$; $LDL^{\mathsf T}$: $\tfrac13n^3$; polar and angles: one or two SVDs; `funm`: a Schur form and $\tfrac13n^3$ |
| **Read first** | [QR factorisation](qr.md), [Symmetric and general eigensolvers](eigensolvers.md), [The singular value decomposition](svd.md) |
| **API** | [`qr`](../../api/lib_jsl_linalg/fn.qr.html), [`rq`](../../api/lib_jsl_linalg/fn.rq.html), [`hessenberg`](../../api/lib_jsl_linalg/fn.hessenberg.html), [`schur_real`](../../api/lib_jsl_linalg/fn.schur_real.html), [`ldl`](../../api/lib_jsl_linalg/fn.ldl.html), [`polar`](../../api/lib_jsl_linalg/fn.polar.html), [`subspace_angles`](../../api/lib_jsl_linalg/fn.subspace_angles.html), [`funm`](../../api/lib_jsl_linalg/fn.funm.html) |

</div>

## Overview

The earlier entries of this chapter use these factorisations internally: Householder QR for least squares, the Hessenberg and Schur forms inside the eigensolvers, the SVD
for ranks and pseudoinverses. This entry exposes them, with SciPy's interfaces and conventions:

- **QR with column pivoting**, $AP = QR$, chooses at each step the column of largest remaining norm. The diagonal of $R$ then decreases, and a gap in it reveals the
  numerical rank more cheaply than the SVD.
- **RQ**, $A = RQ$ with $R$ upper triangular, is QR applied from the other side. It appears in camera calibration, where a $3 \times 3$ projection matrix splits into intrinsic
  (triangular) and rotation factors, and in generalised eigenvalue algorithms.
- The **Hessenberg form** $A = QHQ^{\mathsf H}$, zero below the first subdiagonal, is the first stage of every dense eigenvalue computation. It also makes
  $(A - sI)^{-1}\mathbf b$ cheap for many shifts $s$, as in frequency responses: $O(n^2)$ per shift instead of $O(n^3)$.
- The **real Schur form** $A = ZTZ^{\mathsf T}$ keeps real matrices real. $T$ is block upper triangular, with a $2 \times 2$ block for each complex conjugate pair of eigenvalues.
- The **Bunch-Kaufman factorisation** $A = LDL^{\mathsf T}$ is to symmetric indefinite matrices what Cholesky is to definite ones. It costs half as much as LU and preserves symmetry,
  and its $1 \times 1$ and $2 \times 2$ pivots keep it stable where a plain $LDL^{\mathsf T}$ would divide by small or zero diagonal entries.
- The **polar decomposition** $A = UP$ is the matrix analogue of $z = e^{i\theta}r$: $U$ is the orthogonal matrix nearest to $A$, and $P$ is a symmetric positive semi-definite "stretch".
- **Principal angles** measure how far apart two subspaces are. **`funm`** evaluates any analytic function of a matrix.

## History

Alston Householder (1958) introduced the reflections that bear his name, and they replaced Gram-Schmidt and Givens rotations in most dense codes. Peter Businger and Gene Golub
(1965) added column pivoting to make QR reveal rank. Issai Schur (1909) proved that every square matrix is unitarily similar to a triangular one, and the real
quasi-triangular variant became the output of Francis's QR algorithm (1961) in EISPACK (1974) and LAPACK (1992). LAPACK's `lanv2` (1990s) fixed a standard form for the
$2 \times 2$ blocks, so that results are reproducible.

For symmetric indefinite systems, James Bunch and Beresford Parlett (1971) showed that $2 \times 2$ pivots make symmetric elimination stable. Bunch and Linda Kaufman (1977)
found a strategy that examines only two columns per step, with growth factor bounded by $2.57^{n-1}$, and it is what LAPACK's `sytrf` uses. Léon Autonne (1902) stated the polar
decomposition for matrices. Camille Jordan (1875) defined principal angles, and Åke Björck and Gene Golub (1973) gave the SVD algorithm that computes them. Beresford Parlett (1976) derived the
recurrence for $f(T)$ on a triangular $T$ that `funm` uses, and Philip Davies and Nicholas Higham (2003) blocked it for close eigenvalues.

## The idea

### Householder reflectors and LAPACK's sign convention

A Householder reflector $H = I - \tau\mathbf v\mathbf v^{\mathsf H}$ maps a vector $(\alpha, \mathbf x)$ onto $(\beta, \mathbf 0)$. For real data, $H$ is symmetric and orthogonal with
$\tau = 2/\|\mathbf v\|^2$, and $\beta = \pm\|(\alpha, \mathbf x)\|$. The sign is a choice: taking $\beta$ with the sign opposite to $\alpha$ avoids cancellation when forming $\mathbf v = (\alpha - \beta, \mathbf x)$.
LAPACK's `larfg` makes this choice, normalises $v_1 = 1$, and for complex data makes $\beta$ real, using a non-Hermitian $H$ with complex $\tau$:

$$
\beta = -\operatorname{sign}(\operatorname{Re}\alpha)\,\|(\alpha, \mathbf x)\|, \qquad \tau = \frac{\beta - \alpha}{\beta}, \qquad \mathbf v = \Bigl(1, \frac{\mathbf x}{\alpha - \beta}\Bigr).
$$

Every factorisation built from such reflectors is unique once these choices are fixed. The library makes exactly LAPACK's choices, so its $Q$ and $R$ agree with SciPy's entry by entry, not just up to the
signs of columns. For a complex matrix, even the last step, which has nothing left to eliminate, still reflects, so as to make the last diagonal entry of $R$ (or subdiagonal entry of $H$) real.

### QR with column pivoting

At step $k$, the column of the trailing submatrix with the largest norm is swapped into position $k$ before the reflector is computed. The norms are not recomputed from scratch:
removing the first entry of each column updates them, $\|\mathbf a_j'\|^2 = \|\mathbf a_j\|^2 - |r_{kj}|^2$. When cancellation has eaten most of a norm's digits, the norm is recomputed
(LAPACK's test: when the updated norm has fallen below $\sqrt\varepsilon$ times its last computed value). The pivoting gives $|r_{11}| \ge |r_{22}| \ge \dots$. For a matrix of rank $r$, $|r_{kk}|$ drops to the level of
rounding errors at $k = r + 1$.

### RQ

RQ processes rows from the bottom. The last row is reflected onto its last entry, by a reflector applied from the right; then the row above onto its next-to-last entry, and so on.
After $k = \min(m, n)$ steps, $AH_k\cdots H_1 = R$, so $A = RQ$ with $Q = H_1^{\mathsf H}\cdots H_k^{\mathsf H}$. For a complex matrix the reflector acts on the conjugated row, as in LAPACK's `gerq2`.

### The Hessenberg form

Reflector $i$ zeros column $i$ below the subdiagonal. Because it is applied on both sides, $A \leftarrow H_i^{\mathsf H}AH_i$, the zeros already made are not destroyed, so the similarity
preserves eigenvalues. $n - 2$ reflectors (one more for complex data) reach upper Hessenberg form, $A = QHQ^{\mathsf H}$. A matrix that is the identity plus a low-rank term, for
example, has a Hessenberg form with zero subdiagonal entries early on. The remaining reflectors are then determined by rounding errors, and two correct implementations may differ there.

### The real Schur form and its standard 2 × 2 blocks

For a real matrix, the QR algorithm with double shifts converges to a quasi-triangular $T$ with $1 \times 1$ blocks (real eigenvalues) and $2 \times 2$ blocks (complex pairs). A $2 \times 2$
block can be rotated into infinitely many shapes, so LAPACK fixes one with `lanv2`. A rotation makes the two diagonal entries equal, and then the off-diagonal entries have opposite
signs, so that the eigenvalues are

$$
\begin{pmatrix} a & b \\ c & a \end{pmatrix}, \quad bc < 0: \qquad \lambda = a \pm i\sqrt{-bc} .
$$

If the block's eigenvalues turn out to be real, the same routine triangularises it instead. The library standardises the blocks of its Schur form with a port of `lanv2`,
updating the rest of $T$ and $Z$ with the same rotation.

### Bunch-Kaufman pivoting

Symmetric Gaussian elimination, $A = LDL^{\mathsf T}$ with $D$ diagonal, fails on $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, and is unstable whenever a pivot is small. Row exchanges
would destroy the symmetry. Bunch and Kaufman allow $2 \times 2$ blocks in $D$ instead, and choose at each step between three options, with $\alpha = (1 + \sqrt{17})/8 \approx 0.64$.
Let $\omega_1$ be the largest off-diagonal entry in column $k$, in row $r$, and $\omega_r$ the largest off-diagonal entry in row and column $r$. Then:

- if $|a_{kk}| \ge \alpha\omega_1$, or $|a_{kk}|\,\omega_r \ge \alpha\omega_1^2$: a $1 \times 1$ pivot $a_{kk}$;
- else if $|a_{rr}| \ge \alpha\omega_r$: swap $r$ into position $k$ and take a $1 \times 1$ pivot;
- else: swap $r$ into position $k + 1$, and take the $2 \times 2$ pivot $\begin{pmatrix} a_{kk} & a_{rk} \\ a_{rk} & a_{rr} \end{pmatrix}$.

This value of $\alpha$ balances the element growth of two $1 \times 1$ steps against one $2 \times 2$ step. The cost stays at $n^3/3$, half that of LU.
SciPy returns $L$ with its rows permuted back to the original order, so that $A = LDL^{\mathsf H}$ holds directly, together with the permutation `perm` that makes `L[perm]` triangular. The library does the same.

### Polar decomposition and principal angles

From the SVD $A = W\Sigma V^{\mathsf H}$: $U = WV^{\mathsf H}$ and $P = V\Sigma V^{\mathsf H}$ (or $W\Sigma W^{\mathsf H}$ for $A = PU$). $U$ is the nearest matrix with orthonormal columns to $A$, in both the
Frobenius and spectral norms. That is the solution of the orthogonal Procrustes problem.

The principal angles $\theta_1 \le \theta_2 \le \dots$ between the column spaces of $A$ and $B$ have cosines equal to the singular values of $Q_A^{\mathsf H}Q_B$, for orthonormal bases $Q_A$, $Q_B$. Cosines
near 1 lose the small angles: $\cos\theta = 1 - \theta^2/2$ cannot resolve $\theta < 10^{-8}$. So, as Knyazev and Argentati (2002) recommend, angles whose cosine exceeds $1/\sqrt2$ are computed from their
sines instead, the singular values of $Q_B - Q_AQ_A^{\mathsf H}Q_B$.

### Matrix functions by Schur-Parlett

If $A = ZTZ^{\mathsf H}$ is a complex Schur form, then $f(A) = Zf(T)Z^{\mathsf H}$, and $F = f(T)$ is upper triangular with $F_{ii} = f(T_{ii})$. Since $F$ commutes with $T$, equating the $(i, j)$ entries of
$TF = FT$ gives the rest, one superdiagonal at a time:

$$
F_{ij} = \frac{T_{ij}(F_{jj} - F_{ii}) + \sum_{i<k<j}(T_{ik}F_{kj} - F_{ik}T_{kj})}{T_{jj} - T_{ii}} .
$$

Close eigenvalues make the denominators small, and the error grows like $\|T\|/\min|T_{jj} - T_{ii}|$. `funm` returns that estimate, as SciPy's does. For $e^A$, $\sqrt A$ and $\log A$, the
dedicated `expm`, `sqrtm` and `logm` are more accurate.

## Algorithm

**QR with column pivoting** (LAPACK's `laqp2`): with column norms $\nu_j$, for $k = 1, \dots, \min(m, n)$:

1. Swap the column $j \ge k$ of largest $\nu_j$ (the first, on ties) into position $k$.
2. Compute the reflector for column $k$ below the diagonal (`larfg`), and apply $H_k^{\mathsf H}$ to the remaining columns.
3. Downdate $\nu_j \leftarrow \nu_j\sqrt{1 - (|r_{kj}|/\nu_j)^2}$. Recompute $\nu_j$ from the column when this has lost more than half of the digits since it was last computed.

$Q = H_1H_2\cdots H_k$ is formed by applying the reflectors to the identity in reverse order.

**Bunch-Kaufman** (`sytf2`, lower triangle): for $k = 1, 2, \dots$, choose a $1 \times 1$ or $2 \times 2$ pivot as above. Swap rows and columns in the trailing submatrix, and eliminate:
$A_{22} \leftarrow A_{22} - L_{21}D_1L_{21}^{\mathsf H}$, with $L_{21} = A_{21}D_1^{-1}$. Then advance $k$ by the block size. Record the interchanges as LAPACK does (positive for $1 \times 1$ pivots, negative for $2 \times 2$),
and unpack them into $L$, $D$ and `perm` as SciPy does.

**Real Schur form**: reduce to Hessenberg form, run the double-shift QR algorithm, and standardise each $2 \times 2$ block with `lanv2`'s rotation, applied to the rows and columns of $T$ beyond the block and to $Z$.

## Implementation

The reflector:

```rust,ignore
{{#include ../../../crates/linalg/src/factorizations.rs:larfg}}
```

QR with optional pivoting, and RQ:

```rust,ignore
{{#include ../../../crates/linalg/src/factorizations.rs:qr}}
```

```rust,ignore
{{#include ../../../crates/linalg/src/factorizations.rs:rq}}
```

The standard form of a $2 \times 2$ block:

```rust,ignore
{{#include ../../../crates/linalg/src/factorizations.rs:lanv2}}
```

Principal angles and the Schur-Parlett recurrence:

```rust,ignore
{{#include ../../../crates/linalg/src/factorizations.rs:subspace_angles}}
```

```rust,ignore
{{#include ../../../crates/linalg/src/factorizations.rs:funm}}
```

- All the functions take real or complex matrices, except `schur_real`, which is for real ones. The complex Schur form is [`schur_complex`](matrix-equations.md).
- The Bunch-Kaufman code (`ldl` in `crates/linalg/src/factorizations.rs`) has the lower and upper variants, Hermitian and complex symmetric. Pivot sizes are compared with
  $|\operatorname{Re}z| + |\operatorname{Im}z|$ for complex entries, as LAPACK does.
- The routines are unblocked, as LAPACK is for matrices up to a few dozen columns, where it gives the same results as its blocked code. For large matrices LAPACK's blocked versions are faster, and
  differ in rounding.
- `qr` returns the permutation as a list. Column $j$ of $AP$ is column `permutation[j]` of $A$.

## In practice

The references are from SciPy 1.17.1, written by `book/generator/data/make_decomp_refs.py`. The test matrices are built from a formula, $\sin(1.3i + 0.7j^2 + 0.37ij + s) + 2\delta_{ij}$, in real and
complex, tall, square and wide shapes:

<p class="caption" id="tab-7-14"><strong>Table 7.14.</strong> Further factorisations against SciPy.</p>

{{#include ../generated/linear_algebra/decomp_scipy.md}}

Every factor agrees with SciPy's to rounding error. That includes the pivot sequences of QR and of Bunch-Kaufman, whose tests use 13 $2 \times 2$ pivots. For a rank-deficient
matrix, the columns of $Q$ beyond the rank, and the order in which the remaining columns are pivoted, depend on rounding errors of size $10^{-16}$, and they differ between any two implementations.
The comparison is restricted there to what the matrix determines: $R$ in QR, $P$ in the polar decomposition, and the leading pivots. The Schur forms are compared by their
eigenvalues and by reconstruction, since the order of the eigenvalues along the diagonal depends on the course of the QR iteration. The angles near $10^{-7}$ agree to $10^{-8}$ relative,
which is the accuracy $\varepsilon/\theta$ to which the data determine them.

A first test matrix built from $\sin(1.3i + 0.7j^2 + s)$ alone showed what can go wrong. That formula has rank 2, so with $2I$ added every Hessenberg form had zero subdiagonal entries after two steps, and every RQ of a tall matrix
was degenerate. The two libraries agreed on the reconstruction but not on the factors. The factors were not unique, so neither was wrong.

**Strengths**

- Identical conventions to SciPy and LAPACK, including signs, pivots and the standard form of Schur blocks, so results can be compared, or ported, entry by entry.
- Real and complex matrices, and both triangles for `ldl`.
- Small principal angles accurate to relative precision, from the sines.

**Weaknesses**

- Unblocked algorithms: for matrices with hundreds of rows, LAPACK's blocked versions are several times faster.
- `funm` divides by eigenvalue differences, so it fails for close or repeated eigenvalues (the error estimate says so). The block Schur-Parlett algorithm of Davies and Higham is not implemented.
- `schur_real` has no option to sort the eigenvalues (SciPy's `sort`), and there is no QZ for pencils yet.

## Exercises

1. Show that $H = I - \tau\mathbf v\mathbf v^{\mathsf H}$ with $\tau$ and $\mathbf v$ from `larfg` satisfies $H^{\mathsf H}(\alpha, \mathbf x) = (\beta, \mathbf 0)$, and that $H$ is unitary. When is it Hermitian?
2. Factorise $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ by Bunch-Kaufman. Why does $LDL^{\mathsf T}$ with diagonal $D$ fail?
3. Show that the eigenvalues of $\begin{pmatrix} a & b \\ c & a \end{pmatrix}$ are $a \pm \sqrt{bc}$, and hence complex when $bc < 0$.
4. Use pivoted QR to find the rank of the $6 \times 6$ rank-3 matrix of the test table. How large is $|r_{44}|/|r_{11}|$?
5. Show that $U$ in the polar decomposition minimises $\|A - U\|_F$ over matrices with orthonormal columns.
6. Compute the principal angles between the planes $z = 0$ and $z = \epsilon y$ in $\mathbb R^3$, for $\epsilon = 10^{-10}$. What would the cosine formula alone give?

<details>
<summary>Hints and answers</summary>

1. $\mathbf v^{\mathsf H}(\alpha, \mathbf x) = \alpha + \|\mathbf x\|^2/(\bar\alpha - \beta)$, and with $\|\mathbf x\|^2 = \beta^2 - |\alpha|^2$ this is $\beta(\beta - \alpha)/(\bar\alpha - \beta)$. Then $H^{\mathsf H}(\alpha, \mathbf x) = (\alpha, \mathbf x) - \bar\tau\,\mathbf v\,\mathbf v^{\mathsf H}(\alpha, \mathbf x)$, and $\bar\tau\,\mathbf v^{\mathsf H}(\alpha, \mathbf x) = \frac{\beta - \bar\alpha}{\beta}\cdot\frac{\beta(\beta - \alpha)}{\bar\alpha - \beta} = \alpha - \beta$, which leaves $(\beta, \mathbf 0)$. Unitarity: $H^{\mathsf H}H = I$ reduces to $\tau + \bar\tau = |\tau|^2\|\mathbf v\|^2$, which holds for these $\tau$ and $\mathbf v$. $H$ is Hermitian when $\tau$ is real, in particular for all real data.
2. The diagonal is zero, so the first pivot would be zero. Bunch-Kaufman finds $|a_{11}| = 0 < \alpha\omega_1 = \alpha$, and $|a_{22}| = 0 < \alpha\omega_r$, so it takes the whole matrix as one $2 \times 2$ block: $L = I$, $D = A$.
3. The characteristic polynomial is $(a - \lambda)^2 - bc$.
4. $|r_{22}|/|r_{11}| = 0.40$, $|r_{33}|/|r_{11}| = 0.19$ and $|r_{44}|/|r_{11}| = 5 \times 10^{-17}$: the rank is 3. The pivots beyond the rank depend on these rounding-level norms.
5. $\|A - U\|_F^2 = \|A\|_F^2 + n - 2\operatorname{Re}\operatorname{tr}(U^{\mathsf H}A)$. With $A = W\Sigma V^{\mathsf H}$, $\operatorname{Re}\operatorname{tr}(U^{\mathsf H}W\Sigma V^{\mathsf H}) = \operatorname{Re}\operatorname{tr}(\Sigma V^{\mathsf H}U^{\mathsf H}W) \le \sum\sigma_i$, with equality for $U = WV^{\mathsf H}$.
6. The angles are 0 (along the $x$ axis) and $\arctan\epsilon \approx 10^{-10}$. The cosine of the second is $1 - 5 \times 10^{-21}$, which rounds to 1, so the cosine formula gives 0. The sine formula gives $10^{-10}$ to full precision.

</details>

## Further reading

- G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013: chapters 4 (symmetric indefinite systems), 5 (QR with pivoting), 6 (principal angles), 7 (Hessenberg and Schur forms) and 9 (Schur-Parlett).
- P. Businger and G. H. Golub, "Linear least squares solutions by Householder transformations", *Numerische Mathematik* 7 (1965) 269–276.
- J. R. Bunch and L. Kaufman, "Some stable methods for calculating inertia and solving symmetric linear systems", *Mathematics of Computation* 31 (1977) 163–179.
- B. N. Parlett, "A recurrence among the elements of functions of triangular matrices", *Linear Algebra and its Applications* 14 (1976) 117–121.
- Å. Björck and G. H. Golub, "Numerical methods for computing angles between linear subspaces", *Mathematics of Computation* 27 (1973) 579–594.
- A. V. Knyazev and M. E. Argentati, "Principal angles between subspaces in an A-based scalar product", *SIAM Journal on Scientific Computing* 23 (2002) 2008–2040.
- N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008: chapter 8 (the polar decomposition) and chapter 9 (Schur-Parlett).
- P. I. Davies and N. J. Higham, "A Schur-Parlett algorithm for computing matrix functions", *SIAM Journal on Matrix Analysis and Applications* 25 (2003) 464–485.
- E. Anderson et al., *LAPACK Users' Guide*, 3rd ed., SIAM, 1999.
