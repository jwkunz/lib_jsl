# Polynomial roots

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Find all the roots, real and complex, of a polynomial $p(x) = a_0 x^n + a_1 x^{n-1} + \cdots + a_n$ |
| **Devised by** | Formulas up to degree 4: Tartaglia, Cardano and Ferrari (1545); none beyond: Ruffini (1799) and Abel (1824). Companion matrix eigenvalues: Cleve Moler's `roots` in MATLAB (1970s), shown backward stable by Alan Edelman and H. Murakami (1995); balancing: Beresford Parlett and Christian Reinsch (1969). Ill-conditioning: James Wilkinson (1959) |
| **Accuracy** | Backward stable: the roots of a polynomial with nearby coefficients. The forward error depends on the conditioning of the roots, which for clustered or multiple roots can be very poor |
| **Cost** | $O(n^3)$ operations and $O(n^2)$ memory for degree $n$ |
| **API** | [`polynomial_roots`](../../api/lib_jsl_root_finding/polynomial_roots/fn.polynomial_roots.html) (real roots), [`polynomial_roots_complex`](../../api/lib_jsl_root_finding/polynomial_roots/fn.polynomial_roots_complex.html) (all roots) |

</div>

## Overview

Polynomials are the one class of functions whose roots can all be found at once, reliably, without
starting guesses or brackets. A polynomial of degree $n$ has exactly $n$ complex roots, counted with
multiplicity (the fundamental theorem of algebra), and they are the **eigenvalues** of an $n \times n$
matrix built from the coefficients, the **companion matrix**. Computing them with the
[QR algorithm](../../api/lib_jsl_linalg/index.html) is the method of MATLAB's and NumPy's `roots`.

Polynomial roots are needed throughout science and engineering: the poles and zeros of filters and
control systems, the eigenvalues of small matrices from their characteristic polynomials, the nodes
of Gaussian quadrature, the intersections of curves in computer graphics. The library's
[Prony's method](../interpolation/trigonometric.md#pronys-method) finds the frequencies of a signal
as polynomial roots.

They also carry the most famous warning in numerical analysis: the roots of a polynomial can depend
extremely sensitively on its coefficients.

## History

The quadratic formula was known in Babylon; the cubic and quartic were solved by radicals in
sixteenth-century Italy by Scipione del Ferro, Niccolò Tartaglia and Lodovico Ferrari, and published by
Gerolamo Cardano in *Ars Magna* (1545). Paolo Ruffini (1799) and Niels Henrik Abel (1824) proved that no
such formula exists for the general quintic, and Évariste Galois explained why. From then on, roots
of polynomials of high degree had to be computed numerically, by methods from Newton's and Horner's
to Bernoulli's, Graeffe's and, in 1970, the Jenkins-Traub algorithm.

James Wilkinson, testing a new program on the Pilot ACE computer in the 1950s, chose the polynomial
with roots $1, 2, \ldots, 20$ as an easy test, and found it impossible. His analysis (1959), which he later
called "the most traumatic experience in my career as a numerical analyst", showed that the
difficulty lay in the polynomial, not the program, and established the ideas of conditioning and
backward error. With the QR algorithm for eigenvalues (1961), it became natural to compute
polynomial roots as eigenvalues, the approach Cleve Moler used in MATLAB; Alan Edelman and
H. Murakami (1995) proved that it is backward stable once the matrix is **balanced**.

## The idea

### The companion matrix

The monic polynomial $z^n + c_1 z^{n-1} + \cdots + c_n$, with $c_i = a_i/a_0$, is the characteristic
polynomial of

$$
C = \begin{pmatrix}
-c_1 & -c_2 & \cdots & -c_{n-1} & -c_n \\
1 & 0 & \cdots & 0 & 0 \\
0 & 1 & \cdots & 0 & 0 \\
\vdots & & \ddots & & \vdots \\
0 & 0 & \cdots & 1 & 0
\end{pmatrix},
$$

as expanding $\det(zI - C)$ along the first row shows. Its eigenvalues are the roots. The matrix is
already in upper Hessenberg form, the form the QR algorithm works on, and the eigenvalue
computation costs $O(n^3)$ operations. Specialised structured QR algorithms reduce this to $O(n^2)$.

### Balancing

The entries of $C$ can range over many orders of magnitude: the coefficients of Wilkinson's
polynomial run from 1 to $2.4 \times 10^{18}$. The QR algorithm's rounding errors are proportional to the
norm of the matrix, so the small entries are lost. **Balancing** replaces $C$ by $D^{-1} C D$ for a
diagonal $D$ chosen to make each row and the matching column similar in size. The eigenvalues are
unchanged; with the entries of $D$ powers of 2, not even rounding is introduced. Without balancing,
this library's eigenvalue solver returned 18 zeros and a wild complex pair for Wilkinson's
polynomial; with it, all 20 roots to within $4 \times 10^{-3}$.

### Conditioning: Wilkinson's polynomial

How much does a root $r$ move if a coefficient $a_i$ changes by $\delta a_i$? To first order,

$$
\delta r \approx -\frac{r^{\,n-i}}{p'(r)}\,\delta a_i .
$$

For $W(x) = \prod_{k=1}^{20}(x - k)$ and the coefficient of $x^{19}$ (which is $-210$), this factor is about
$2 \times 10^9$ for the roots 15 and 16. Wilkinson changed that coefficient by $2^{-23} \approx 10^{-7}$, a
relative change of $6 \times 10^{-10}$, and ten of the roots became five complex pairs:

![Wilkinson's polynomial](../generated/root_finding/wilkinson.svg)

No algorithm can do better than the conditioning allows: merely storing the coefficients of $W$ in
double precision perturbs them by more than enough to move the upper roots by $10^{-3}$. A good
algorithm is **backward stable**: it computes the exact roots of a polynomial whose coefficients are
close to the given ones. The companion matrix method with balancing achieves this in a norm-wise
sense.

{{#include ../generated/root_finding/polynomial_accuracy.md}}

The lesson for practice: a polynomial given by its roots or by its values should not be converted to
coefficients if it can be avoided. Polynomials of high degree are better represented in another
basis, such as the [Chebyshev basis](../interpolation/polynomial.md#chebyshev-series).

### Multiple roots

A root of multiplicity $m$ is perturbed by about $\varepsilon^{1/m}$ times its scale: a triple root, $10^{-5}$.
The computed eigenvalues form a small regular cluster around it, often a real value and a nearly
real complex pair. The *mean* of the cluster is far more accurate than its members, because the
perturbations average out: for $(x - 1)^3$ the members are off by $10^{-5}$ and the mean by $10^{-16}$.

### Real roots

[`polynomial_roots`](../../api/lib_jsl_root_finding/polynomial_roots/fn.polynomial_roots.html) returns
the distinct real roots. It takes the eigenvalues that are real, or nearly real ($|\operatorname{Im} z| \le 10^{-3}|z|$)
with $p$ vanishing at their real part to within rounding error; groups eigenvalues between which $p$
is indistinguishable from zero into one root at their mean; and polishes isolated roots with a few
steps of Newton's method.

## Algorithm

1. Strip leading zero coefficients and factor out roots at zero (trailing zeros).
2. Form the companion matrix of the monic polynomial.
3. Balance it, reduce it to Hessenberg form, and compute its eigenvalues by the shifted QR algorithm.
4. For real roots: filter, cluster and polish as above.

## Implementation

The companion matrix and its eigenvalues:

```rust,ignore
{{#include ../../../crates/root_finding/src/polynomial_roots.rs:companion}}
```

The real roots:

```rust,ignore
{{#include ../../../crates/root_finding/src/polynomial_roots.rs:real_roots}}
```

- Coefficients run from the highest power down, as in MATLAB and NumPy: `[1, 0, -2]` is $x^2 - 2$.
- The "rounding error" in evaluating $p(x)$ by Horner's rule is bounded by a small multiple of
  $n\varepsilon\sum_i |a_i|\,|x|^{n-i}$; values below it carry no information about the sign of $p$.
- An earlier version isolated the real roots between the roots of $p'$, found recursively. It reported
  critical points as roots (25 "roots" for degree 10) and overflowed at degree 20; it was replaced by
  the eigenvalue method.

## In practice

```rust,ignore
{{#include ../../../crates/root_finding/examples/root_finding_polynomial.rs}}
```

**Strengths**

- All roots at once, complex ones included, with no starting guesses.
- Backward stable, using a mature eigenvalue algorithm.

**Weaknesses**

- $O(n^3)$ cost and $O(n^2)$ memory: for degrees in the thousands, use structured or iterative methods
  (Aberth-Ehrlich, or eigenvalues of a colleague matrix in the Chebyshev basis).
- Roots of ill-conditioned polynomials are inaccurate, as with any method, and multiple roots are
  found only to about $\varepsilon^{1/m}$ in the complex output.
- Distinct real roots too close together for $p$ to separate in floating point are merged.

## Exercises

1. Show by expanding the determinant that the characteristic polynomial of the companion matrix is
   $z^n + c_1 z^{n-1} + \cdots + c_n$.
2. Compute the condition factor $r^{19}/W'(r)$ for each root of Wilkinson's polynomial and plot it.
   Which roots are most sensitive?
3. Compute the roots of $(x - 1)^m$ for $m = 2, \ldots, 8$ with `polynomial_roots_complex` and verify that
   they lie close to a circle of radius proportional to $\varepsilon^{1/m}$.
4. The Chebyshev polynomial $T_{20}$ has well-separated roots $\cos((2k - 1)\pi/40)$. Find them from its
   monomial coefficients and measure the errors. Why are they so much smaller than for $W$?
5. Find the poles of a Butterworth filter of order 8 as the roots of $1 + (-s^2)^8$ and check that they
   lie on a circle.

## Further reading

- J. H. Wilkinson, "The evaluation of the zeros of ill-conditioned polynomials", *Numerische
  Mathematik* 1 (1959) 150–180, and "The perfidious polynomial", in *Studies in Numerical Analysis*,
  MAA, 1984.
- A. Edelman and H. Murakami, "Polynomial roots from companion matrix eigenvalues", *Mathematics of
  Computation* 64 (1995) 763–776.
- B. N. Parlett and C. Reinsch, "Balancing a matrix for calculation of eigenvalues and eigenvectors",
  *Numerische Mathematik* 13 (1969) 293–304.
- M. A. Jenkins and J. F. Traub, "A three-stage variable-shift iteration for polynomial zeros", *SIAM
  Journal on Numerical Analysis* 7 (1970) 545–566.
- V. Y. Pan, "Solving a polynomial equation: some history and recent progress", *SIAM Review* 39 (1997)
  187–220.
