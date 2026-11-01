# Rational interpolation and approximation

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Approximate a function by a ratio of polynomials $p(x)/q(x)$, from samples or from a Taylor series, so as to capture poles, steep gradients and behaviour beyond the reach of polynomials |
| **Devised by** | Padé approximants: Henri Padé (1892), after Jacobi and Frobenius. Rational extrapolation: Roland Bulirsch and Josef Stoer (1964). Barycentric rational interpolation: Jean-Paul Berrut (1988), Michael Floater and Kai Hormann (2007). AAA: Yuji Nakatsukasa, Olivier Sète and Lloyd N. Trefethen (2018) |
| **Accuracy** | Floater-Hormann: $O(h^{d+1})$ on any points. Bulirsch-Stoer and AAA: often geometric, exact for rational functions of sufficient degree. Padé: matches the Taylor series to order $L + M$ |
| **Cost** | Floater-Hormann: $O(nd)$ weights, $O(n)$ per point. Bulirsch-Stoer: $O(n^2)$ per point. AAA: an SVD per added term. Padé: one $M \times M$ linear solve |
| **API** | [`FloaterHormann`](../../api/lib_jsl_interpolation/one_d/floater_hormann/struct.FloaterHormann.html), [`BulirschStoer`](../../api/lib_jsl_interpolation/one_d/bulirsch_stoer/struct.BulirschStoer.html), [`AaaApproximation`](../../api/lib_jsl_interpolation/one_d/aaa/struct.AaaApproximation.html), [`PadeApproximant`](../../api/lib_jsl_interpolation/one_d/pade_approximants/struct.PadeApproximant.html) |

</div>

## Overview

A polynomial can never become infinite, and it grows without bound as $|x| \to \infty$. Many
functions do not behave like that: they have poles, approach constants, or change steeply near
singularities just off the interval of interest. A **rational function**, a ratio of two
polynomials, can do all of these things, and so rational approximations are often dramatically
better than polynomial ones of the same number of parameters.

The library offers four rational methods for different situations. **Floater-Hormann** interpolation
is a robust, pole-free alternative to polynomial interpolation at equally spaced or irregular points.
**Bulirsch-Stoer** interpolation builds the diagonal rational interpolant through all the data and
estimates its own error. The **AAA algorithm** chooses a small set of support points automatically
and finds a near-best rational approximation to sampled data. **Padé approximants** turn a Taylor
series into a rational function that is often accurate far beyond the series' radius of convergence.

## History

Rational approximations of power series were studied by Jacobi and Frobenius, but it was Henri Padé's
1892 thesis that organised them into the **Padé table** of all $[L/M]$ approximants and studied its
structure. Padé approximants became a standard tool of theoretical physics in the twentieth century,
for summing divergent perturbation series.

Roland Bulirsch and Josef Stoer (1964) devised a recursive tableau, like Neville's algorithm for
polynomials, for evaluating the diagonal rational interpolant, and used it in their extrapolation
method for differential equations. Barycentric rational interpolation grew from Jean-Paul Berrut's
observation (1988) that changing the weights of the barycentric formula gives rational interpolants
guaranteed to have no poles; Michael Floater and Kai Hormann (2007) found the family of such
interpolants with arbitrarily high order. Most recently, Yuji Nakatsukasa, Olivier Sète and Nick
Trefethen's **AAA algorithm** (2018, "adaptive Antoulas-Anderson") combined the barycentric form with
greedy point selection and least squares, and made rational approximation practical and automatic.

## The idea

### Barycentric rational functions

The [barycentric formula](polynomial.md#the-barycentric-formula)

$$
r(x) = \frac{\sum_j \dfrac{w_j}{x - x_j}\, y_j}{\sum_j \dfrac{w_j}{x - x_j}}
$$

interpolates the data for *any* nonzero weights $w_j$: as $x \to x_k$ the $k$-th terms dominate both sums
and $r(x) \to y_k$. Only for the special weights $w_j = 1/\prod_{k\ne j}(x_j - x_k)$ is $r$ a polynomial;
other weights give rational functions of degree at most $n - 1$. Choosing the weights well is the key
to both Floater-Hormann interpolation and AAA.

### Floater-Hormann interpolation

Floater and Hormann blend the polynomial interpolants of degree $d$ through each run of $d + 1$
consecutive points. In barycentric form their weights are

$$
w_k = (-1)^{k - d} \sum_{i \in J_k} \prod_{\substack{j = i \\ j \ne k}}^{i + d} \frac{1}{|x_k - x_j|},
$$

with $J_k$ the runs that contain $x_k$. The interpolant has **no real poles** for any $d$, converges like
$O(h^{d+1})$ for smooth functions on any distribution of points, and with $d = n - 1$ reduces to the
polynomial interpolant. A moderate $d$ (3 to 8) combines high order with stability at equally spaced
points, exactly where polynomials fail.

### Bulirsch-Stoer rational interpolation

Bulirsch and Stoer evaluate the diagonal rational interpolant through all $n$ points with a tableau
of corrections, analogous to Neville's algorithm for polynomials: each column raises the degree of
numerator or denominator by one, and the last correction applied estimates the error. The
evaluation costs $O(n^2)$ per point, and a pole of the interpolant at the evaluation point is reported
as an error. When the data come from a rational function of lower degree, later corrections vanish
and the library stops early; the error estimate is then at rounding level.

### The AAA algorithm

AAA approximates sampled data $f(z_i)$, $i = 1, \ldots, m$, by a barycentric rational function whose
**support points** are a subset of the samples:

1. Start with no support points and the approximation equal to the mean of $f$.
2. Add the sample where the current approximation is worst as a support point (so the approximant
   interpolates it).
3. Choose the weights to minimise $\big\|f(z_i) \sum_j \frac{w_j}{z_i - z_j} - \sum_j \frac{w_j f_j}{z_i - z_j}\big\|$
   over the remaining samples, with $\|\mathbf w\| = 1$. This **linearised** least-squares problem is solved
   by the smallest singular vector of the **Loewner matrix** $A_{ij} = (f(z_i) - f_j)/(z_i - z_j)$.
4. Repeat until the error at all samples is below the tolerance.

AAA typically finds an approximation within a few digits of the best rational approximation, with
remarkably few terms, and works on any set of sample points. Its main pitfall is **spurious poles**:
occasionally a pole and a nearby zero almost cancel (a *Froissart doublet*), producing a narrow spike
between samples. Production codes detect such doublets and remove them; the library leaves this to
the user.

### Rational methods on equally spaced data

The Runge function $1/(1 + 25x^2)$ sampled at equally spaced points defeats polynomial interpolation
(see [polynomial interpolation](polynomial.md#the-interpolation-error-and-the-runge-phenomenon)). The
rational methods succeed:

![Rational interpolation of the Runge function](../generated/interpolation/rational_runge.svg)

Floater-Hormann with $d = 3$ converges algebraically, as its theory promises. Bulirsch-Stoer and AAA
reach machine precision with about twenty points, but this is partly a special case: the Runge
function is itself rational, of type $(0, 2)$, so any rational method of sufficient degree recovers it
exactly. For non-rational functions with nearby singularities they still converge geometrically, as
the `tan x` example below shows.

### Padé approximants

The $[L/M]$ **Padé approximant** of a function with Taylor series $\sum c_k x^k$ is the rational function
$P_L(x)/Q_M(x)$, $Q_M(0) = 1$, whose own Taylor series agrees with $f$'s through the term $x^{L+M}$. The
condition $f Q_M - P_L = O(x^{L+M+1})$ is linear in the coefficients: the denominator comes from an
$M \times M$ linear system,

$$
\sum_{j=1}^M q_j\, c_{L+i-j} = -c_{L+i}, \qquad i = 1, \ldots, M,
$$

and then the numerator from $p_i = \sum_{j=0}^{\min(i, M)} q_j c_{i-j}$. Because a rational function can mimic
the singularities that limit a power series, Padé approximants often converge far outside the
series' disc of convergence. For $\ln(1 + x)$, whose Taylor series converges only for $|x| < 1$, the
$[10/10]$ approximant built from the same 21 coefficients remains accurate far beyond:

{{#include ../generated/interpolation/pade_log.md}}

## Algorithm

- **Floater-Hormann**: weights by the formula above; evaluation by the barycentric formula.
- **Bulirsch-Stoer**: for each evaluation point, build the rational tableau from the nearest data point
  outwards, adding the corrections until they reach rounding level.
- **AAA**: greedy support selection, a Loewner matrix and its SVD per step, until the sample error is
  below the tolerance or the term limit is reached.
- **Padé**: solve the $M \times M$ system for $Q$, form $P$ by convolution, evaluate both by Horner's rule.

## Implementation

The core of the AAA iteration:

```rust,ignore
{{#include ../../../crates/interpolation/src/one_d/aaa.rs:aaa_step}}
```

- All four methods require distinct abscissas; Floater-Hormann also requires them in increasing order.
- The AAA weights come from the [SVD in `lib_jsl_linalg`](../../api/lib_jsl_linalg/trait.SVD.html).
- The Padé system can be singular, when the Taylor coefficients make the requested $[L/M]$ degenerate;
  the constructor then reports an error, and a neighbouring entry of the Padé table should be used.

## In practice

```rust,ignore
{{#include ../../../crates/interpolation/examples/interpolation_rational.rs}}
```

**Strengths**

- Poles, steep gradients and asymptotic behaviour are captured with few parameters.
- Floater-Hormann gives stable, high-order interpolation on equally spaced data.
- AAA is automatic and often near-optimal; Padé extends power series beyond their radius of
  convergence.

**Weaknesses**

- Rational interpolants other than Floater-Hormann may have poles in the interval (Bulirsch-Stoer
  reports them; AAA may produce spurious ones).
- Bulirsch-Stoer costs $O(n^2)$ per evaluation.
- Padé systems can be ill-conditioned or singular, and the approximants may have spurious poles too.

## Exercises

1. Show that the barycentric formula interpolates $y_k$ at $x_k$ for any nonzero weights.
2. Compute the Floater-Hormann weights for $d = 0$ (Berrut's interpolant) on equally spaced points and
   show they are $\pm1$ up to a constant.
3. Compute the $[1/1]$ Padé approximant of $e^x$ and compare it with the degree-2 Taylor polynomial
   at $x = 1$.
4. Use AAA on samples of $|x|$ at 1000 points in $[-1, 1]$. How many terms are needed for $10^{-6}$?
   Where do the poles of the approximant lie? (Newman showed in 1964 that rational approximation of
   $|x|$ converges like $e^{-\pi\sqrt n}$, far faster than polynomials.)
5. Find data for which AAA produces a Froissart doublet, and remove it by deleting the corresponding
   support point and refitting.

## Further reading

- H. Padé, *Sur la représentation approchée d'une fonction par des fractions rationnelles*, thesis,
  Paris, 1892.
- R. Bulirsch and J. Stoer, "Fehlerabschätzungen und Extrapolation mit rationalen Funktionen bei
  Verfahren vom Richardson-Typus", *Numerische Mathematik* 6 (1964) 413–427.
- M. S. Floater and K. Hormann, "Barycentric rational interpolation with no poles and high rates of
  approximation", *Numerische Mathematik* 107 (2007) 315–331.
- Y. Nakatsukasa, O. Sète and L. N. Trefethen, "The AAA algorithm for rational approximation", *SIAM
  Journal on Scientific Computing* 40 (2018) A1494–A1522.
- G. A. Baker and P. Graves-Morris, *Padé Approximants*, 2nd ed., Cambridge University Press, 1996.
