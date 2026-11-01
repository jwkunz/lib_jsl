# Polynomial interpolation

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Find the polynomial of degree $n - 1$ through $n$ points $(x_i, y_i)$, or approximate a function $f$ on an interval by a polynomial |
| **Devised by** | Newton's divided differences (1687, 1711); Lagrange's form (Waring 1779, Lagrange 1795); the barycentric form (Jacobi 1825, Taylor 1945, Salzer 1972; popularised by Berrut and Trefethen 2004); Chebyshev polynomials (Pafnuty Chebyshev, 1854); the Runge phenomenon (Carl Runge, 1901) |
| **Accuracy** | Through Chebyshev points: geometric convergence $O(\rho^{-n})$ for analytic $f$. Through equally spaced points: may diverge |
| **Cost** | Newton: $O(n^2)$ set-up, $O(n)$ per evaluation. Barycentric: $O(n^2)$ weights, $O(n)$ per evaluation. Chebyshev series: $O(n^2)$ fit, $O(n)$ Clenshaw evaluation |
| **API** | [`PolynomialInterpolator`](../../api/lib_jsl_interpolation/one_d/polynomial_interpolator/struct.PolynomialInterpolator.html), [`LagrangeInterpolant`](../../api/lib_jsl_interpolation/one_d/lagrange_interpolant/struct.LagrangeInterpolant.html), [`ChebyshevApproximation`](../../api/lib_jsl_interpolation/one_d/chebyshev_approximations/struct.ChebyshevApproximation.html) |

</div>

## Overview

Given $n$ points with distinct abscissas, there is exactly one polynomial of degree at most $n - 1$
passing through all of them. Finding it, and evaluating it between the points, is the oldest
problem in numerical analysis, and the foundation of numerical integration, differentiation and
much of approximation theory.

The library offers three forms. **Newton's divided differences** build the polynomial incrementally
and evaluate it by Horner's rule. The **barycentric Lagrange formula** is the numerically preferred
form for evaluation at many points. A **Chebyshev series** approximates a function you can evaluate
anywhere, not just at given data, to near machine precision with a few dozen terms.

The single most important fact about polynomial interpolation is that **the choice of points
matters enormously**. Equally spaced points can make the interpolant diverge wildly as $n$ grows,
the **Runge phenomenon**, while Chebyshev points give rapid convergence for every smooth function.

## History

Interpolation tables were the computers of earlier centuries: astronomical, navigational and
mathematical tables listed values at regular steps, and users interpolated between them. Isaac
Newton's divided-difference formula (in the *Principia*, 1687, Book III, Lemma V, and *Methodus
differentialis*, 1711) remains the standard way to build an interpolant incrementally. The form now named after Joseph-Louis Lagrange
(1795) had been published by Edward Waring in 1779 and was known to Euler.

The barycentric form, a rearrangement of Lagrange's formula that is both cheaper and more stable,
appears in Jacobi's 1825 thesis and was rediscovered several times during the twentieth century; Jean-
Paul Berrut and Nick Trefethen's 2004 survey "Barycentric Lagrange interpolation" made it the
modern standard.

In 1901 Carl Runge showed that interpolating the innocent function $1/(1 + 25x^2)$ at equally spaced
points on $[-1, 1]$ gives polynomials that diverge near the ends of the interval. The cure lies in
the polynomials introduced by Pafnuty Chebyshev in 1854 in his study of steam-engine linkages: they
oscillate equally between $-1$ and $1$, and their extrema cluster towards the ends of the interval
exactly as interpolation points should.

## The idea

### Newton's divided differences

Write the interpolant as

$$
p(x) = a_0 + a_1(x - x_0) + a_2(x - x_0)(x - x_1) + \cdots + a_{n-1}(x - x_0)\cdots(x - x_{n-2}).
$$

Each new term vanishes at all the earlier points, so adding a point adds a term without changing
the others. The coefficients are the **divided differences** $a_k = f[x_0, \ldots, x_k]$, defined
recursively by $f[x_i] = y_i$ and

$$
f[x_i, \ldots, x_{i+k}] = \frac{f[x_{i+1}, \ldots, x_{i+k}] - f[x_i, \ldots, x_{i+k-1}]}{x_{i+k} - x_i}.
$$

They are computed in place in $O(n^2)$ operations, and the polynomial is evaluated by nesting the
products (Horner's rule) in $O(n)$.

### The barycentric formula

Lagrange's formula $p(x) = \sum_j y_j \ell_j(x)$, with $\ell_j(x) = \prod_{k \ne j} (x - x_k)/(x_j - x_k)$,
costs $O(n^2)$ per evaluation as written. Dividing it by the interpolant of the constant 1, which is
exactly 1, gives the **barycentric formula**

$$
p(x) = \frac{\displaystyle\sum_{j} \frac{w_j}{x - x_j}\, y_j}{\displaystyle\sum_{j} \frac{w_j}{x - x_j}},
\qquad w_j = \frac{1}{\prod_{k \ne j} (x_j - x_k)}.
$$

The weights depend only on the nodes and are computed once; each evaluation is then $O(n)$. The
formula is also remarkably stable: errors in the weights cancel between numerator and denominator,
and Higham (2004) proved it is forward stable for well-chosen points.

### The interpolation error and the Runge phenomenon

> **Interpolation error.** If $f$ has $n$ continuous derivatives, then for each $x$ there is a $\xi$ in
> the interval spanned by $x$ and the nodes with
> $$f(x) - p(x) = \frac{f^{(n)}(\xi)}{n!}\, \prod_{j=0}^{n-1} (x - x_j).$$

The first factor depends on $f$; the second, the **node polynomial**, depends only on the points. For
equally spaced points the node polynomial is far larger near the ends of the interval than in the
middle, by a factor that grows exponentially with $n$. If the derivatives of $f$ also grow fast, as
they do for $1/(1 + 25x^2)$ (which has poles at $\pm i/5$, close to the interval), the interpolant
diverges near the ends:

![The Runge phenomenon](../generated/interpolation/runge.svg)

### Chebyshev points

The **Chebyshev points** $x_j = \cos(\pi j/(n-1))$ are the projections onto the real axis of equally
spaced points on a semicircle. They cluster near the ends with density proportional to
$1/\sqrt{1 - x^2}$, exactly cancelling the growth of the node polynomial, which then becomes a scaled
Chebyshev polynomial, equioscillating with the smallest possible maximum. For the barycentric formula
their weights are known in closed form: $w_j = (-1)^j$, halved at the two ends.

The result is a convergence theory of great elegance. If $f$ is analytic inside the ellipse with foci
$\pm1$ and semi-axis sum $\rho > 1$ (a **Bernstein ellipse**), interpolation in Chebyshev points
converges like $\rho^{-n}$. For the Runge function, the largest such ellipse passes through the poles
at $\pm i/5$, giving $\rho = 1/5 + \sqrt{1 + 1/25} \approx 1.22$:

![Convergence for the Runge function](../generated/interpolation/runge_convergence.svg)

Equally spaced interpolation diverges, by a factor of $10^5$ at 50 points, while both Chebyshev
methods converge at exactly the predicted rate.

### Chebyshev series

When $f$ can be evaluated anywhere, it is natural to expand it in **Chebyshev polynomials**
$T_k(\cos\theta) = \cos k\theta$:

$$
f(x) \approx \tfrac12 c_0 + \sum_{k=1}^{n-1} c_k\, T_k(t), \qquad t = \frac{2x - a - b}{b - a}.
$$

Because $T_k(\cos\theta) = \cos k\theta$, this is a Fourier cosine series in disguise, and the coefficients
are computed from samples at the Chebyshev points by a discrete cosine sum. They decay as fast as $f$
is smooth, so trailing coefficients below a tolerance can simply be dropped (`truncate`).
[`ChebyshevApproximation`](../../api/lib_jsl_interpolation/one_d/chebyshev_approximations/struct.ChebyshevApproximation.html)
evaluates the series by **Clenshaw's recurrence**, the analogue of Horner's rule for Chebyshev
polynomials, and differentiates or integrates it by simple recurrences on the coefficients.
Chebyshev series are the basis of the Chebfun system and of the
[spectral methods](../integration/spectral-methods.md) for differential equations.

## Algorithm

- **Newton**: divided differences in place, then Horner evaluation of the Newton form.
- **Barycentric**: weights $w_j = 1/\prod_{k \ne j}(x_j - x_k)$; evaluation by the two sums above,
  returning $y_j$ exactly when $x = x_j$.
- **Chebyshev series**: sample $f$ at $x_k = \cos(\pi(k + \tfrac12)/n)$ mapped to $[a, b]$; compute
  $c_j = \tfrac2n \sum_k f(x_k) \cos(\pi j (k + \tfrac12)/n)$; evaluate by Clenshaw's recurrence.

## Implementation

The barycentric evaluation, shared with the [Floater-Hormann](rational.md) rational interpolant:

```rust,ignore
{{#include ../../../crates/interpolation/src/one_d/lagrange_interpolant.rs:barycentric}}
```

- Both interpolating forms reject repeated abscissas. (The Newton form used to divide by zero and
  return infinite coefficients instead.)
- The barycentric weights are products of $n - 1$ differences and can overflow or underflow for
  hundreds of points on a wide interval; rescaling all weights by a common factor leaves the formula
  unchanged, which is the usual remedy.
- The Chebyshev series uses the points $\cos(\pi(k + \tfrac12)/n)$ (Chebyshev points of the first
  kind, which exclude the endpoints); the interpolation figure uses the second-kind points
  $\cos(\pi j/(n-1))$, which include them. Both cluster correctly.

## In practice

```rust,ignore
{{#include ../../../crates/interpolation/examples/interpolation_polynomial.rs}}
```

**Strengths**

- One global formula, smooth and easy to differentiate and integrate.
- With Chebyshev points, geometric convergence for analytic functions: often machine precision
  with a few dozen points.

**Weaknesses**

- With equally spaced points, high degrees are useless: the Runge phenomenon, and huge sensitivity to
  noise in the data. Use [piecewise interpolation](splines.md) or [rational](rational.md) methods
  for such data.
- A single global polynomial cannot follow a function with a kink or a jump; the error then falls
  only algebraically, with oscillations near the discontinuity.

## Exercises

1. Show that the Newton coefficients of $p$ through $(0, 1), (1, 2), (2, 0), (3, -1)$ are
   $1, 1, -\tfrac32, \tfrac23$, and use them to check $p(1.5) = 1.125$.
2. Show that the barycentric weights for the Chebyshev points $\cos(\pi j/n)$ are proportional to
   $(-1)^j$, halved at $j = 0$ and $j = n$.
3. Compute the maximum of $|\prod_j (x - x_j)|$ on $[-1, 1]$ for 11 equally spaced and 11 Chebyshev
   points. How different are they?
4. Interpolate $|x|$ in Chebyshev points. How does the error decrease with $n$? Compare with $e^x$.
5. Add noise of size $10^{-6}$ to samples of $e^x$ at 30 equally spaced points and interpolate. How
   large is the error? Repeat with Chebyshev points.

## Further reading

- C. Runge, "Über empirische Funktionen und die Interpolation zwischen äquidistanten Ordinaten",
  *Zeitschrift für Mathematik und Physik* 46 (1901) 224–243.
- J.-P. Berrut and L. N. Trefethen, "Barycentric Lagrange interpolation", *SIAM Review* 46 (2004)
  501–517.
- N. J. Higham, "The numerical stability of barycentric Lagrange interpolation", *IMA Journal of
  Numerical Analysis* 24 (2004) 547–556.
- L. N. Trefethen, *Approximation Theory and Approximation Practice*, extended ed., SIAM, 2019. The
  modern account of Chebyshev interpolation and its convergence.
- W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery, *Numerical Recipes*, 3rd ed.,
  Cambridge University Press, 2007, sections 3.2 and 5.8.
