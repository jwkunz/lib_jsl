# Hypergeometric functions

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Evaluate the Gauss hypergeometric function ${}_2F_1(a, b; c; z)$, Kummer's confluent function ${}_1F_1(a; b; x)$ and the limit function ${}_0F_1(; b; x)$ for real arguments |
| **Devised by** | John Wallis (1655), the name; Leonhard Euler (1769); Carl Friedrich Gauss (1812), the theory of ${}_2F_1$; Ernst Kummer (1836), transformations and the confluent function; Bernhard Riemann (1857), the differential equation |
| **Accuracy** | About $10^{-15}$ relative for moderate parameters; $10^{-13}$ near the singular point $z = 1$ and for large $|x|$ |
| **Cost** | A power series of tens to hundreds of terms, after a transformation that makes it converge quickly |
| **API** | [`hypergeometric`](../../api/lib_jsl_special_functions/hypergeometric_functions/fn.hypergeometric.html), [`hypergeometric_1f1`](../../api/lib_jsl_special_functions/hypergeometric_functions/fn.hypergeometric_1f1.html), [`hypergeometric_0f1`](../../api/lib_jsl_special_functions/hypergeometric_functions/fn.hypergeometric_0f1.html) |

</div>

## Overview

The **hypergeometric functions** are the great unifiers of the special functions. The series

$$
{}_pF_q(a_1, \ldots, a_p; b_1, \ldots, b_q; z) = \sum_{n=0}^\infty\frac{(a_1)_n\cdots(a_p)_n}{(b_1)_n\cdots(b_q)_n}\,\frac{z^n}{n!}, \qquad (a)_n = a(a + 1)\cdots(a + n - 1),
$$

is characterised by the ratio of successive terms being a rational function of $n$. With suitable
parameters it reproduces almost every function in this chapter: logarithms and inverse
trigonometric functions, the error function and incomplete gamma functions, Bessel functions,
elliptic integrals, and the classical orthogonal polynomials. A routine for ${}_2F_1$ is therefore a
routine for all of them, and identities between hypergeometric functions become identities between
the special functions.

## History

John Wallis used the word "hypergeometric" in 1655 for series generalising the geometric series.
Euler studied ${}_2F_1$ and found its integral representation and its transformation $z \to z/(z - 1)$. Gauss's
1812 memoir established the series' convergence (for $|z| < 1$), its value at $z = 1$, and the contiguous
relations between functions whose parameters differ by integers. Ernst Kummer (1836) found the 24
solutions of the hypergeometric differential equation and introduced the confluent function ${}_1F_1$,
obtained by letting the singular point at $z = 1$ merge with the one at infinity. Bernhard Riemann (1857)
characterised ${}_2F_1$ by its three regular singular points $0$, $1$ and $\infty$, a viewpoint that shaped complex
analysis and differential equations.

## The idea

### The Gauss function ₂F₁

${}_2F_1(a, b; c; z)$ solves the hypergeometric equation $z(1 - z)w'' + [c - (a + b + 1)z]w' - abw = 0$. Its series
converges for $|z| < 1$, slowly near $z = 1$, where the function may be singular. Linear
**transformations** map any real $z < 1$ into the fast region:

- the **Pfaff** transformation ${}_2F_1(a, b; c; z) = (1 - z)^{-a}{}_2F_1\big(a, c - b; c; \frac z{z - 1}\big)$ maps $z < -\tfrac12$ into $(\tfrac13, 1)$, where the series or the connection formula applies;
- the **connection formula** expresses ${}_2F_1$ at $z$ through two functions at $1 - z$, for $z \ge \tfrac12$, with gamma function
  coefficients; when $c - a - b$ is an integer these coefficients are infinite and a limiting form is needed
  (the library then sums the original series, slowly but correctly);
- at $z = 1$, **Gauss's theorem** ${}_2F_1(a, b; c; 1) = \frac{\Gamma(c)\Gamma(c - a - b)}{\Gamma(c - a)\Gamma(c - b)}$ for $c - a - b > 0$.

If $a$ or $b$ is a non-positive integer the series terminates, giving a polynomial (Jacobi, Legendre and
Chebyshev polynomials are of this form). If $c$ is a non-positive integer the function is undefined.

### Kummer's function ₁F₁

The confluent function ${}_1F_1(a; b; x) = M(a, b, x)$ solves $xw'' + (b - x)w' - aw = 0$. Its series converges for all
$x$, but for large negative $x$ its terms alternate and grow before they shrink, losing digits to
cancellation. **Kummer's transformation** $M(a, b, x) = e^xM(b - a, b, -x)$ replaces it by a series with positive
terms. For negative integer $a$ it is a Laguerre polynomial.

![Kummer's function](../generated/special_functions/kummer.svg)

### The limit function ₀F₁

${}_0F_1(; b; x) = \sum\frac{x^n}{(b)_n\,n!}$ is a Bessel function in disguise:
${}_0F_1(; b; -z^2/4) = \Gamma(b)(z/2)^{1-b}J_{b-1}(z)$, and with $I_{b-1}$ for positive argument. For $x > 0$ the series has positive
terms. For large negative $x$ its terms alternate and grow to about $e^{2\sqrt{|x|}}$ times the result before
decreasing, so the series can lose up to $0.87\sqrt{|x|}$ digits; at $x = -100$ it lost seven. Below $x = -10$ the library therefore uses the
Bessel relation, with the accurate $J_\nu$ of the [Bessel entry](bessel.md). (An earlier version always
summed the series.)

### Special cases

Evaluating the library's hypergeometric functions against elementary and special functions:

{{#include ../generated/special_functions/hypergeometric_identities.md}}

### Accuracy

{{#include ../generated/special_functions/hypergeometric_accuracy.md}}

## Algorithm

- ${}_2F_1$: `NaN` for $z > 1$ (unless the series terminates) or $c$ a non-positive integer; Gauss's theorem at $z = 1$; the
  Pfaff transformation for $z < -\tfrac12$; the connection formula for $z \ge \tfrac12$ unless $c - a - b$ is an integer; otherwise the
  series.
- ${}_1F_1$: the series for $x \ge 0$, Kummer's transformation for $x < 0$.
- ${}_0F_1$: the series, or the Bessel relation for $x < -10$.

## Implementation

```rust,ignore
{{#include ../../../crates/special_functions/src/hypergeometric_functions.rs:kummer}}
```

- Real arguments only; ${}_2F_1$ for $z > 1$ is complex and returns `NaN`.
- Large parameters (hundreds) make the series long and can lose accuracy; specialised methods
  (recurrences in the parameters, uniform asymptotics) are needed there.

## In practice

```rust,ignore
{{#include ../../../crates/special_functions/examples/special_hypergeometric.rs}}
```

**Strengths**

- One routine for many special functions, with the transformations that make it converge.
- Exact polynomial values for terminating series.

**Weaknesses**

- Real arguments, and moderate parameters only.
- The case of integer $c - a - b$ near $z = 1$ is slow.

## Exercises

1. Show that the ratio of successive terms of ${}_2F_1$ is $\frac{(a + n)(b + n)}{(c + n)(n + 1)}z$, and use it to prove
   convergence for $|z| < 1$.
2. Verify $\ln(1 + z) = z\,{}_2F_1(1, 1; 2; -z)$ by comparing series.
3. Use the Pfaff transformation to evaluate ${}_2F_1(1, 1; 2; -10)$ and compare with $\ln(11)/10$.
4. Show that the Legendre polynomial $P_n(x) = {}_2F_1(-n, n + 1; 1; \frac{1 - x}2)$ for $n = 2$.
5. Estimate the number of digits lost summing ${}_1F_1(a; b; -x)$ directly for $x = 20$, and check with
   `hypergeometric_1f1`, which uses Kummer's transformation.

## Further reading

- G. E. Andrews, R. Askey and R. Roy, *Special Functions*, Cambridge University Press, 1999, chapters 2
  and 4.
- J. W. Pearson, S. Olver and M. A. Porter, "Numerical methods for the computation of the confluent and
  Gauss hypergeometric functions", *Numerical Algorithms* 74 (2017) 821–866.
- NIST Digital Library of Mathematical Functions, chapters 13, 15 and 16, <https://dlmf.nist.gov/15>.
