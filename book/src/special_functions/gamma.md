# The gamma function and related functions

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Evaluate the gamma function $\Gamma(x)$, its logarithm and logarithmic derivative $\psi(x)$, the beta function $B(a, b)$, factorials and binomial coefficients |
| **Devised by** | Leonhard Euler (1729), the gamma and beta functions; Adrien-Marie Legendre (1811), the notation $\Gamma$; James Stirling (1730), the asymptotic series; Cornelius Lanczos (1964), the approximation used here |
| **Accuracy** | About $10^{-15}$ relative for $\Gamma$ and $\ln\Gamma$, $10^{-14}$ for $\psi$; factorials and binomial coefficients exact in integer arithmetic while they fit |
| **Cost** | A dozen arithmetic operations and one `exp` and `powf` |
| **API** | [`gamma`](../../api/lib_jsl_special_functions/gamma/fn.gamma.html), [`ln_gamma`](../../api/lib_jsl_special_functions/gamma/fn.ln_gamma.html), [`digamma`](../../api/lib_jsl_special_functions/gamma/fn.digamma.html), [`beta`](../../api/lib_jsl_special_functions/beta/fn.beta.html), [`ln_beta`](../../api/lib_jsl_special_functions/beta/fn.ln_beta.html), [`factorial`](../../api/lib_jsl_special_functions/factorial/fn.factorial.html), [`factorial_f64`](../../api/lib_jsl_special_functions/factorial/fn.factorial_f64.html), [`ln_factorial`](../../api/lib_jsl_special_functions/factorial/fn.ln_factorial.html), [`binomial_coefficient`](../../api/lib_jsl_special_functions/binomial_coefficients/fn.binomial_coefficient.html), [`ln_binomial_coefficient`](../../api/lib_jsl_special_functions/binomial_coefficients/fn.ln_binomial_coefficient.html) |

</div>

## Overview

The **gamma function** extends the factorial from the integers to real (and complex) numbers,
with $\Gamma(n + 1) = n!$. It is the most important special function: it normalises probability
distributions (gamma, beta, chi-square, Student's $t$), appears in the volume of a sphere in $n$
dimensions, in fractional derivatives and integrals, in the reflection formulas of almost every
other special function, and in the functional equation of the zeta function. Its logarithm is
needed whenever ratios of large factorials appear, and its logarithmic derivative, the **digamma
function**, appears in statistics (maximum-likelihood estimates for gamma and beta distributions)
and in sums of rational functions.

## History

In 1729 Daniel Bernoulli and Christian Goldbach asked how to interpolate the factorial between the
integers. Leonhard Euler answered in letters to Goldbach with an infinite product and, a few weeks
later, the integral $\int_0^1(-\ln t)^x\,dt$, equivalent to the modern definition below; he also
introduced the beta integral. Adrien-Marie Legendre gave the function the letter $\Gamma$ and the
shift by one that makes $\Gamma(n) = (n - 1)!$. James Stirling's asymptotic formula (1730) for $\ln n!$
became the standard method of computing it. Harald Bohr and Johannes Mollerup (1922) proved that
$\Gamma$ is the only log-convex function interpolating the factorial, which settles why it is the
"right" extension.

For computers, Cornelius Lanczos (1964) found an approximation, a few terms of a rational series
times an elementary factor, that is accurate to near machine precision for all positive arguments;
with coefficients by Paul Godfrey and others it is the method used in most libraries, including
this one.

## The idea

### The gamma function

For $x > 0$,

$$
\Gamma(x) = \int_0^\infty t^{x-1}e^{-t}\,dt, \qquad \Gamma(x + 1) = x\,\Gamma(x), \qquad \Gamma(1) = 1,
$$

so $\Gamma(n + 1) = n!$. The recurrence continues $\Gamma$ to negative arguments, where it has simple poles
at $0, -1, -2, \ldots$. Two further identities are indispensable: the **reflection formula**
$\Gamma(x)\Gamma(1 - x) = \pi/\sin\pi x$, which gives $\Gamma(\tfrac12) = \sqrt\pi$ and maps negative arguments to
positive ones, and the duplication formula $\Gamma(x)\Gamma(x + \tfrac12) = 2^{1-2x}\sqrt\pi\,\Gamma(2x)$.

![The gamma function](../generated/special_functions/gamma.svg)

$\Gamma$ grows faster than any exponential and overflows double precision beyond $x \approx 171.6$; $1/\Gamma$ is
an entire function, with zeros where $\Gamma$ has poles. For large $x$, **Stirling's series**

$$
\ln\Gamma(x) = (x - \tfrac12)\ln x - x + \tfrac12\ln 2\pi + \frac1{12x} - \frac1{360x^3} + \cdots
$$

is asymptotic: very accurate for large $x$, divergent if summed to infinity.

The **Lanczos approximation** writes, for $x > 0$ and a chosen constant $g$,

$$
\Gamma(x + 1) = \sqrt{2\pi}\,(x + g + \tfrac12)^{x+1/2}e^{-(x+g+1/2)}\Big[c_0 + \sum_{k=1}^{N}\frac{c_k}{x + k}\Big],
$$

with coefficients $c_k$ that depend on $g$. With $g = 7$ and nine coefficients the relative error is
about $10^{-15}$ for every $x > 0$, with no case analysis. The library uses it for $x \ge \tfrac12$, the
reflection formula below, and exact products for integers up to 171.

### The digamma function

The **digamma function** $\psi(x) = \Gamma'(x)/\Gamma(x)$ satisfies $\psi(x + 1) = \psi(x) + 1/x$ and $\psi(1) = -\gamma$
(the Euler-Mascheroni constant). It is computed by recurring upward to a large argument and summing
its asymptotic series $\psi(x) \sim \ln x - \frac1{2x} - \sum_k\frac{B_{2k}}{2k\,x^{2k}}$ with Bernoulli numbers $B_{2k}$, and by
reflection for negative $x$.

### The beta function

The **beta function** is

$$
B(a, b) = \int_0^1 t^{a-1}(1 - t)^{b-1}\,dt = \frac{\Gamma(a)\,\Gamma(b)}{\Gamma(a + b)},
$$

the normalising constant of the beta distribution. It is computed through logarithms of gamma
functions, to avoid overflow in the numerator and denominator.

### Factorials and binomial coefficients

$n!$ fits in a `u64` up to $n = 20$ and in an `f64` up to $n = 170$; the binomial coefficient
$\binom nk = \frac{n!}{k!\,(n-k)!}$ is computed exactly in integers by the multiplicative formula
$\prod_{i=1}^{k}\frac{n - k + i}{i}$, dividing at each step so that intermediate values stay small, with
`checked` versions returning `None` on overflow. For large arguments, `ln_factorial` and
`ln_binomial_coefficient` use $\ln\Gamma$.

### Accuracy

{{#include ../generated/special_functions/gamma_accuracy.md}}

The largest error of `gamma`, $10^{-13}$ at $x = 170.5$, comes from the exponential in the Lanczos formula:
a relative error $\delta$ in the argument of $e^{x}$ becomes an error $x\delta$ in the result.

## Algorithm

`gamma(x)`: `NaN` at the poles; the exact product for integers up to 171; reflection for $x < \tfrac12$;
`+∞` beyond 171.7; otherwise the Lanczos formula, with the power split in two to avoid premature
overflow. `ln_gamma` uses the logarithm of the Lanczos formula, with reflection for $x < \tfrac12$.

## Implementation

```rust,ignore
{{#include ../../../crates/special_functions/src/gamma.rs:gamma}}
```

- `factorial` panics on overflow (use `factorial_checked` for `Option`); `binomial_coefficient`
  likewise, with `binomial_coefficient_checked`.
- `ln_gamma` returns $\ln|\Gamma(x)|$ for negative non-integer $x$.

## In practice

```rust,ignore
{{#include ../../../crates/special_functions/examples/special_gamma.rs}}
```

**Strengths**

- Near machine precision for all arguments, with simple code.
- Logarithmic versions avoid overflow in ratios of huge factorials.

**Weaknesses**

- Real arguments only (no complex gamma).
- Relative accuracy degrades slowly near the overflow limit and near the poles.

## Exercises

1. Prove $\Gamma(x + 1) = x\Gamma(x)$ by integration by parts, and compute $\Gamma(\tfrac12)$ from a Gaussian
   integral.
2. Use Stirling's series to estimate $\ln 100!$ with two and three terms, and compare with
   `ln_factorial(100)`.
3. Show that the volume of the unit ball in $\mathbb R^n$ is $\pi^{n/2}/\Gamma(n/2 + 1)$, and find the dimension in
   which it is largest.
4. Verify the duplication formula numerically for several $x$ and estimate the rounding error.
5. Compute $\binom{1000}{500}$ as $\exp(\ln\binom{1000}{500})$. How many correct digits does it have, and why?

## Further reading

- P. J. Davis, "Leonhard Euler's integral: a historical profile of the gamma function", *American
  Mathematical Monthly* 66 (1959) 849–869.
- C. Lanczos, "A precision approximation of the gamma function", *SIAM Journal on Numerical
  Analysis* B1 (1964) 86–96.
- NIST Digital Library of Mathematical Functions, chapter 5, <https://dlmf.nist.gov/5>.
- E. Artin, *The Gamma Function*, Holt, Rinehart and Winston, 1964 (Dover reprint, 2015).
