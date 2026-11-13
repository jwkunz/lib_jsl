# Incomplete gamma and beta functions and the error function

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Evaluate the regularised incomplete gamma functions $P(a, x)$ and $Q(a, x)$, the regularised incomplete beta function $I_x(a, b)$, the error function and its complement and inverse, and Dawson's integral |
| **Devised by** | Incomplete gamma: Adrien-Marie Legendre and Friedrich Prym (19th century); incomplete beta: Karl Pearson's tables (1934); the error function: named by James Glaisher (1871); Dawson's integral: H. G. Dawson (1897); continued-fraction evaluation: William Lentz (1976), modified by Ian Thompson and Alan Barnett (1986); Dawson by George Rybicki (1989) |
| **Accuracy** | About $10^{-15}$ relative for moderate parameters, $10^{-13}$ for parameters in the hundreds |
| **Cost** | A series or continued fraction of $O(\sqrt{a})$ to $O(a)$ terms; Dawson's integral, a sum of a few dozen exponentials |
| **API** | [`incomplete_gamma`](../../api/lib_jsl_special_functions/incomplete_gamma/fn.incomplete_gamma.html), [`incomplete_gamma_upper`](../../api/lib_jsl_special_functions/incomplete_gamma/fn.incomplete_gamma_upper.html), [`incomplete_beta`](../../api/lib_jsl_special_functions/incomplete_beta/fn.incomplete_beta.html), [`error_function`](../../api/lib_jsl_special_functions/error_function/fn.error_function.html), [`erfc`](../../api/lib_jsl_special_functions/error_function/fn.erfc.html), [`erf_inv`](../../api/lib_jsl_special_functions/error_function/fn.erf_inv.html), [`dawson_integral`](../../api/lib_jsl_special_functions/dawsons_integral/fn.dawson_integral.html) |

</div>

## Overview

Cutting the integrals that define the gamma and beta functions short gives the **incomplete**
gamma and beta functions. Divided by the complete values, they are the cumulative distribution
functions of the gamma and beta distributions, and through them of the Poisson, chi-square,
binomial, Student's $t$ and $F$ distributions (see [statistical functions](statistical.md)). The
**error function**, the integral of a Gaussian, is a special case of the incomplete gamma function,
and gives the probabilities of the normal distribution, the solution of the heat equation from a
step, and the shape of a diffused interface.

The difficulty is that these functions are probabilities: values near 0 and near 1 both matter,
and the complement $1 - P$ must be computed directly, not by subtraction, when $P$ is close to 1.

## History

The incomplete gamma function was studied by Legendre and named and analysed by Friedrich Prym
(1877) and others in the nineteenth century; Karl Pearson published extensive tables of both
incomplete functions for statistics (1922, 1934). The error function arose with the normal
distribution in the work of de Moivre, Laplace and Gauss; the name and the notation $\operatorname{erf}$ are due to
James Glaisher (1871). H. G. Dawson tabulated the integral now named after him in 1897.

On computers, the standard methods became the power series for small arguments and continued
fractions for large ones, evaluated by William Lentz's algorithm (1976), made robust by Ian
Thompson and Alan Barnett (1986). George Rybicki (1989) gave an elegant method for Dawson's
integral based on a sampling theorem.

## The idea

### The incomplete gamma function

The gamma function is an integral from 0 to $\infty$. Stopping the integral at a finite point $x$
gives the **incomplete** gamma function, and dividing by $\Gamma(a)$ so that the result runs from 0
to 1 gives the **regularised** forms

$$
P(a, x) = \frac1{\Gamma(a)}\int_0^x t^{a-1}e^{-t}\,dt, \qquad Q(a, x) = 1 - P(a, x) = \frac1{\Gamma(a)}\int_x^\infty t^{a-1}e^{-t}\,dt .
$$

$P$ is the fraction of the integral accumulated up to $x$, and $Q$ is the remaining tail. These
are the cumulative distribution functions of the gamma, chi-squared and Poisson distributions,
which is why they matter. As a function of $x$, $P(a, x)$ rises from 0 to 1, most steeply near
$x = a$ (where the integrand peaks), over a width of about $\sqrt a$:

![The incomplete gamma function](../generated/special_functions/incomplete_gamma.svg)

<p class="caption" id="fig-6-2"><strong>Figure 6.2.</strong> The incomplete gamma function.</p>

No single formula computes $P$ efficiently everywhere. Two expansions, each good on one side of
the rise, cover the whole range.

**Left of the rise: a series.** For $x < a + 1$,

$$
P(a, x) = \frac{x^a e^{-x}}{\Gamma(a + 1)}\sum_{n=0}^\infty\frac{x^n}{(a + 1)(a + 2)\cdots(a + n)} .
$$

Each term is the previous one multiplied by $x/(a + n)$. When $x < a + 1$ this ratio is below 1
from the start, so the terms decrease and the series converges quickly.

**Right of the rise: a continued fraction.** For $x \ge a + 1$ the series would need many terms.
There, Legendre's **continued fraction** for the tail converges quickly instead:

$$
Q(a, x) = \frac{x^a e^{-x}}{\Gamma(a)}\cdot\cfrac{1}{x + 1 - a - \cfrac{1\cdot(1 - a)}{x + 3 - a - \cfrac{2\,(2 - a)}{x + 5 - a - \cdots}}} .
$$

A continued fraction is an expression of the form $b_0 + a_1/(b_1 + a_2/(b_2 + \cdots))$. Cutting
it off after $n$ levels gives an ordinary fraction, its $n$-th **convergent**, and the value of
the continued fraction is the limit of the convergents. Continued fractions often converge in
regions where power series converge slowly or not at all, which makes them the natural partner
of a series.

**Why each expansion is used where it is.** There is a second reason for the split, besides
speed. Left of the rise $P$ is the smaller of the pair, and right of it $Q$ is. Each expansion
computes the *smaller* quantity directly, to full relative accuracy, and the other follows by
subtracting from 1, which loses nothing when the subtracted value is small. Doing it the other
way round, computing a tiny tail as 1 minus a number very close to 1, would cancel most of its
digits.

**Evaluating a continued fraction.** Written as above, a continued fraction seems to need
evaluating from the bottom up, which requires deciding in advance how many levels to keep. The
**modified Lentz algorithm** evaluates it from the top down instead. It keeps track of the ratios
of successive convergents and multiplies the running value by one such ratio per level, stopping
when the ratio is 1 to within rounding error.

### The incomplete beta function

The same construction applied to the [beta function](gamma.md#the-beta-function) gives the
regularised incomplete beta function

$$
I_x(a, b) = \frac1{B(a, b)}\int_0^x t^{a-1}(1 - t)^{b-1}\,dt ,
$$

which rises from 0 at $x = 0$ to 1 at $x = 1$. It is the cumulative distribution function of
the beta distribution, and through it of Student's $t$, the $F$ and the binomial distributions.

It has a continued fraction (derived from Gauss's continued fraction for the hypergeometric
function ${}_2F_1$) that converges rapidly for $x < (a + 1)/(a + b + 2)$, the region below the mean
of the beta distribution. Above that point the problem is reflected: swapping the roles of $t$
and $1 - t$ in the integral gives the symmetry $I_x(a, b) = 1 - I_{1-x}(b, a)$, and the right side
is in the region of rapid convergence. The prefactor $x^a(1 - x)^b/(aB(a, b))$ is computed from
logarithms to avoid overflow for large parameters.

### The error function

The **error function** and its complement are

$$
\operatorname{erf} x = \frac2{\sqrt\pi}\int_0^x e^{-t^2}\,dt, \qquad \operatorname{erfc} x = 1 - \operatorname{erf} x .
$$

$\operatorname{erf}$ is the area under the bell curve $e^{-t^2}$ from 0 to $x$, scaled to tend to 1; it gives
the probability that a normally distributed measurement falls within a given distance of its
mean. It is a special case of the incomplete gamma function. Substituting $t = \sqrt u$ in the
integral, so that $dt = du/(2\sqrt u)$, turns it into $\frac1{\sqrt\pi}\int_0^{x^2}u^{-1/2}e^{-u}\,du$,
and since $\Gamma(\tfrac12) = \sqrt\pi$,

$$
\operatorname{erf} x = P(\tfrac12, x^2), \qquad \operatorname{erfc} x = Q(\tfrac12, x^2) \quad (x \ge 0).
$$

The library computes both through the incomplete gamma function.

Why provide $\operatorname{erfc}$ as a separate function, when it is just $1 - \operatorname{erf}$? Because of the
upper tail. For $x$ beyond about 5.9, $\operatorname{erf} x$ differs from 1 by less than the spacing of
double-precision numbers, so it rounds to exactly 1 and $1 - \operatorname{erf} x$ evaluates to zero. The
true value is small but not zero, and tail probabilities of that size matter in practice (error
rates of communication links, for example). Computed directly, $\operatorname{erfc} x$ keeps full relative
accuracy down to the underflow threshold near $x = 26.5$:

![erfc and 1 − erf](../generated/special_functions/erfc.svg)

<p class="caption" id="fig-6-3"><strong>Figure 6.3.</strong> erfc and 1 − erf.</p>

The **inverse error function** has no convenient series. It is found by solving
$\operatorname{erf} y = x$ for $y$ as a root-finding problem: start from Winitzki's closed-form approximation
and refine it by Halley's method, a relative of [Newton's method](../root_finding/newton.md)
that also uses the second derivative and converges cubically. Both $\operatorname{erf}'$ and $\operatorname{erf}''$ are
elementary, so the extra derivative costs almost nothing.

### Dawson's integral

**Dawson's integral** $F(x) = e^{-x^2}\int_0^x e^{t^2}\,dt$ appears in the theory of plasma oscillations, in
spectral line shapes (it is the imaginary part of the Faddeeva function on the real axis) and in heat
conduction. Rybicki's method starts from the sampling theorem, which gives the exact limit

$$
F(x) = \lim_{h \to 0}\frac1{\sqrt\pi}\sum_{n\text{ odd}}\frac{e^{-(x - nh)^2}}{n},
$$

and makes it practical by centring the sum on the even multiple of $h$ nearest $x$, so that with $h = 0.2$
and twenty terms each side the error is about $10^{-15}$. For $|x| < 0.2$ the Taylor series
$x - \tfrac23x^3 + \tfrac4{15}x^5 - \cdots$ is used.

### Accuracy

<p class="caption" id="tab-6-2"><strong>Table 6.2.</strong> Columns: function, test points, median relative error, largest relative error, and more.</p>

{{#include ../generated/special_functions/incomplete_accuracy.md}}

## Algorithm

**$P(a, x)$ and $Q(a, x)$.**

1. If $x < a + 1$: sum the series for $P$, multiplying each term by $x/(a + n)$ to get the next,
   until a term is below $\varepsilon$ times the sum. Then $Q = 1 - P$.
2. Otherwise: evaluate the continued fraction for $Q$ by the modified Lentz algorithm (below).
   Then $P = 1 - Q$.

In both cases the common factor $x^ae^{-x}/\Gamma(a)$ is computed as
$\exp(a\ln x - x - \ln\Gamma(a))$ to avoid overflow.

**$I_x(a, b)$.**

1. Return 0 or 1 at the ends $x = 0$ and $x = 1$.
2. If $x < (a + 1)/(a + b + 2)$: evaluate the continued fraction by the modified Lentz algorithm
   and multiply by the prefactor.
3. Otherwise: compute $I_{1-x}(b, a)$ by step 2 and return 1 minus it.

**$\operatorname{erf}$ and $\operatorname{erfc}$.** For $x \ge 0$, return $P(\tfrac12, x^2)$ and $Q(\tfrac12, x^2)$. For
$x < 0$ use the symmetries $\operatorname{erf}(-x) = -\operatorname{erf}x$ and $\operatorname{erfc}(-x) = 2 - \operatorname{erfc}x$.

**The modified Lentz algorithm** evaluates a continued fraction
$b_0 + \cfrac{a_1}{b_1 + \cfrac{a_2}{b_2 + \cdots}}$ from the top down.

1. Set $f = b_0$, $C = f$ and $D = 0$.
2. For $j = 1, 2, \ldots$: set $D \leftarrow b_j + a_jD$ and $C \leftarrow b_j + a_j/C$; then
   $D \leftarrow 1/D$; then multiply $f$ by $\Delta = CD$.
3. Stop when $|\Delta - 1|$ is below a few $\varepsilon$.

$C$ and $D$ are ratios of successive numerators and of successive denominators of the
convergents, so $\Delta$ is the ratio of one convergent to the previous one. The modification
is a guard: whenever $f$, $C$ or $D$ (before inversion) is exactly zero it is replaced by a tiny
number such as $10^{-30}$, which prevents a division by zero without affecting the result.

## Implementation

The continued fraction for $Q(a, x)$ by the modified Lentz algorithm:

```rust,ignore
{{#include ../../../crates/special_functions/src/incomplete_gamma.rs:gamma_cf}}
```

The continued fraction for $I_x(a, b)$:

```rust,ignore
{{#include ../../../crates/special_functions/src/incomplete_beta.rs:beta_cf}}
```

- Arguments outside the domain ($a \le 0$, $x < 0$, $x \notin [0, 1]$ for the beta function) give `NaN`.
- The number of terms grows like $\sqrt a$ near the transition $x \approx a$: parameters of a few thousand are
  fine; for larger ones, uniform asymptotic expansions (Temme) would be faster.

## In practice

```rust,ignore
{{#include ../../../crates/special_functions/examples/special_incomplete.rs}}
```

**Strengths**

- Full relative accuracy in both tails, through the direct computation of the smaller of $P$ and $Q$.
- One continued-fraction engine serves the gamma, beta and error functions.

**Weaknesses**

- Slow for very large parameters, where asymptotic expansions are better.
- Real arguments only (no complex error function or Faddeeva function).

## Exercises

1. Show that $P(1, x) = 1 - e^{-x}$ and that $P(n, x)$ for integer $n$ is the probability of at least $n$ events
   of a Poisson process of rate 1 by time $x$.
2. Show that $\operatorname{erf} x = P(\tfrac12, x^2)$ by substituting $t = s^2$.
3. Derive the series for $P(a, x)$ by repeated integration by parts of $\int_0^x t^{a-1}e^{-t}\,dt$.
4. Compute $\operatorname{erfc} x$ for $x = 1, 2, \ldots, 25$ and compare with the asymptotic formula $e^{-x^2}/(x\sqrt\pi)$. At what $x$ do
   they agree to six digits?
5. Implement the modified Lentz algorithm for the continued fraction of $\tan x$ and compare with `f64::tan`.

<details>
<summary>Hints and answers</summary>

1. $P(1, x) = \int_0^xe^{-t}dt = 1 - e^{-x}$. For integer $n$, $P(n, x) = 1 - e^{-x}\sum_{k<n}x^k/k!$, which is the probability that a Poisson process of rate 1 has at least $n$ events by time $x$.
2. $\operatorname{erf}x = \tfrac2{\sqrt\pi}\int_0^xe^{-s^2}ds$; with $t = s^2$ this is $\tfrac1{\Gamma(1/2)}\int_0^{x^2}t^{-1/2}e^{-t}dt = P(\tfrac12, x^2)$.
3. Integrate by parts: $\int_0^xt^{a-1}e^{-t}dt = \tfrac{x^a}ae^{-x} + \tfrac1a\int_0^xt^ae^{-t}dt$. Repeating gives $\gamma(a, x) = x^ae^{-x}\sum_{n\ge0}x^n/(a(a + 1)\cdots(a + n))$.
4. The first correction to the asymptotic formula is a relative $-1/(2x^2)$, so even at $x = 25$ the formula is off by about $8\times10^{-4}$: six-digit agreement is not reached anywhere in $1 \le x \le 25$ (and the function underflows near $x = 26.5$).
5. Use $\tan x = x/(1 - x^2/(3 - x^2/(5 - \cdots)))$ with the modified Lentz algorithm (tiny-value protection for zero denominators); the continued fraction converges quickly for $|x| < 1$ but needs more terms near $\pi/2$.

</details>

## Further reading

- W. J. Lentz, "Generating Bessel functions in Mie scattering calculations using continued fractions",
  *Applied Optics* 15 (1976) 668–671.
- I. J. Thompson and A. R. Barnett, "Coulomb and Bessel functions of complex arguments and order",
  *Journal of Computational Physics* 64 (1986) 490–509.
- A. R. DiDonato and A. H. Morris, "Computation of the incomplete gamma function ratios and their
  inverse", *ACM Transactions on Mathematical Software* 12 (1986) 377–393.
- G. B. Rybicki, "Dawson's integral and the sampling theorem", *Computers in Physics* 3 (1989) 85–87.
- NIST Digital Library of Mathematical Functions, chapters 7 and 8, <https://dlmf.nist.gov/7>.
