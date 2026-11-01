# Exponential, sine, cosine and Fresnel integrals

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Evaluate the exponential integrals $E_1(x)$, $E_n(x)$ and $\operatorname{Ei}(x)$, the sine and cosine integrals $\operatorname{Si}(x)$ and $\operatorname{Ci}(x)$, and the Fresnel integrals $S(x)$ and $C(x)$ |
| **Devised by** | Exponential and logarithmic integrals: Leonhard Euler (1768), Johann von Soldner (1809); sine and cosine integrals: Euler, Lorenzo Mascheroni (1790); Fresnel integrals: Euler (1744), Augustin-Jean Fresnel (1818); the Cornu spiral: Marie Alfred Cornu (1874) |
| **Accuracy** | About $10^{-15}$ relative, $10^{-14}$ for the Fresnel integrals at very large arguments |
| **Cost** | A power series for small arguments, a (complex) continued fraction or asymptotic series for large ones |
| **API** | [`exponential_integral`](../../api/lib_jsl_special_functions/exponential_integrals/fn.exponential_integral.html), [`exponential_integral_en`](../../api/lib_jsl_special_functions/exponential_integrals/fn.exponential_integral_en.html), [`exponential_integral_ei`](../../api/lib_jsl_special_functions/exponential_integrals/fn.exponential_integral_ei.html), [`sine_integral`](../../api/lib_jsl_special_functions/sine_integrals/fn.sine_integral.html), [`sine_integral_shifted`](../../api/lib_jsl_special_functions/sine_integrals/fn.sine_integral_shifted.html), [`cosine_integral`](../../api/lib_jsl_special_functions/cosine_integrals/fn.cosine_integral.html), [`fresnel_integrals`](../../api/lib_jsl_special_functions/fresnel_integrals/fn.fresnel_integrals.html) |

</div>

## Overview

Integrating an exponential, sine or cosine divided by its argument, or of the square of the
argument, gives functions that cannot be written in closed form but appear constantly. The
**exponential integrals** arise in radiative transfer, groundwater flow (the Theis well function),
and number theory, where the logarithmic integral $\operatorname{li}(x) = \operatorname{Ei}(\ln x)$ counts the primes. The **sine and
cosine integrals** describe the Gibbs phenomenon, the radiation of antennas and the response of
ideal filters. The **Fresnel integrals** describe the diffraction of light at an edge, and the
clothoid curves of road and railway design.

All of them are related through the exponential integral of a complex argument, which is how the
library computes them.

## History

Euler introduced the logarithmic and exponential integrals in his *Institutiones calculi integralis*
(1768). Johann von Soldner, a Bavarian astronomer, tabulated the logarithmic integral in 1809 (it is
sometimes called Soldner's integral), and Gauss and Legendre conjectured that it approximates the
number of primes below $x$, which became the prime number theorem (1896). Lorenzo Mascheroni studied
the sine and cosine integrals in 1790, computing Euler's constant $\gamma$ to 32 digits on the way (19 of
them correct).

Euler met the Fresnel integrals in 1744 while studying elastic curves. Augustin-Jean Fresnel used
them in 1818 to explain diffraction by the wave theory of light, and Marie Alfred Cornu (1874) drew
the spiral that bears his name, an indispensable graphical tool for diffraction calculations before
computers.

## The idea

### Exponential integrals

For $x > 0$,

$$
E_1(x) = \int_x^\infty\frac{e^{-t}}{t}\,dt, \qquad E_n(x) = \int_1^\infty\frac{e^{-xt}}{t^n}\,dt, \qquad \operatorname{Ei}(x) = -\,\mathrm{PV}\!\!\int_{-x}^\infty\frac{e^{-t}}{t}\,dt,
$$

the last a principal value defined for all real $x \ne 0$, with $\operatorname{Ei}(-x) = -E_1(x)$. For small $x$,

$$
E_1(x) = -\gamma - \ln x - \sum_{k=1}^\infty\frac{(-x)^k}{k\cdot k!},
$$

a series that converges for every $x$ but suffers cancellation for large $x$, where $E_1(x) \approx e^{-x}/x$. For
$x > 1$ the library uses the continued fraction

$$
E_n(x) = e^{-x}\cfrac{1}{x + n - \cfrac{1\cdot n}{x + n + 2 - \cfrac{2(n + 1)}{x + n + 4 - \cdots}}},
$$

evaluated by the modified Lentz algorithm. $\operatorname{Ei}(x)$ for $x > 0$ uses its power series
$\gamma + \ln x + \sum x^k/(k\cdot k!)$ for moderate $x$ (all terms positive, no cancellation) and the asymptotic series
$\frac{e^x}{x}\sum k!/x^k$ for large $x$.

### Sine and cosine integrals

$$
\operatorname{Si}(x) = \int_0^x\frac{\sin t}{t}\,dt, \qquad \operatorname{Ci}(x) = \gamma + \ln x + \int_0^x\frac{\cos t - 1}{t}\,dt .
$$

$\operatorname{Si}$ is odd and tends to $\pm\pi/2$; $\operatorname{Ci}$ tends to 0, oscillating, and to $-\infty$ logarithmically at 0. Both come from the
exponential integral of an imaginary argument,

$$
E_1(ix) = -\operatorname{Ci}(x) + i\big(\operatorname{Si}(x) - \tfrac\pi2\big),
$$

so for large $x$ the library evaluates the continued fraction for $E_1$ in complex arithmetic, and for small $x$
the power series. The value $\tfrac2\pi\operatorname{Si}(\pi) = 1.17898$ is the size of the overshoot of a Fourier series at a jump,
the **Gibbs phenomenon**.

![Sine, cosine and exponential integrals](../generated/special_functions/sine_cosine_integrals.svg)

### Fresnel integrals

$$
S(x) = \int_0^x\sin\!\big(\tfrac\pi2t^2\big)\,dt, \qquad C(x) = \int_0^x\cos\!\big(\tfrac\pi2t^2\big)\,dt .
$$

Both tend to $\tfrac12$ as $x \to \infty$, oscillating with decreasing amplitude. Plotting $(C(t), S(t))$ gives the **Cornu
spiral**, or clothoid, a curve whose curvature grows linearly with arc length, which is why it joins
straight track to circular curves smoothly:

![The Cornu spiral](../generated/special_functions/cornu_spiral.svg)

$C + iS$ is an error function of a complex argument, $\frac{1+i}2\operatorname{erf}\big(\frac{\sqrt\pi}2(1 - i)x\big)$. The library uses the power
series for small $x$ and, for large $x$, a complex continued fraction for $\operatorname{erfc}$.

### Accuracy

{{#include ../generated/special_functions/exponential_accuracy.md}}

## Algorithm

- $E_n(x)$: the continued fraction for $x > 1$, the series otherwise ($E_0 = e^{-x}/x$ in closed form).
- $\operatorname{Ei}(x)$: $-E_1(-x)$ for $x < 0$; the power series, or the asymptotic series beyond the point where its terms
  start to grow.
- $\operatorname{Si}$, $\operatorname{Ci}$, $S$, $C$: power series below a threshold, the complex continued fraction above it.

## Implementation

```rust,ignore
{{#include ../../../crates/special_functions/src/exponential_integrals.rs:en}}
```

- `sine_integral_shifted` returns $\operatorname{Si}(x) - \pi/2$, accurate for large $x$ where $\operatorname{Si}$ itself is close to $\pi/2$.
- `fresnel_integrals(x)` returns the pair `(S(x), C(x))`, with the $\tfrac\pi2t^2$ convention.

## In practice

```rust,ignore
{{#include ../../../crates/special_functions/examples/special_exponential_integrals.rs}}
```

**Strengths**

- Full accuracy across the whole real line, from one continued-fraction engine.

**Weaknesses**

- Real arguments only (the complex exponential integral is computed internally but not exposed).

## Exercises

1. Show that $E_{n+1}(x) = \frac1n\big(e^{-x} - xE_n(x)\big)$, and explain why this recurrence is unstable upward for $x > n$.
2. Derive the asymptotic series of $E_1(x)$ by repeated integration by parts, and find the term at which
   it should be truncated for $x = 10$.
3. Show that $\operatorname{li}(x) = \operatorname{Ei}(\ln x)$, and compare $\operatorname{li}(10^k)$ with the number of primes below $10^k$ for
   $k = 3, \ldots, 9$.
4. Prove that the curvature of the Cornu spiral at parameter $t$ is $\pi t$.
5. Compute the Gibbs overshoot by summing the Fourier series of a square wave to $N$ terms, and compare
   its maximum with $\tfrac2\pi\operatorname{Si}(\pi)$.

## Further reading

- W. Gautschi and W. F. Cahill, "Exponential integral and related functions", in M. Abramowitz and
  I. A. Stegun (eds.), *Handbook of Mathematical Functions*, chapter 5, 1964.
- NIST Digital Library of Mathematical Functions, chapters 6 and 7, <https://dlmf.nist.gov/6>.
- W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery, *Numerical Recipes*, 3rd ed.,
  Cambridge University Press, 2007, §6.3 and §6.8–6.9.
