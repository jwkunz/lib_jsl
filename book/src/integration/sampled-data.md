# Integrating sampled data

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Integrate a function known only at fixed samples, equally or unequally spaced, and compute its running integral |
| **Devised by** | The trapezoid rule (antiquity); Thomas Simpson (1743), after Bonaventura Cavalieri and James Gregory; Werner Romberg (1955); unequal-spacing and end corrections: Kirk Cartwright (2017) |
| **Accuracy** | Trapezoid $O(h^2)$, Simpson $O(h^4)$, Romberg up to $O(h^{2k+2})$ for $2^k + 1$ samples; agrees with `scipy.integrate` to rounding |
| **Cost** | $O(n)$ for every rule |
| **Read first** | [Newton-Cotes rules](newton-cotes.md), [Romberg integration](romberg.md) |
| **API** | [`trapezoid`](../../api/lib_jsl_integration/sampled/fn.trapezoid.html), [`simpson`](../../api/lib_jsl_integration/sampled/fn.simpson.html), [`romb`](../../api/lib_jsl_integration/sampled/fn.romb.html), [`cumulative_trapezoid`](../../api/lib_jsl_integration/sampled/fn.cumulative_trapezoid.html), [`cumulative_simpson`](../../api/lib_jsl_integration/sampled/fn.cumulative_simpson.html), [`Spacing`](../../api/lib_jsl_integration/sampled/enum.Spacing.html) |

</div>

## Overview

The [Newton-Cotes](newton-cotes.md) and [Romberg](romberg.md) entries integrate a *function*, choosing where to evaluate it. Measured data
come as fixed samples, often unequally spaced: a logged time series, a measured spectrum, an experiment's output. The same rules apply,
rewritten for given points. This entry provides SciPy's set of them:

- `trapezoid` and `simpson` for the total integral;
- `romb` for $2^k + 1$ equally spaced samples;
- `cumulative_trapezoid` and `cumulative_simpson` for the running integral at every sample, such as velocity from sampled acceleration.

## History

The trapezoid rule is ancient. Thomas Simpson popularised the parabolic rule in 1743, though Bonaventura Cavalieri and James Gregory had
used it a century earlier. Werner Romberg (1955) showed how Richardson extrapolation of trapezoid sums on halving grids reaches high
order from equally spaced data. Unequally spaced Simpson rules were long a matter of folklore. Kirk Cartwright (2017) gave the end
correction for an even number of samples that SciPy adopted in 2023, replacing an average of two inconsistent rules.

## The idea

### Rules for given samples

Each rule integrates a simple interpolant of the samples:

- **Trapezoid**: the piecewise-linear interpolant, $\sum_i \tfrac12(y_i + y_{i+1})h_i$. It works for any spacing, and its error is $O(h^2)$.
- **Simpson**: a parabola through each pair of intervals. For unequal widths $h_0, h_1$, the integral of the parabola through three samples is
  $$\frac{h_0 + h_1}{6}\Big[\Big(2 - \frac{h_1}{h_0}\Big)y_0 + \frac{(h_0 + h_1)^2}{h_0h_1}y_1 + \Big(2 - \frac{h_0}{h_1}\Big)y_2\Big],$$
  which for equal widths is the familiar $\tfrac h3(y_0 + 4y_1 + y_2)$. With an even number of samples, one interval is left over. Cartwright's correction integrates it with the parabola through the last three
  samples, which keeps the rule exact for quadratics.
- **Romberg**: trapezoid sums with spacings $h, 2h, 4h, \ldots$ come from subsets of the same $2^k + 1$ samples. Richardson extrapolation removes the error terms $h^2, h^4, \ldots$ one by one.

For smooth functions the orders show directly:

<p class="caption" id="tab-2-8"><strong>Table 2.8.</strong> Errors for ∫₀³ (e^(−x) sin 2x + 0.1x²) dx from equally spaced samples.</p>

{{#include ../generated/integration/sampled_convergence.md}}

Each doubling divides the trapezoid error by 4 and Simpson's by 16. Romberg's improves faster still, reaching rounding level with 129 samples.
It also needs exactly $2^k + 1$ equally spaced samples, while the other two rules accept any.

### Running integrals

`cumulative_trapezoid` adds the trapezoids one at a time. `cumulative_simpson` needs an integral over *each single interval*. For each interval it integrates
the parabola through that interval and its neighbour: the first interval of each pair uses the parabola through the pair, and
the second uses the parabola through the pair seen from the other end. Every partial sum is then exact for quadratics, so the running integral
of sampled data is accurate to $O(h^3)$ at every sample, not just at the end.

## Algorithm

**`simpson(y, spacing)`**: for an odd number of samples, sum the three-point rule over pairs of intervals. For an even number, do the same over all but
the last interval, then add $\alpha y_{n-1} + \beta y_{n-2} - \eta y_{n-3}$ with Cartwright's coefficients
$\alpha = \frac{2h_1^2 + 3h_0h_1}{6(h_0 + h_1)}$, $\beta = \frac{h_1^2 + 3h_0h_1}{6h_0}$, $\eta = \frac{h_1^3}{6h_0(h_0 + h_1)}$.

**`romb(y, dx)`**: $R_{0,0} = \tfrac12(y_0 + y_n)\cdot nh$; for each level, add the new midpoints to the trapezoid sum, then extrapolate along the row.

## Implementation

```rust,ignore
{{#include ../../../crates/integration/src/sampled.rs:simpson}}
```

```rust,ignore
{{#include ../../../crates/integration/src/sampled.rs:romb}}
```

- `Spacing::Uniform(dx)` or `Spacing::Points(&x)` gives the sample positions, as SciPy's `dx` or `x`.
- The cumulative functions return one value per interval, or one per sample when `initial` is given.

## In practice

Every rule is compared with `scipy.integrate` from SciPy 1.17.1 (references from `book/generator/data/make_sampled_refs.py`):

<p class="caption" id="tab-2-9"><strong>Table 2.9.</strong> Sampled-data integration against SciPy.</p>

{{#include ../generated/integration/sampled_scipy.md}}

**Strengths**

- Any spacing for the trapezoid and Simpson rules, with exact formulas for unequal intervals.
- Running integrals with Simpson accuracy at every sample.

**Weaknesses**

- The rules assume a smooth integrand between samples. Noisy data are better smoothed first, for example by a [smoothing spline](../interpolation/b-splines.md#smoothing-splines), which can then be integrated exactly.
- Romberg needs $2^k + 1$ equally spaced samples.

## Exercises

1. Show that the unequal-spacing Simpson formula reduces to $\tfrac h3(y_0 + 4y_1 + y_2)$ when $h_0 = h_1 = h$.
2. Show that the trapezoid rule is exact for linear functions on any spacing, and Simpson's for cubics on equal spacing with an odd number of samples.
3. Show that Simpson's rule on equal spacing is the first Richardson extrapolation of the trapezoid rule, $\tfrac13(4T_h - T_{2h})$.

<details>
<summary>Hints and answers</summary>

1. With $h_0 = h_1 = h$: $\tfrac{2h}{6}(y_0 + 4y_1 + y_2)$, since $(2h)^2/h^2 = 4$ and $2 - 1 = 1$.
2. Linear interpolants of linear functions are exact. For cubics, the error of the parabola, $f'''(\xi)(x - x_0)(x - x_1)(x - x_2)/6$, integrates to zero over a symmetric pair of intervals because the cubic factor is odd about $x_1$.
3. $\tfrac13\big(4\cdot\tfrac h2(y_0 + 2y_1 + y_2) - h(y_0 + y_2)\big) = \tfrac h3(y_0 + 4y_1 + y_2)$.

</details>

## Further reading

- P. J. Davis and P. Rabinowitz, *Methods of Numerical Integration*, 2nd ed., Academic Press, 1984, chapter 2.
- K. V. Cartwright, "Simpson's rule cumulative integration with MS Excel and irregularly-spaced data", *Journal of Mathematical Sciences and Mathematics Education* 12 (2017) 1–9.
- W. Romberg, "Vereinfachte numerische Integration", *Det Kongelige Norske Videnskabers Selskab Forhandlinger* 28 (1955) 30–36.
