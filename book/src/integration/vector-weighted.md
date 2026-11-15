# Vector integrands and weighted quadrature

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Approximate $\int_a^b \mathbf f(x)\,dx$ for a vector-valued $\mathbf f$, and $\int_a^b f(x)\,w(x)\,dx$ for a weight $w$ that oscillates or is singular |
| **Devised by** | Robert Piessens, Elise de Doncker-Kapenga, Christoph Überhuber and David Kahaner (QUADPACK, 1983); Peter Wynn (the ε algorithm, 1956); Louis Filon (1928), Charles Clenshaw and Alan Curtis (1960), the Filon-Clenshaw-Curtis rule |
| **Accuracy** | Set by the requested tolerance; Gauss-Kronrod error estimates are usually pessimistic by orders of magnitude |
| **Cost** | `quad_vec`: one evaluation of $\mathbf f$ per node, shared by every component. `quad_weighted`: tens to hundreds of evaluations of $f$; for $\cos\omega x$ and $\sin\omega x$ on a finite interval, independent of $\omega$ |
| **Read first** | [Adaptive Gauss-Kronrod quadrature](gauss-kronrod.md), [Tanh-sinh quadrature](tanh-sinh.md) |
| **API** | [`quad_vec`](../../api/lib_jsl_integration/quad_vec/fn.quad_vec.html), [`QuadVecOptions`](../../api/lib_jsl_integration/quad_vec/struct.QuadVecOptions.html), [`quad_weighted`](../../api/lib_jsl_integration/weighted/fn.quad_weighted.html), [`Weight`](../../api/lib_jsl_integration/weighted/enum.Weight.html) |

</div>

## Overview

Two extensions of [adaptive Gauss-Kronrod quadrature](gauss-kronrod.md) are covered here.

The first handles **vector integrands**. Many problems need many integrals of closely related functions, for example a family of Bessel or Fourier coefficients, the
moments of a distribution, or a response computed at each of a hundred frequencies. Integrating each component on its own repeats the
work of locating the integrand's difficult regions, and evaluates any expensive shared part of the integrand once per component.
SciPy's `quad_vec` integrates the whole vector at once. All the components share one set of subintervals, and the refinement is
driven by a norm of the vector error. The library's `quad_vec` is a port of it, and it reproduces SciPy's results exactly, including the number of evaluations.

The second covers **weighted integrals**, in which the integrand is a smooth $f$ times a known, troublesome weight:

- an oscillating factor, $\cos\omega x$ or $\sin\omega x$, possibly on $[a, \infty)$ (a Fourier integral);
- an algebraic-logarithmic endpoint singularity, $(x - a)^\alpha (b - x)^\beta$, possibly multiplied by $\ln(x - a)$, $\ln(b - x)$ or both;
- a pole inside the interval, $1/(x - c)$, for which the integral exists only as a **Cauchy principal value**.

SciPy's `quad(f, a, b, weight=…)` handles these with QUADPACK's QAWO, QAWF, QAWS and QAWC routines. The library's `quad_weighted`
accepts the same cases. For oscillation on a finite interval it uses a Filon-Clenshaw-Curtis rule, the idea behind QAWO, which integrates
the oscillation exactly. For the other weights it builds on other methods in this chapter, as explained under [Weighted integrals](#weighted-integrals).

## History

QUADPACK, a Fortran library by Robert Piessens, Elise de Doncker-Kapenga, Christoph Überhuber and David Kahaner, was
published as a book in 1983 and became the standard for one-dimensional quadrature. Its central routine QAGS combines 21-point
Gauss-Kronrod rules with global adaptive bisection and extrapolation. The weighted routines use **modified Clenshaw-Curtis**
rules. They expand the smooth factor $f$ in Chebyshev polynomials and integrate each polynomial against the weight exactly, from
moments that are computed by recurrences. Piessens and Branders developed these rules for oscillatory and singular weights in the 1970s.
They go back to Louis Filon (1928), who integrated a quadratic interpolant of $f$ against $\cos\omega x$ exactly. With Charles Clenshaw and Alan Curtis's
Chebyshev-point interpolation (1960) in place of Filon's quadratics, the **Filon-Clenshaw-Curtis** rule's error falls as both the number of points and the frequency grow.
Domínguez, Graham and Smyshlyaev (2011) proved its stability.

For Fourier integrals on $[a, \infty)$, QAWF follows a much older idea. The integral over each half-period of the oscillation
is a term of an alternating series, and the series' limit can be found by extrapolation. Peter Wynn's **ε algorithm** (1956) is an
efficient recursive form of Shanks' transformation (1955), and it is the extrapolation QUADPACK uses.

Vector-valued quadrature with a shared mesh is newer in general-purpose libraries. SciPy added `quad_vec` in version 1.4 (2019).
It is written in Python around the same Gauss-Kronrod rules, with heuristics for refining many subintervals in each round.

## The idea

### One mesh for many integrals

For a vector $\mathbf f(x) = (f_1(x), \dots, f_m(x))$, apply a Gauss-Kronrod pair to every component on each subinterval
$[x_0, x_1]$. Each component $j$ gets an estimate $K_j$ and an embedded Gauss estimate $G_j$, and the subinterval's error estimate is a norm
of the vector of differences,

$$
E = \bigl\| h\,(\mathbf K - \mathbf G) \bigr\|, \qquad h = \tfrac12 (x_1 - x_0),
$$

rescaled as in QUADPACK. The norm is either the Euclidean norm (SciPy's default, `'2'`) or the largest component (`'max'`). The
global error is the sum of the subinterval errors. Global adaptive bisection then works just as in the scalar case: the subintervals with the largest errors are
split until the global error meets the tolerance $\max(\varepsilon_{\text{abs}}, \varepsilon_{\text{rel}}\|\mathbf I\|)$.

A subinterval is refined if *any* component is difficult on it, so every component is evaluated on a mesh that is fine enough for all of them. For related
integrands this costs little, since their difficult regions coincide. The integrand is called once per node and returns a whole vector, so any work common to the
components (a matrix factorisation, a simulation step, a call to a special function) is done once.

### Infinite intervals

An infinite range is mapped to a finite one before the subdivision starts. On $[a, \infty)$, SciPy uses $x = a + (1 - t)/t$ with
$t \in (0, 1]$, and on $(-\infty, \infty)$ it uses $x = (1 - |t|)/t$ on $[-1, 1]$, split at $t = 0$:

$$
\int_a^\infty \mathbf f(x)\,dx = \int_0^1 \mathbf f\!\left(a + \frac{1 - t}{t}\right)\frac{dt}{t^2}.
$$

The 15-point Kronrod rule is the default on the mapped interval. Break points given by the user are mapped too.

### Weighted integrals

A weight such as $\cos 25x$ or $x^{-1/2}\ln x$ ruins the polynomial behaviour that Gauss-Kronrod relies on. Each case has its own remedy.

**Oscillation on a finite interval.** A rule that samples $f(x)\cos\omega x$ needs several points per period, so its cost grows with $\omega$. The
**Filon-Clenshaw-Curtis** rule samples only $f$. On a subinterval with midpoint $c$ and half-width $h$, substitute $x = c + ht$ and interpolate $g(t) = f(c + ht)$ at the
Chebyshev points $t_j = \cos(j\pi/N)$ by $p(t) = \sum_{k=0}^N c_kT_k(t)$. The coefficients come from a cosine transform of the values. Then

$$
\int_{c-h}^{c+h} f(x)\,e^{i\omega x}\,dx \approx h\,e^{i\omega c}\sum_{k=0}^N c_k\,\mu_k(\omega h), \qquad \mu_k(\kappa) = \int_{-1}^1 T_k(t)\,e^{i\kappa t}\,dt,
$$

and the cosine and sine integrals are its real and imaginary parts. The **modified moments** $\mu_k$ do not depend on $f$. The library computes them from the Jacobi-Anger
expansion $e^{i\kappa t} = J_0(\kappa) + 2\sum_{m\ge1} i^mJ_m(\kappa)T_m(t)$ and the closed form of $\int_{-1}^1 T_kT_m\,dt$. The Bessel values come from Miller's backward recurrence,
which is stable for all orders. The accuracy depends only on how well $p$ approximates $f$, measured by the size of its last coefficients: the frequency costs nothing.

**Fourier integrals.** On $[a, \infty)$ there are infinitely many half-periods. Let $S_k$ be the integral up to the $k$-th zero of the
weight. If $f$ decays monotonically, the terms $S_k - S_{k-1}$ alternate in sign and shrink, but the partial sums may converge very slowly: for
$f(x) = 1/\sqrt x$ the error after $k$ half-periods is of order $k^{-1/2}$. Wynn's ε algorithm extrapolates the sequence. Starting from
$\varepsilon^{(n)}_{-1} = 0$ and $\varepsilon^{(n)}_0 = S_n$, it builds the table

$$
\varepsilon^{(n)}_{k+1} = \varepsilon^{(n+1)}_{k-1} + \frac{1}{\varepsilon^{(n+1)}_k - \varepsilon^{(n)}_k},
$$

whose even columns $\varepsilon^{(n)}_{2k}$ are Shanks' estimates of the limit. They are exact when $S_n$ is a constant plus a sum of $k$
geometric sequences, and they converge much faster than $S_n$ for alternating series. The change between the two latest estimates in an even column
serves as the error estimate.

**Algebraic-logarithmic singularities.** On a finite interval, $(x - a)^\alpha (b - x)^\beta$ with $\alpha, \beta > -1$ is integrable but singular, or
infinitely steep, at the ends. Logarithmic factors make it worse. This is the situation [tanh-sinh quadrature](tanh-sinh.md) is designed for:
its change of variables clusters the nodes double exponentially near the endpoints and makes the singularities harmless. The weight
needs the distances $x - a$ and $b - x$ to full relative accuracy, and these are supplied exactly by the complement form of the tanh-sinh rule.

**Cauchy principal values.** For $a < c < b$ the integral of $f(x)/(x - c)$ diverges, but the symmetric limit

$$
\operatorname{p.v.}\!\int_a^b \frac{f(x)}{x - c}\,dx = \lim_{\epsilon\to 0}\left( \int_a^{c-\epsilon} + \int_{c+\epsilon}^b \right) \frac{f(x)}{x - c}\,dx
$$

exists when $f$ is smooth near $c$. With $d = \min(c - a,\, b - c)$, the parts of the integral at the same distance $t$ on either side of $c$
can be paired:

$$
\operatorname{p.v.}\!\int_{c-d}^{c+d} \frac{f(x)}{x - c}\,dx = \int_0^d \frac{f(c + t) - f(c - t)}{t}\,dt.
$$

The right-hand integrand tends to $2f'(c)$ as $t \to 0$, so it is smooth, and the remaining part of $[a, b]$ outside $[c - d, c + d]$ is an ordinary integral.

## Algorithm

`quad_vec` follows SciPy's implementation step by step:

1. If a limit is infinite, map the integrand and the break points as above, and use the 15-point rule unless another is chosen.
2. Apply the rule (21-point Gauss-Kronrod by default) to each piece of $[a, b]$ cut at the break points. Sum the integrals and the errors.
3. Keep the subintervals in a priority queue ordered by error. In each round, pop the worst subintervals, up to 128 of them, while
   the remaining error is more than an eighth of the tolerance. Bisect each, and update the global integral and error by the differences.
4. After each round, stop when the global error falls below an eighth of the tolerance, when it falls below the accumulated rounding error, or when the number of subintervals
   reaches `limit` (status 1). The test comes after the bisections, so even a polynomial that the first rule integrates exactly costs three applications of it.

`quad_weighted` reduces each case to `quad_vec` with one component, or to tanh-sinh:

1. **Cos / Sin on $[a, b]$:** on each subinterval, interpolate $f$ at 17, then 33, then 65 nested Chebyshev points, until the last three coefficients, times $2h$, are below the tolerance. Integrate the interpolant exactly against $e^{i\omega x}$, and bisect the subinterval with the largest error estimate until the sum meets the tolerance (at most 200 subintervals).
2. **Cos / Sin on $[a, \infty)$:** integrate over $[a, z_1]$, then over the half-periods $[z_k, z_{k+1}]$ between the weight's zeros. After each
   term, extrapolate the partial sums (the last 40 at most) with the ε algorithm, and stop once at least eight terms have been
   summed and the extrapolation error meets the tolerance.
3. **Algebraic:** tanh-sinh quadrature of $f(x)w(x)$, with the weight formed from the exact endpoint distances.
4. **Cauchy:** the symmetric part $\int_0^d (f(c + t) - f(c - t))/t\,dt$, plus the ordinary integral of $f(x)/(x - c)$ over what is left of $[a, b]$.

## Implementation

The local rule for a vector integrand computes the Kronrod and Gauss sums for every component together, then forms QUADPACK's
error estimate in the chosen norm, bounded below by an estimate of the rounding error:

```rust,ignore
{{#include ../../../crates/integration/src/quad_vec.rs:gk}}
```

The driver maps infinite intervals, then runs the adaptive loop on a finite one:

```rust,ignore
{{#include ../../../crates/integration/src/quad_vec.rs:quad_vec}}
```

Some details:

- **The number of components** is found by evaluating $\mathbf f$ once at a finite point of the original range, before the
  mapping. This evaluation is not counted, so the evaluation counts are the same as SciPy's.
- **The priority queue** orders subintervals by error, then by their end points, just as Python's `heapq` orders the tuples
  $(-E, x_0, x_1)$. Ties are therefore broken as in SciPy, and the two implementations bisect the same subintervals in the same order.
- **The trapezoid rule** (`VecRule::Trapezoid`), for integrands that are not smooth enough for Gauss-Kronrod, keeps the values computed
  for a subinterval so that its two halves reuse them, as SciPy does with a cache.
- **The result** reports the evaluation count, the number of subintervals and a status (0 converged, 1 subinterval limit reached).

The weighted integrals:

```rust,ignore
{{#include ../../../crates/integration/src/weighted.rs:quad_weighted}}
```

The Filon-Clenshaw-Curtis rule and its moments:

```rust,ignore
{{#include ../../../crates/integration/src/weighted.rs:filon}}
```

```rust,ignore
{{#include ../../../crates/integration/src/weighted.rs:fcc_moments}}
```

The ε algorithm keeps only two columns of the table at a time:

```rust,ignore
{{#include ../../../crates/integration/src/weighted.rs:wynn}}
```

## In practice

The references are from SciPy 1.17.1, written by `book/generator/data/make_quadode_refs.py`. `quad_vec` reproduces
SciPy's subdivision exactly. The values agree to the last bit, and so do the numbers of evaluations and subintervals; only the error estimates
differ in their last digits:

<p class="caption" id="tab-2-14"><strong>Table 2.14.</strong> `quad_vec` against SciPy.</p>

{{#include ../generated/integration/quad_vec_scipy.md}}

Weighted integrals, compared with SciPy's `quad(weight=…)`:

<p class="caption" id="tab-2-15"><strong>Table 2.15.</strong> `quad_weighted` against SciPy's QUADPACK routines.</p>

{{#include ../generated/integration/weighted_scipy.md}}

The algebraic and Cauchy results agree with QUADPACK to between $10^{-14}$ and the last bit, and the first oscillatory one to $10^{-14}$. On $\sin 40x/(1 + x^2)$ the
library's result is $1.4 \times 10^{-9}$ (relative) from a reference computed to $10^{-13}$, and QUADPACK's is closer. Both lie well within the default absolute tolerance of $1.49 \times 10^{-8}$. The three Fourier
integrals agree only to $10^{-8}$ or $10^{-9}$, as expected from extrapolations with the default tolerance of $1.49 \times 10^{-8}$. The exact
value of the first is $\tfrac\pi2 e^{-3} = 0.07820534411\ldots$. SciPy's result is $3 \times 10^{-11}$ from it, and the library's $2 \times 10^{-10}$.

For oscillation on a finite interval, the Filon-Clenshaw-Curtis rule needs fewer evaluations than QUADPACK's QAWO: 33 against 75, and 147 against 225.
The cost does not depend on $\omega$. The same 147 evaluations give $\int_{-2}^7 \cos(1000x)/(1 + x^2)\,dx$ to $5 \times 10^{-13}$, against QUADPACK's 125 for this frequency. The other weights
use general-purpose methods that resolve $f(x)w(x)$ rather than $f$ alone, so they need more evaluations than QUADPACK's special rules: about twice as many for the Fourier
integrals, and two to four times as many for the singular weights. The one exception is the asymmetric principal value, where the library needs fewer.

A complete program:

```rust,ignore
{{#include ../../../crates/integration/examples/quad_vec_weighted.rs}}
```

which prints

```text
J_0..5(2.5) = [-0.048383776468, 0.497094102464, 0.446059058440, 0.216600391039, 0.073781880054, 0.019501625135]
  105 evaluations, 3 subintervals, error estimate 7.4e-14
Fourier:   0.078205343895  error 2.2e-10
Algebraic: -4.000000000000  error 0.0e0
Cauchy:    1.098612288668  error 2.2e-16
```

Six Bessel functions cost 105 evaluations of the vector integrand. Integrated one at a time with the same tolerance, they need 546.

**Strengths**

- `quad_vec` shares the mesh and every evaluation of $\mathbf f$ among the components, and it handles infinite intervals and break points as `quad` does.
- Its results are identical to SciPy's, evaluation for evaluation, so code ported from SciPy behaves the same.
- `quad_weighted` covers QUADPACK's weighted cases with methods that are accurate to the requested tolerance and simple to verify.
- Oscillatory integrals on finite intervals cost the same at any frequency, and fewer evaluations than QUADPACK on the tests here.

**Weaknesses**

- Components with difficulties in different places all pay for each other's refinement. Integrate such components separately.
- The global error estimate is a norm over components. With the Euclidean norm and a relative tolerance, small components are
  computed to an absolute accuracy set by the largest ones.
- For Fourier integrals and singular weights, `quad_weighted` needs two to four times as many evaluations as QUADPACK's special rules.
- At high frequency an oscillatory integral is small, of order $1/\omega$, while the rounding error stays near $\varepsilon\max|f|\,(b - a)$. A relative tolerance then becomes unreachable, and an absolute one should be used.
- The Fourier extrapolation assumes the half-period integrals alternate and shrink regularly. It can fail, reporting `converged = false`, when $f$
  itself oscillates or does not decay.

## Exercises

1. Show that the map $x = a + (1 - t)/t$ sends $(0, 1]$ onto $[a, \infty)$, and that $dx = -dt/t^2$. For which decay rates of $f$ is
   the mapped integrand bounded as $t \to 0$?
2. Use `quad_vec` to compute $\int_0^1 x^k\,dx$ for $k = 0, \dots, 9$ in one call. How many evaluations does it take? Compare with ten
   separate calls of the scalar adaptive rule.
3. Apply the ε algorithm by hand to the partial sums of $\sum_{k \ge 1} (-1)^{k+1}/k$ (whose limit is $\ln 2$) for $S_1, S_2, S_3$. What is
   $\varepsilon^{(1)}_2$, and how close is it to $\ln 2$?
4. Show that $\bigl(f(c + t) - f(c - t)\bigr)/t \to 2f'(c)$ as $t \to 0$. Why is it important to pair the points symmetrically about $c$, rather
   than integrate $f(x)/(x - c)$ by an ordinary rule with a node avoiding $c$?
5. Compute $\int_0^\infty \sin x / x\,dx = \pi/2$ with `quad_weighted`. Why must $a > 0$ be used for the Fourier part, and how do you deal with $[0, a]$?
6. Show that $\mu_0(\kappa) = 2\sin\kappa/\kappa$ and $\mu_1(\kappa) = 2i(\sin\kappa - \kappa\cos\kappa)/\kappa^2$. What goes wrong if these are evaluated as written for small $\kappa$?

<details>
<summary>Hints and answers</summary>

1. As $t$ falls from 1 to $0^+$, $(1 - t)/t$ rises from 0 to $\infty$, and $dx/dt = -1/t^2$. The mapped integrand is $f(x)/t^2 \approx f(x)\,x^2$ for large $x$, so it is bounded when $f$ decays at least like $x^{-2}$. For slower decay it is unbounded at $t = 0$, but still integrable when $f = O(x^{-1-\delta})$.
2. The 21-point Gauss-Kronrod rule and its 10-point Gauss rule are both exact for degree 9, so the error estimate is at rounding level. `quad_vec` still bisects once before testing it, so it takes $3 \times 21 = 63$ evaluations, and ten separate calls of `quad_vec` would take 630. (QUADPACK tests before bisecting and would need 21 per integral.)
3. With $S_1 = 1$, $S_2 = 1/2$, $S_3 = 5/6$: $\varepsilon^{(1)}_2 = S_2 + 1/\bigl(1/(S_3 - S_2) - 1/(S_2 - S_1)\bigr) = 1/2 + 1/(3 + 2) = 0.7$, against $\ln 2 = 0.6931\ldots$. The partial sum $S_3$ is off by 0.14.
4. By Taylor's theorem $f(c \pm t) = f(c) \pm t f'(c) + O(t^2)$, so the difference is $2t f'(c) + O(t^3)$. An ordinary rule applied to $f(x)/(x - c)$ approximates an integral that does not exist. Its results depend on how close its nodes come to $c$ on each side, and do not converge as the rule is refined.
5. The library requires a finite $a$ and treats $f(x) = 1/x$ as a smooth factor, so $a = 0$ would put a singularity in $f$. Take $a = 1$ with `Weight::Sin(1.0)` and $f = 1/x$ for the tail, and integrate $\sin x / x$ over $[0, 1]$ (where it is smooth) with an ordinary rule.
6. $\int_{-1}^1 e^{i\kappa t}\,dt = (e^{i\kappa} - e^{-i\kappa})/(i\kappa) = 2\sin\kappa/\kappa$. For $\mu_1$, integrate $t e^{i\kappa t}$ by parts. For small $\kappa$ both formulas cancel catastrophically: $\sin\kappa - \kappa\cos\kappa \approx \kappa^3/3$ loses about $2\log_{10}(1/\kappa)$ digits. The Jacobi-Anger form avoids the subtraction.

</details>

## Further reading

- R. Piessens, E. de Doncker-Kapenga, C. W. Überhuber and D. K. Kahaner, *QUADPACK: A Subroutine Package for Automatic Integration*,
  Springer (1983).
- L. N. G. Filon, "On a quadrature formula for trigonometric integrals", *Proceedings of the Royal Society of Edinburgh* 49 (1928) 38–47.
- C. W. Clenshaw and A. R. Curtis, "A method for numerical integration on an automatic computer", *Numerische Mathematik* 2 (1960) 197–205.
- V. Domínguez, I. G. Graham and V. P. Smyshlyaev, "Stability and error estimates for Filon-Clenshaw-Curtis rules for highly oscillatory integrals", *IMA Journal of Numerical Analysis* 31 (2011) 1253–1280.
- P. Wynn, "On a device for computing the $e_m(S_n)$ transformation", *Mathematical Tables and Other Aids to Computation* 10 (1956) 91–96.
- D. Shanks, "Non-linear transformations of divergent and slowly convergent sequences", *Journal of Mathematics and Physics* 34 (1955) 1–42.
- R. Piessens and M. Branders, "Computation of oscillating integrals", *Journal of Computational and Applied Mathematics* 1 (1975) 153–164.
- C. Brezinski and M. Redivo Zaglia, *Extrapolation Methods: Theory and Practice*, North-Holland (1991). The ε algorithm and its relatives.
- P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python", *Nature Methods* 17 (2020) 261–272.
