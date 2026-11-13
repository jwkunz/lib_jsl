# Sampling from probability distributions

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Turn uniform random numbers into samples from a given distribution: continuous (normal, gamma, beta, …), discrete (Poisson, binomial, …) or multivariate (multivariate normal, Dirichlet) |
| **Devised by** | Inversion: John von Neumann and Stanisław Ulam (1940s); rejection: von Neumann (1951); George Box and Mervin Muller (1958), normal pairs; Joachim Ahrens and Ulrich Dieter (1974), gamma, Poisson and binomial recursions; George Marsaglia and Wai Wan Tsang (2000), gamma |
| **Accuracy** | Exact in distribution (up to the 53-bit resolution of the uniforms); every sampler passes a Kolmogorov-Smirnov test at 100 000 samples (table below) |
| **Cost** | 1–60 ns per sample: one or two uniforms for inversion and Box-Muller, a few for rejection |
| **API** | [`distributions`](../../api/lib_jsl_random/distributions/index.html): `uniform_distribution`, `exponential_distribution`, `weibull_distribution`, `gaussian_distribution_box_muller`, `log_normal_distribution`, `chi_square_distribution`, `students_t_distribution`, `gamma_distribution`, `beta_distribution`, `cauchy_distribution`, `bernoulli_distribution`, `geometric_distribution`, `poisson_distribution_knuth`, `binomial_distribution`, `dirichlet_distribution`, [`MultivariateNormal`](../../api/lib_jsl_random/distributions/multivariate_normal/struct.MultivariateNormal.html) |

</div>

## Overview

A handful of general techniques turn uniform samples into samples from any distribution. **Inversion**
applies the inverse CDF: if $U$ is uniform, $F^{-1}(U)$ has distribution $F$. It is exact and uses one uniform per
sample, but needs a cheap inverse CDF. **Transformations** exploit special structure, such as the polar form of two
independent normals. **Rejection** samples from a simpler envelope distribution and accepts each proposal with a
probability that corrects the shape. **Composition** builds a distribution from others: a beta from two gammas,
Student's $t$ from a normal and a chi-square, a Dirichlet vector from normalised gammas, a multivariate normal from a
linear map of independent normals.

## History

Inversion and rejection were developed by John von Neumann and Stanisław Ulam for the first Monte Carlo
computations on ENIAC in the late 1940s; von Neumann's 1951 paper "Various techniques used in connection with random
digits" introduced rejection sampling. George Box and Mervin Muller (1958) found the transformation that gives exact
normal pairs from two uniforms. Joachim Ahrens and Ulrich Dieter (1970s) designed fast exact samplers for the gamma,
Poisson and binomial distributions, collected with many others in Luc Devroye's *Non-Uniform Random Variate Generation*
(1986). George Marsaglia and Wai Wan Tsang's gamma sampler (2000), a rejection from a transformed normal that accepts over
95% of proposals, is the modern default.

## The idea

### Inversion

A generator supplies uniform numbers $U$ on $(0, 1)$. Every other distribution must be
manufactured from them, and there are three general techniques: inversion, transformation and
rejection.

**Inversion** uses the cumulative distribution function $F(x) = P(X \le x)$. If $F$ is continuous
and increasing, set

$$
X = F^{-1}(U).
$$

To see that $X$ has the right distribution, compute its CDF. Since $F$ is increasing,
$X \le x$ exactly when $U \le F(x)$, and for a uniform variable the probability of being below
a number $p$ in $(0, 1)$ is $p$ itself:

$$
P(X \le x) = P\big(U \le F(x)\big) = F(x).
$$

**Example: the exponential distribution.** With rate $\lambda$ its CDF is $F(x) = 1 - e^{-\lambda x}$.
Solving $u = 1 - e^{-\lambda x}$ for $x$ gives

$$
X = -\frac{\ln(1 - U)}{\lambda} .
$$

Since $1 - U$ is also uniform, one could write $\ln U$ instead. The library keeps $1 - U$, because
$U$ can be exactly 0, and $\ln 0$ is infinite, whereas $1 - U$ is never 0.

The Weibull, the Cauchy ($X = \tan(\pi(U - \tfrac12))$) and, after rounding down, the geometric
distribution are sampled in the same way. Inversion needs a formula for $F^{-1}$, which many
distributions (the normal among them) do not have. For discrete distributions, inversion becomes
a search: find the smallest $k$ with $F(k) \ge U$.

### The normal distribution: Box-Muller

The normal CDF cannot be inverted in closed form, but a pair of normals can be produced by a
change of variables. If $U_1$ and $U_2$ are independent uniforms, then

$$
Z_1 = \sqrt{-2\ln U_1}\cos(2\pi U_2), \qquad Z_2 = \sqrt{-2\ln U_1}\sin(2\pi U_2)
$$

are independent standard normals.

The reason is easiest to see backwards. Take two independent standard normals and regard them
as the coordinates of a point in the plane. Their joint density is proportional to
$e^{-(z_1^2 + z_2^2)/2} = e^{-r^2/2}$, which depends only on the distance $r$ from the origin. So
the direction of the point is uniform over the circle, and independent of its distance. As for
the distance, changing to polar coordinates shows that $R^2$ has an exponential distribution
with mean 2. Both ingredients are easy to sample: the angle is $2\pi U_2$, and by inversion the
squared radius is $-2\ln U_1$. Converting the polar coordinates back to Cartesian ones gives the
formulas above.

The library returns both values, and scales them to $N(\mu, \sigma^2)$ as $\mu + \sigma Z$. Several
other distributions are built on the normal by their definitions: the log-normal is $e^Z$; the
chi-square with $k$ degrees of freedom is the sum of $k$ squared normals; Student's $t$ is a
normal divided by $\sqrt{\chi^2_\nu/\nu}$.

### Rejection: the gamma distribution

When neither inversion nor a transformation is available, **rejection sampling** almost always
is. Picture the region under the graph of the density $f$. If points are scattered uniformly
over that region, their $x$-coordinates have density $f$, because the chance of landing in a thin
vertical strip is proportional to its height. The difficulty is scattering points uniformly
under $f$. Rejection does it indirectly:

1. Choose an **envelope**: a density $g$ that is easy to sample, and a constant $c$ such that
   $f(x) \le c\,g(x)$ everywhere, so that the graph of $cg$ lies above that of $f$.
2. Draw $X$ from $g$ and an independent uniform $U$. The point $(X, U\,c\,g(X))$ is uniform under
   the graph of $cg$.
3. **Accept** $X$ if the point lies under the graph of $f$, that is if $U \le f(X)/(c\,g(X))$.
   Otherwise reject it and return to step 2.

The accepted points are uniform under $f$, so the accepted values have density $f$. The area
under $cg$ is $c$ and the area under $f$ is 1, so the probability of acceptance is $1/c$ and on
average $c$ proposals are needed per sample. A good envelope hugs the density closely.

**Marsaglia and Tsang's gamma sampler** is a refined example, for shape $\alpha \ge 1$. It sets
$X = d(1 + cZ)^3$ with $Z$ normal, $d = \alpha - \tfrac13$ and $c = 1/\sqrt{9d}$. The cube of a
normal variable, suitably shifted and scaled, already has a distribution very close to the
gamma, so the rejection step only has to correct a small discrepancy: the value is accepted when
$\ln U < \tfrac12Z^2 + d - dV + d\ln V$ with $V = (1 + cZ)^3$, which happens more than 95% of the
time.

Shapes below 1 use the identity $\operatorname{Gamma}(\alpha) = \operatorname{Gamma}(\alpha + 1)\cdot U^{1/\alpha}$. Other
distributions follow from gammas: the beta distribution is $X/(X + Y)$ for independent gammas of
shapes $a$ and $b$, and the Dirichlet normalises a vector of gammas to sum to 1.

### Discrete distributions: Poisson and binomial

A Poisson variable with rate $\lambda$ counts the events of a **Poisson process**, in which events
occur independently at an average rate $\lambda$, during one unit of time. The gaps between
successive events of such a process are independent exponentials with rate $\lambda$. So one way
to sample is to add up exponential gaps until their sum exceeds 1, and count how many fitted.

**Knuth's sampler** is this idea with the logarithms removed. Each gap is $-\ln U_i/\lambda$, so
the sum of the gaps exceeds 1 exactly when the *product* of the uniforms falls below
$e^{-\lambda}$. The sampler multiplies uniforms together until that happens, and returns the number
of factors, less one. It needs about $\lambda + 1$ uniforms, so it is suitable only for small
rates.

For large $\lambda$ the library uses the exact recursion of Ahrens and Dieter, which avoids
simulating every event. Let $m = \lfloor7\lambda/8\rfloor$, and draw the time $G$ of the $m$-th event
directly, as a $\operatorname{Gamma}(m)$ variable (the sum of $m$ exponential gaps).

- If $G$ is before the end of the interval, at least $m$ events have occurred, and the count is
  $m$ plus a Poisson count for the remaining time, with the smaller rate $\lambda - G$.
- If $G$ is after the end, fewer than $m$ events occurred. The $m - 1$ earlier events are
  uniformly distributed over $(0, G)$, so the number of them that fall inside the interval is
  binomial.

Either way the problem has been reduced to a much smaller one. The binomial sampler uses
inversion for small means, and otherwise splits recursively in a similar way, on a
beta-distributed order statistic, in $O(\log n)$ beta samples.

The original Poisson sampler multiplied one uniform by itself, with its loop condition inverted, and returned $-1$ for every $\lambda$.

### Multivariate normal

If $\Sigma = LL^{\mathsf T}$ is the [Cholesky factorisation](../linear_algebra/cholesky.md) of a covariance matrix and $\mathbf z$ is a vector of independent
standard normals, then $\boldsymbol\mu + L\mathbf z$ is $N(\boldsymbol\mu, \Sigma)$. `MultivariateNormal` factorises once and samples in $O(n^2)$.

### Testing the samplers

For each distribution, 100 000 samples are compared with the exact moments and CDF (from the
[special functions](../special_functions/statistical.md)). The **Kolmogorov-Smirnov statistic** $D$, the largest difference between
the empirical and the exact CDF, should be below $1.36/\sqrt n$ about 95% of the time:

<p class="caption" id="tab-9-4"><strong>Table 9.4.</strong> Sample moments of each sampler against the exact values.</p>

{{#include ../generated/random/audit.md}}

All pass; the Weibull's $D$ sits at the critical value, which is to be expected for one of seventeen tests at the 5% level (with other
seeds its mean is within $1.3$ standard errors). The histograms show three of the samples against their densities:

![Histograms against densities](../generated/random/histograms.svg)

<p class="caption" id="fig-9-1"><strong>Figure 9.1.</strong> Histograms against densities.</p>

## Algorithm

Each sampler is a few lines; these are the main ones.

**Exponential** with rate $\lambda$ (inversion): draw $U$; return $-\ln(1 - U)/\lambda$.

**Normal** (Box-Muller): draw $U_1$ and $U_2$; compute $r = \sqrt{-2\ln U_1}$ (taking care that the argument of the logarithm is never zero) and
$\theta = 2\pi U_2$; return the pair $r\cos\theta$, $r\sin\theta$. For $N(\mu, \sigma^2)$ return
$\mu + \sigma Z$.

**Gamma** with shape $\alpha \ge 1$ (Marsaglia and Tsang). Set $d = \alpha - \tfrac13$ and
$c = 1/\sqrt{9d}$. Repeat:

1. draw a standard normal $Z$ and set $V = (1 + cZ)^3$; if $V \le 0$, start again;
2. draw a uniform $U$; if $\ln U < \tfrac12Z^2 + d - dV + d\ln V$, return $dV$.

For $\alpha < 1$, sample with shape $\alpha + 1$ and multiply by $U^{1/\alpha}$. Multiply by the
scale parameter at the end.

**Beta**$(a, b)$: draw $X \sim \operatorname{Gamma}(a)$ and $Y \sim \operatorname{Gamma}(b)$; return $X/(X + Y)$.

**Poisson** with a small rate $\lambda$ (Knuth): set $L = e^{-\lambda}$, $k = 0$ and $p = 1$;
repeat "multiply $p$ by a new uniform and add 1 to $k$" until $p \le L$; return $k - 1$. For
a large rate, use the recursion on the time of the $m$-th event described above, which ends
in a call with a small rate.

**Multivariate normal** $N(\boldsymbol\mu, \Sigma)$: factor $\Sigma = LL^{\mathsf T}$ once; for each sample draw
a vector $\mathbf z$ of independent standard normals and return $\boldsymbol\mu + L\mathbf z$.

**A discrete distribution by inversion**: draw $U$; add up the probabilities
$P(0), P(1), \ldots$ until the running total reaches $U$; return the index reached.

## Implementation

Box-Muller:

```rust,ignore
{{#include ../../../crates/random/src/distributions/gaussian.rs:box_muller}}
```

Marsaglia-Tsang:

```rust,ignore
{{#include ../../../crates/random/src/distributions/gamma.rs:marsaglia_tsang}}
```

Poisson:

```rust,ignore
{{#include ../../../crates/random/src/distributions/poisson.rs:poisson}}
```

- Parameter conventions: the exponential and Poisson take a **rate**, the gamma a shape and a **scale**, the geometric counts **failures**
  before the first success.
- The chi-square and Student's $t$ samplers draw normals with a given mean and standard deviation; use mean 0 and 1 for the standard
  distributions.
- Invalid parameters give errors (or `NaN` for the Poisson sampler, which returns a bare `f64`).

## In practice

```rust,ignore
{{#include ../../../crates/random/examples/random_distributions.rs}}
```

**Strengths**

- Exact samplers for the common distributions, tested against their CDFs.
- Generic over the uniform generator.

**Weaknesses**

- Box-Muller needs a logarithm, a square root and a sine and cosine per pair; the ziggurat method is several times faster.
- No sampler for arbitrary user densities (see [MCMC](mcmc.md) for unnormalised densities).

## Exercises

1. Derive the inversion formula for the Weibull distribution from its CDF $1 - e^{-(x/\lambda)^k}$.
2. Prove that the Box-Muller transformation gives independent standard normals, using the Jacobian of the polar change of variables.
3. Sample the half-normal distribution by rejection from an exponential envelope, and find the best rate and the acceptance rate.
4. Show that if $X \sim \operatorname{Gamma}(a)$ and $Y \sim \operatorname{Gamma}(b)$ are independent then $X/(X + Y) \sim \operatorname{Beta}(a, b)$.
5. Check the `MultivariateNormal` sampler by estimating the covariance of $10^5$ samples of a $3 \times 3$ case.

<details>
<summary>Hints and answers</summary>

1. Solve $u = 1 - e^{-(x/\lambda)^k}$ for $x$: $x = \lambda(-\ln(1 - u))^{1/k}$, and since $1 - u$ is also uniform, $x = \lambda(-\ln u)^{1/k}$ works.
2. For independent uniforms, $R = \sqrt{-2\ln U_1}$ and $\Theta = 2\pi U_2$ have joint density $re^{-r^2/2}/(2\pi)$ on $r > 0$, $0 \le \theta < 2\pi$. The map $(r, \theta) \mapsto (x, y) = (r\cos\theta, r\sin\theta)$ has Jacobian $r$, so $(x, y)$ has density $e^{-(x^2 + y^2)/2}/(2\pi)$, a product of two standard normal densities.
3. The target is $f(x) = \sqrt{2/\pi}\,e^{-x^2/2}$ and the envelope $g(x) = \lambda e^{-\lambda x}$. The ratio $f/g$ is largest at $x = \lambda$, where it equals $M(\lambda) = \sqrt{2/\pi}\,e^{\lambda^2/2}/\lambda$; this is smallest for $\lambda = 1$, giving $M = \sqrt{2e/\pi} \approx 1.316$ and an acceptance rate $1/M \approx 0.76$.
4. The joint density is $\propto x^{a-1}y^{b-1}e^{-(x+y)}$. Put $s = x + y$ and $w = x/s$ (Jacobian $s$): it factorises as $s^{a+b-1}e^{-s}\cdot w^{a-1}(1 - w)^{b-1}$, so $w \sim \operatorname{Beta}(a, b)$, independent of $s$.
5. The sample covariance should match the target matrix to within about $1/\sqrt{10^5} \approx 0.003$ relative to the entries' scale.

</details>

## Further reading

- L. Devroye, *Non-Uniform Random Variate Generation*, Springer, 1986 (free online from the author).
- G. Marsaglia and W. W. Tsang, "A simple method for generating gamma variables", *ACM Transactions on Mathematical Software* 26
  (2000) 363–372.
- G. E. P. Box and M. E. Muller, "A note on the generation of random normal deviates", *Annals of Mathematical Statistics* 29 (1958)
  610–611.
- J. H. Ahrens and U. Dieter, "Computer methods for sampling from gamma, beta, Poisson and binomial distributions", *Computing* 12
  (1974) 223–246.
