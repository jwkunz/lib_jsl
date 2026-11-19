# Multivariate distributions and further tests

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Model the amplitude of a noisy signal (Rice) and random vectors and matrices (multivariate normal and t, Dirichlet, Wishart, multinomial); test goodness of fit and differences between groups without normality (Cramér-von Mises, Brunner-Munzel); compare every pair of means at once (Tukey's HSD); and measure effects in contingency tables (odds ratio, relative risk, Cramér's V) |
| **Devised by** | Stephen Rice (1944), the Rice distribution; John Wishart (1928); Harald Cramér (1928) and Richard von Mises (1931), Theodore Anderson and Donald Darling (1952), Theodore Anderson (1962); Edgar Brunner and Ullrich Munzel (2000); John Tukey (1953), Clyde Kramer (1956), Paul Games and John Howell (1976); Jerome Cornfield (1956) and Barnet Woolf (1955), odds ratios; Harald Cramér (1946), association |
| **Accuracy** | Agrees with `scipy.stats` to $10^{-13}$ or better throughout; the Rice tail probabilities are more accurate than SciPy's |
| **Cost** | Densities: $O(d^3)$ once for a factorisation, then $O(d^2)$ per point; Cramér-von Mises: $O(n\log n)$, exact two-sample p-values by dynamic programming over lattice paths; studentized range: a double integral of a few hundred thousand evaluations of $\Phi$ |
| **Read first** | [Probability distributions](distributions.md), [More hypothesis tests](more-tests.md) |
| **API** | [`Rice`](../../api/lib_jsl_statistics/distributions/struct.Rice.html); [`multivariate`](../../api/lib_jsl_statistics/multivariate/index.html) (`MultivariateNormal`, `MultivariateT`, `Dirichlet`, `Wishart`, `InvWishart`, `Multinomial`); [`goodness_of_fit`](../../api/lib_jsl_statistics/goodness_of_fit/index.html) (`cramervonmises`, `cramervonmises_2samp`); [`brunnermunzel`](../../api/lib_jsl_statistics/comparisons/fn.brunnermunzel.html), [`monte_carlo_test`](../../api/lib_jsl_statistics/comparisons/fn.monte_carlo_test.html); [`posthoc`](../../api/lib_jsl_statistics/posthoc/index.html) (`StudentizedRange`, `tukey_hsd`); [`contingency`](../../api/lib_jsl_statistics/contingency/index.html) (`odds_ratio`, `relative_risk`, `association`) |

</div>

## Overview

This entry fills the gaps between the earlier statistics entries and `scipy.stats`:

- **The Rice distribution** of the magnitude $|\mathbf s + \mathbf n|$ of a constant signal plus Gaussian noise. It is the envelope distribution in radio, radar and MRI. With no signal it reduces to the Rayleigh distribution.
- **Multivariate distributions.** The multivariate normal and t model random vectors. The Dirichlet distribution models random probability vectors, for example the posterior of multinomial
  proportions. The Wishart and inverse Wishart distributions model random covariance matrices, and the multinomial models counts in several categories.
- **Goodness of fit.** The Cramér-von Mises statistic integrates the squared distance between distribution functions, $\omega^2 = n\int(F_n - F)^2\,dF$. It is often more powerful than
  Kolmogorov-Smirnov's maximum distance.
- **Brunner-Munzel**, a rank test for whether one group tends to have larger values than another, valid even when the groups have different spreads or shapes. The Mann-Whitney test assumes that they do not.
- **Tukey's honestly significant difference** (HSD), which compares all $\binom k2$ pairs of group means with a family-wise error rate held at $\alpha$.
- **Effect sizes for contingency tables**: the odds ratio (conditional and sample), the relative risk, and Cramér's V and its relatives.
- **Monte Carlo tests**, which take the null distribution of any statistic from simulations of the null hypothesis.

## History

Stephen Rice (1944) derived the distribution of the envelope of a sinusoid in Gaussian noise in his study of random noise at Bell Laboratories. John Wishart (1928) found the distribution of
the sample covariance matrix of normal data, and Maurice Bartlett (1933) gave the triangular decomposition used to simulate it.

Harald Cramér (1928) and Richard von Mises (1931) proposed the integrated squared distance between distribution functions. Theodore Anderson and Donald Darling (1952) found its limiting
distribution as a series of Bessel functions, Anderson (1962) extended the test to two samples, and Sándor Csörgő and Julian Faraway (1996) gave the finite-sample correction that SciPy uses.

John Tukey's (1953) "honestly significant difference" used the distribution of the range of normal means, studentized by an independent estimate of spread. Clyde Kramer (1956) extended it to unequal group sizes, and Paul Games
and John Howell (1976) to unequal variances. Edgar Brunner and Ullrich Munzel (2000) solved the nonparametric Behrens-Fisher problem: testing $P(X < Y) = \tfrac12$ without
assuming equal distributions under the null hypothesis.

For contingency tables, Barnet Woolf (1955) gave the log-odds confidence interval, and Jerome Cornfield (1956) the exact conditional one. Harald Cramér (1946) normalised the χ² statistic
to a measure of association between 0 and 1.

## The idea

### The Rice distribution

If $X = |\boldsymbol\nu + \mathbf n|$ with $\mathbf n \sim N(\mathbf 0, \sigma^2I_2)$ and $|\boldsymbol\nu| = \nu$, then

$$
f(x) = \frac{x}{\sigma^2}\exp\Bigl(-\frac{x^2 + \nu^2}{2\sigma^2}\Bigr)I_0\Bigl(\frac{x\nu}{\sigma^2}\Bigr), \qquad x \ge 0 .
$$

The product $e^{-(x^2 + \nu^2)/2\sigma^2}I_0(x\nu/\sigma^2)$ overflows for strong signals, although the density is a modest number. Writing it with the scaled Bessel function
$I_0(z)e^{-z}$ gives $e^{-(x-\nu)^2/2\sigma^2}I_0(z)e^{-z}$, which never overflows.

$X^2/\sigma^2$ is non-central χ² with 2 degrees of freedom and non-centrality $\lambda = (\nu/\sigma)^2$. That makes the distribution function a Poisson mixture of central χ² ones,

$$
P(X \le x) = \sum_{j\ge0} e^{-\lambda/2}\frac{(\lambda/2)^j}{j!}\,P\Bigl(1 + j, \frac{x^2}{2\sigma^2}\Bigr),
$$

where $P$ is the regularised incomplete gamma function. Using $Q = 1 - P$ instead gives the survival function. Both are sums of positive terms, so each is accurate to rounding in its own tail.
The library sums outwards from the Poisson mode until the terms no longer matter.

The moments follow from Laguerre functions, which can be written with Bessel functions of argument $t/2$, $t = \nu^2/2\sigma^2$. For example,
$E[X] = \sigma\sqrt{\pi/2}\,\bigl((1 + t)I_0(t/2) + tI_1(t/2)\bigr)e^{-t/2}$, and $E[X^3]$ follows by a contiguous relation of the confluent hypergeometric function.

### Multivariate distributions

All the densities need a log-determinant and a quadratic form. Both come from the Cholesky factor $\Sigma = LL^{\mathsf T}$: $\log|\Sigma| = 2\sum\log L_{ii}$, and
$(\mathbf x - \boldsymbol\mu)^{\mathsf T}\Sigma^{-1}(\mathbf x - \boldsymbol\mu) = \|L^{-1}(\mathbf x - \boldsymbol\mu)\|^2$. Sampling uses the same factor, $\mathbf x = \boldsymbol\mu + L\mathbf z$. The multivariate t is a normal divided by an
independent $\sqrt{\chi^2_\nu/\nu}$. A Dirichlet vector is a normalised vector of independent gamma deviates. A Wishart matrix is $LAA^{\mathsf T}L^{\mathsf T}$ by Bartlett's decomposition,
where $A$ is lower triangular with $A_{ii} = \sqrt{\chi^2_{\nu - i}}$ and standard normal entries below the diagonal.

### The Cramér-von Mises tests

For a sample $x_{(1)} \le \dots \le x_{(n)}$ and a hypothesised distribution function $F$:

$$
\omega^2 = \frac1{12n} + \sum_{i=1}^n\Bigl(\frac{2i - 1}{2n} - F(x_{(i)})\Bigr)^2 .
$$

Its limiting distribution is Anderson and Darling's Bessel series

$$
P(\omega^2 \le x) = \sum_{k\ge0}\frac{\Gamma(k + \tfrac12)\sqrt{4k + 1}}{\Gamma(k + 1)\,\pi^{3/2}\sqrt x}\,e^{-q_k}K_{1/4}(q_k), \qquad q_k = \frac{(4k + 1)^2}{16x},
$$

and Csörgő and Faraway's correction, a second Bessel series divided by $n$, makes it accurate for small samples. The two-sample statistic uses the ranks of each sample in the pooled one.
For small samples its exact distribution is a sum over all $\binom{m+n}{m}$ ways of interleaving the samples. Xiao, Gordon and Yakovlev count those efficiently: a dynamic programme over
lattice paths, where a step adds $(av - bw)^2$ to the statistic, keeps for each path end the number of paths reaching each value. Because the counts are exact integers, the p-value is exact.

### The Brunner-Munzel test

Mann-Whitney tests $H_0: F_X = F_Y$. Brunner and Munzel test only $p = P(X < Y) + \tfrac12P(X = Y) = \tfrac12$. With pooled ranks $R$ and within-sample ranks $R^{(x)}$, the
"placements" $R_i - R_i^{(x)}$ estimate how many of the other sample lie below each observation. Their variances $S_x^2$, $S_y^2$ give

$$
W = \frac{n_xn_y(\bar R_y - \bar R_x)}{(n_x + n_y)\sqrt{n_xS_x^2 + n_yS_y^2}} ,
$$

which is referred to a $t$ distribution with Satterthwaite's degrees of freedom.

### Tukey's honestly significant difference

If $k$ groups have equal means, the largest difference between sample means, divided by its standard error, follows the **studentized range** distribution:

$$
P(Q \le q) = \int_0^\infty f_S(s)\,W(qs)\,ds, \qquad W(w) = k\int_{-\infty}^\infty\varphi(z)\bigl(\Phi(z) - \Phi(z - w)\bigr)^{k-1}dz .
$$

Here $W$ is the distribution of the range of $k$ standard normals, and $f_S$ the density of $S = \sqrt{\chi^2_\nu/\nu}$. Comparing every pair with the $1 - \alpha$ point of this distribution
keeps the probability of any false difference at $\alpha$, because the largest difference is below it with probability $1 - \alpha$. The library evaluates both integrals by composite 16-point
Gauss-Legendre quadrature. The inner range is $|z| \le 8.5$. The outer one covers the part of $s$ that holds all but $10^{-17}$ of its mass.

### Effect sizes for contingency tables

For a 2 × 2 table $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$, the **sample odds ratio** is $ad/(bc)$, with Woolf's interval
$\exp\bigl(\log\mathrm{OR} \pm z\sqrt{1/a + 1/b + 1/c + 1/d}\bigr)$. The **conditional** odds ratio conditions on all margins. Then $a$ alone is random, with Fisher's noncentral hypergeometric
distribution $P(a) \propto \binom{n_1}{a}\binom{n_2}{m_1 - a}\omega^a$. The estimate solves $E_\omega[a] = a_{\text{obs}}$, and the exact confidence limits solve $P_\omega(a \ge a_{\text{obs}}) = \alpha/2$ and
$P_\omega(a \le a_{\text{obs}}) = \alpha/2$. This is the interval that matches Fisher's exact test. The relative risk $(a/n_1)/(c/n_2)$ has Katz's log-scale interval. **Cramér's V**,
$\sqrt{\chi^2/(N\min(r - 1, c - 1))}$, puts the strength of association of an $r \times c$ table on a scale from 0 (independence) to 1.

## Algorithm

**Exact two-sample Cramér-von Mises p-value** (sample sizes $m$, $n$; $a = \operatorname{lcm}(m, n)/m$, $b = \operatorname{lcm}(m, n)/n$):

1. Keep, for each $v = 0, \dots, m$, a table from accumulated values to path counts. Start with $\{0: 1\}$ at $v = 0$.
2. For $w = 0, \dots, n$, sweep $v = 0, \dots, m$: merge the table for $v$ into a running table, add $(av - bw)^2$ to every key, and store it as the new table for $v$.
3. The p-value is the count of paths whose value is at least the observed (rescaled) statistic, divided by $\binom{m+n}m$.

**Studentized range quantile:** bracket the quantile by doubling, then use regula falsi with the Illinois modification to a relative tolerance of $10^{-13}$.

## Implementation

```rust,ignore
{{#include ../../../crates/statistics/src/distributions.rs:rice}}
```

```rust,ignore
{{#include ../../../crates/statistics/src/multivariate.rs:wishart}}
```

```rust,ignore
{{#include ../../../crates/statistics/src/goodness_of_fit.rs:cramervonmises_2samp}}
```

```rust,ignore
{{#include ../../../crates/statistics/src/comparisons.rs:brunnermunzel}}
```

```rust,ignore
{{#include ../../../crates/statistics/src/posthoc.rs:tukey_hsd}}
```

```rust,ignore
{{#include ../../../crates/statistics/src/contingency.rs:odds_ratio}}
```

- The Cramér-von Mises series are summed exactly as SciPy sums them, stopping after the first term below $10^{-7}$, so the p-values agree to rounding. That threshold limits their absolute accuracy to about $10^{-7}$.
- `tukey_hsd` returns the full $k \times k$ matrices, as SciPy does. Its `confidence_interval` evaluates the studentized range quantile once for each distinct number of degrees of freedom.
- `monte_carlo_test` takes the null distribution as a closure `rvs(sample index, size, rng)` and the library's random generators, so its results are reproducible for a given seed. They are not SciPy's for the same seed.
- Dunnett's test, which compares treatments with a control, is not implemented. SciPy computes its p-values with a randomised quasi-Monte Carlo integration of the multivariate t distribution.

## In practice

The references are from SciPy 1.17.1, written by `book/generator/data/make_stats27_refs.py`:

<p class="caption" id="tab-10-12"><strong>Table 10.12.</strong> The Rice and multivariate distributions against `scipy.stats`.</p>

{{#include ../generated/statistics/stats27_distributions.md}}

The densities, moments and entropies agree with SciPy to rounding. In the far upper tail of the Rice distribution, near $10^{-9}$, SciPy's survival function is off by up to $2 \times 10^{-7}$
relative, while the library's agrees with direct integration of the density to $10^{-14}$. At $\nu/\sigma = 40$, SciPy's moment formula overflows to NaN. The library's
moments, from scaled Bessel functions, are finite, and the mean approaches the normal limit $\nu + \sigma^2/2\nu$. The fitted parameters differ from SciPy's in the fifth digit. The library's
estimate has the higher likelihood, because SciPy's optimiser stops earlier.

<p class="caption" id="tab-10-13"><strong>Table 10.13.</strong> Tests and effect sizes against `scipy.stats`.</p>

{{#include ../generated/statistics/stats27_tests.md}}

**Strengths**

- The Rice distribution is accurate in both tails and for strong signals, where SciPy loses digits or returns NaN.
- The tests reproduce SciPy's statistics and p-values to rounding, including the exact two-sample Cramér-von Mises p-values and the exact conditional odds ratio intervals.
- Tukey's HSD comes with simultaneous confidence intervals, in the Tukey-Kramer (unequal sizes) and Games-Howell (unequal variances) forms.

**Weaknesses**

- The studentized range costs a double integral per evaluation, about 30 ms, and its quantile about ten evaluations (0.3 s).
- The multivariate normal needs a positive definite covariance (SciPy's `allow_singular` is not supported), and there is no multivariate normal distribution function.
- No Dunnett test, and no `stats.sampling` generators.

## Exercises

1. Show that the Rice distribution with $\nu = 0$ is the Rayleigh distribution, and that $X^2/\sigma^2$ is non-central χ² with 2 degrees of freedom.
2. Show that $\omega^2$ is $n\int(F_n - F)^2\,dF$ for the empirical distribution function $F_n$.
3. With three groups of four observations and pooled standard deviation 1, how large must a difference of means be to be significant at the 5 % level by Tukey's HSD? (Use $q_{0.95}(3, 9) = 3.95$.)
4. Find the conditional and sample odds ratios of $\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$, and explain why the exact interval is so wide.
5. Show that a vector of independent Gamma($\alpha_i$) variables, divided by its sum, is Dirichlet($\boldsymbol\alpha$).

<details>
<summary>Hints and answers</summary>

1. $I_0(0) = 1$ leaves $f(x) = (x/\sigma^2)e^{-x^2/2\sigma^2}$. $X^2/\sigma^2$ is the sum of the squares of two independent normals with means $\nu_1/\sigma$, $\nu_2/\sigma$ and unit variance: non-central χ² with non-centrality $\nu^2/\sigma^2$.
2. Between order statistics, $F_n$ is constant at $i/n$. Integrate $(i/n - u)^2$ over $u \in [F(x_{(i)}), F(x_{(i+1)})]$ and sum, which gives the same expression after rearranging.
3. $q\cdot\mathrm{se} = 3.95\sqrt{1/4} = 1.98$. Here $\mathrm{se} = \sqrt{\mathrm{MSE}(1/n_i + 1/n_j)/2} = \sqrt{1/4}$.
4. Both are 1. With so few counts, the conditional distribution of $a$ has support $\{0, 1, 2\}$, and almost any $\omega$ is compatible with it: the 95 % interval is $(0.0064, 156)$.
5. Change variables to $s = \sum g_i$ and $x_i = g_i/s$. The Jacobian is $s^{K-1}$, and the joint density factorises into a Gamma($\sum\alpha_i$) density in $s$ and the Dirichlet density in $\mathbf x$.

</details>

## Further reading

- S. O. Rice, "Mathematical analysis of random noise", *Bell System Technical Journal* 23 (1944) 282–332 and 24 (1945) 46–156.
- J. Wishart, "The generalised product moment distribution in samples from a normal multivariate population", *Biometrika* 20A (1928) 32–52.
- T. W. Anderson and D. A. Darling, "Asymptotic theory of certain 'goodness of fit' criteria based on stochastic processes", *Annals of Mathematical Statistics* 23 (1952) 193–212.
- T. W. Anderson, "On the distribution of the two-sample Cramér-von Mises criterion", *Annals of Mathematical Statistics* 33 (1962) 1148–1159.
- S. Csörgő and J. J. Faraway, "The exact and asymptotic distributions of Cramér-von Mises statistics", *Journal of the Royal Statistical Society B* 58 (1996) 221–234.
- Y. Xiao, A. Gordon and A. Yakovlev, "A C++ program for the Cramér-von Mises two-sample test", *Journal of Statistical Software* 17 (2006) 1–15.
- E. Brunner and U. Munzel, "The nonparametric Behrens-Fisher problem: asymptotic theory and a small-sample approximation", *Biometrical Journal* 42 (2000) 17–25.
- C. Y. Kramer, "Extension of multiple range tests to group means with unequal numbers of replications", *Biometrics* 12 (1956) 307–310.
- P. A. Games and J. F. Howell, "Pairwise multiple comparison procedures with unequal n's and/or variances", *Journal of Educational Statistics* 1 (1976) 113–125.
- B. Woolf, "On estimating the relation between blood group and disease", *Annals of Human Genetics* 19 (1955) 251–253.
- J. Cornfield, "A statistical problem arising from retrospective studies", *Proceedings of the Third Berkeley Symposium* 4 (1956) 135–148.
- G. Marsaglia and W. W. Tsang, "A simple method for generating gamma variables", *ACM Transactions on Mathematical Software* 26 (2000) 363–372.
