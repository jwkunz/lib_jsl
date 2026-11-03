# The bootstrap

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Estimate the standard error, bias and confidence interval of any statistic, without a formula for its sampling distribution |
| **Devised by** | Maurice Quenouille (1949) and John Tukey (1958), the jackknife; Bradley Efron (1979), the bootstrap; Efron (1987), the bias-corrected and accelerated (BCa) interval |
| **Accuracy** | Percentile intervals are first-order accurate (coverage error $O(n^{-1/2})$); BCa intervals second-order ($O(n^{-1})$). Both under-cover for small, skewed samples (table below) |
| **Cost** | $B$ evaluations of the statistic on resamples of size $n$ (typically $B = 1000$ to $10\,000$), plus $n$ jackknife evaluations for BCa |
| **API** | [`bootstrap`](../../api/lib_jsl_statistics/bootstrap/fn.bootstrap.html), [`BootstrapResult`](../../api/lib_jsl_statistics/bootstrap/struct.BootstrapResult.html) (`percentile_interval`, `bca_interval`) |

</div>

## Overview

The uncertainty of a statistic $\hat\theta = s(x_1, \ldots, x_n)$ is described by its **sampling distribution**: how it would vary over repeated samples from
the population. For a mean, theory gives the answer $\sigma/\sqrt n$; for a median, a correlation, a ratio or the output of a complicated
procedure, there may be no usable formula. The **bootstrap** replaces the unknown population by the sample itself. Draw $n$
values *with replacement* from the data, recompute the statistic, repeat $B$ times, and treat the spread of the **replicates**
$\hat\theta^*_1, \ldots, \hat\theta^*_B$ as an estimate of the sampling distribution.

## History

Maurice Quenouille (1949) introduced the **jackknife** to reduce bias, recomputing a statistic with each observation left out
in turn, and John Tukey (1958) used it to estimate variances and named it after a rough-and-ready tool. Bradley Efron's
"Bootstrap methods: another look at the jackknife" (1979) generalised the idea to resampling with replacement, and the
cheapness of computing made it practical. Efron (1987) introduced the bias-corrected and accelerated interval, and the
second-order theory of bootstrap intervals was developed by Peter Hall (1988, 1992), Thomas DiCiccio and others. Efron and
Robert Tibshirani's *An Introduction to the Bootstrap* (1993) brought the method into everyday practice.

## The idea

### Resampling and the plug-in principle

The empirical distribution $\hat F$ puts mass $1/n$ on each observation. The bootstrap estimates any property of the sampling distribution
under $F$ by the same property under $\hat F$, computed by simulation. The **standard error** is the standard deviation of the replicates,
and the **bias** is $\bar\theta^* - \hat\theta$. A resample contains on average $1 - (1 - 1/n)^n \approx 63\%$ of the distinct observations, some repeated.

### Percentile intervals

The simplest interval takes the $\alpha/2$ and $1 - \alpha/2$ quantiles of the replicates. It is correct when some monotone transformation of $\hat\theta$
is normal and unbiased, but it inherits the bias and skewness of the bootstrap distribution, and its coverage error is $O(n^{-1/2})$.

### The BCa interval

Efron's **bias-corrected and accelerated** interval uses different quantiles of the same replicates. The bias correction
$z_0 = \Phi^{-1}\big(\#\{\hat\theta^*_b < \hat\theta\}/B\big)$ measures the median bias of the replicates; the **acceleration** $a$, how fast the standard error changes with
$\theta$, is estimated from the skewness of the jackknife values $\hat\theta_{(i)}$,

$$
a = \frac{\sum(\bar\theta_{(\cdot)} - \hat\theta_{(i)})^3}{6\big(\sum(\bar\theta_{(\cdot)} - \hat\theta_{(i)})^2\big)^{3/2}}.
$$

The interval endpoints are the replicate quantiles at levels $\Phi\big(z_0 + \frac{z_0 + z_\alpha}{1 - a(z_0 + z_\alpha)}\big)$ for $z_\alpha = \pm\Phi^{-1}(1 - \alpha/2)$. BCa intervals are
transformation-respecting and second-order accurate: their coverage error is $O(n^{-1})$.

### How well do they cover?

{{#include ../generated/statistics/bootstrap_coverage.md}}

For the mean of exponential data, all three intervals under-cover at $n = 10$ and approach 90% as $n$ grows; the bootstrap intervals are
slightly narrower than the $t$ interval because the bootstrap divides by $n$ rather than $n - 1$ and ignores the extra uncertainty of
estimating the variance. For the standard deviation, a statistic driven by the few largest values of a skewed sample, coverage
is poor: 56–61% at $n = 10$, still 81–83% at $n = 100$, with BCa consistently better than percentile. The bootstrap cannot invent
information about tails the sample has not seen.

![Bootstrap replicates](../generated/statistics/bootstrap_sd.svg)

The replicates of the standard deviation from one sample of 30 are lumpy, because the result depends mainly on how many copies
of the largest observation a resample contains. The BCa interval is shifted to the right of the percentile one, correcting
for the skewness, and covers the true value.

## Algorithm

1. For $b = 1, \ldots, B$: draw $n$ indices uniformly with replacement; evaluate $\hat\theta^*_b$ on the resample.
2. Standard error: the standard deviation of the replicates; bias: their mean minus $\hat\theta$.
3. Percentile interval: quantiles of the sorted replicates.
4. BCa: $z_0$ from the fraction below $\hat\theta$; $a$ from the $n$ jackknife values; adjusted quantile levels.

## Implementation

```rust,ignore
{{#include ../../../crates/statistics/src/bootstrap.rs:bootstrap}}
```

```rust,ignore
{{#include ../../../crates/statistics/src/bootstrap.rs:bca}}
```

- `bootstrap` takes a seed, so results are reproducible; the statistic is any `Fn(&[f64]) -> f64`.
- `bca_interval` needs the original data and statistic again, for the jackknife.

## In practice

```rust,ignore
{{#include ../../../crates/statistics/examples/statistics_bootstrap.rs}}
```

```text
median = 115.0 ms, bootstrap standard error = 9.9, bias = +0.84
95% percentile interval: [99.5, 138.0]
95% BCa interval:        [98.0, 133.0]
coefficient of variation = 0.426, 95% BCa interval [0.262, 0.582]
```

**Strengths**

- Works for any statistic, however complicated, with no derivation.
- BCa intervals correct for bias and skewness, with second-order accuracy.

**Weaknesses**

- Under-covers for small samples and for statistics that depend on the tails; it cannot do better than the sample.
- Fails for some statistics: the maximum, and statistics of heavy-tailed data with infinite variance.
- Independent observations only; dependent data (time series) need block resampling, which is not implemented.
- $B$ evaluations of the statistic can be expensive.

## Exercises

1. Show that the probability that a given observation is absent from a resample tends to $e^{-1}$.
2. Compute the exact bootstrap distribution of the mean of the sample $\{1, 2, 4\}$ by enumerating all 27 resamples.
3. Show that the bootstrap variance of the mean is $\frac{1}{n^2}\sum(x_i - \bar x)^2$, and compare with $s^2/n$.
4. Show that the bootstrap fails for the sample maximum of a uniform distribution: what is the probability that the maximum
   of a resample equals the sample maximum?
5. Compare the coverage of percentile and BCa intervals for the correlation coefficient of bivariate normal data with $\varrho = 0.8$, $n = 20$.

## Further reading

- B. Efron, "Bootstrap methods: another look at the jackknife", *Annals of Statistics* 7 (1979) 1–26.
- B. Efron, "Better bootstrap confidence intervals", *Journal of the American Statistical Association* 82 (1987) 171–185.
- B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman and Hall, 1993.
- T. J. DiCiccio and B. Efron, "Bootstrap confidence intervals", *Statistical Science* 11 (1996) 189–228.
- A. C. Davison and D. V. Hinkley, *Bootstrap Methods and their Application*, Cambridge University Press, 1997.
