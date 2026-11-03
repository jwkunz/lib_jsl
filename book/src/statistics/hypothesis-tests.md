# Hypothesis tests

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Decide whether data are consistent with a null hypothesis (equal means, a given distribution, independence) and quantify the evidence against it by a p-value |
| **Devised by** | Karl Pearson (1900), the $\chi^2$ test; "Student" (William Sealy Gosset, 1908), the $t$ test; R. A. Fisher (1920s), significance testing and the analysis of variance; Jerzy Neyman and Egon Pearson (1933), errors of two kinds and power; Andrey Kolmogorov (1933) and Nikolai Smirnov (1939), distribution tests; Frank Wilcoxon (1945), Henry Mann and Donald Whitney (1947), rank tests; Bernard Welch (1947), unequal variances |
| **Accuracy** | Statistics and p-values agree with SciPy to $10^{-14}$ for every test with an exact reference distribution; the Kolmogorov-Smirnov p-values are asymptotic approximations (table below) |
| **Cost** | $O(n)$ for the $t$ tests, $\chi^2$ and ANOVA; $O(n \log n)$ for the rank and Kolmogorov-Smirnov tests |
| **API** | [`hypothesis_tests`](../../api/lib_jsl_statistics/hypothesis_tests/index.html): `t_test_one_sample`, [`t_test_two_sample`](../../api/lib_jsl_statistics/hypothesis_tests/fn.t_test_two_sample.html), `t_test_paired`, `chi_square_goodness_of_fit`, `chi_square_independence`, `anova_one_way`, `ks_test_one_sample`, `ks_test_two_sample`, [`mann_whitney_u`](../../api/lib_jsl_statistics/hypothesis_tests/fn.mann_whitney_u.html), `kolmogorov_survival`, [`Alternative`](../../api/lib_jsl_statistics/hypothesis_tests/enum.Alternative.html), [`TestResult`](../../api/lib_jsl_statistics/hypothesis_tests/struct.TestResult.html) |

</div>

## Overview

A hypothesis test computes a **statistic** that measures departure from a **null hypothesis** $H_0$, and its **p-value**: the probability,
if $H_0$ were true, of a statistic at least as extreme as the one observed. A small p-value is evidence against $H_0$; rejecting when
$p < \alpha$ makes the probability of a false rejection (the **size**, or type I error rate) equal to $\alpha$, *provided the reference distribution
is right*. The probability of rejecting when $H_0$ is false is the **power**. Every test here returns a `TestResult` with the statistic, the
p-value and the degrees of freedom, and the location tests accept a two-sided or one-sided `Alternative`.

Tests fall into two families. **Parametric** tests ($t$, ANOVA) assume normal data and are the most powerful when that holds.
**Nonparametric** tests (rank tests, Kolmogorov-Smirnov) assume only continuity, and keep their size and power when tails are heavy.

## History

Karl Pearson's $\chi^2$ test of goodness of fit (1900) was the first modern significance test. William Sealy Gosset, a chemist at the
Guinness brewery in Dublin, published the $t$ distribution under the pseudonym "Student" (1908) to handle the small samples of
brewing experiments; R. A. Fisher proved it, extended it to two samples and regression, and invented the analysis of variance
(1918–1925) for agricultural field trials at Rothamsted. Jerzy Neyman and Egon Pearson (1933) recast testing as a decision between
hypotheses with errors of two kinds. Andrey Kolmogorov (1933) found the limiting distribution of the largest deviation of an
empirical distribution function, and Nikolai Smirnov (1939) extended it to two samples. Frank Wilcoxon (1945) proposed rank-sum
tests, and Henry Mann and Donald Whitney (1947) studied them. Bernard Welch (1947) and Franklin Satterthwaite (1946) solved
the practical side of the Behrens-Fisher problem of comparing means with unequal variances.

## The idea

### Student's and Welch's $t$ tests

For a sample of size $n$ with mean $\bar x$ and standard deviation $s$, the **one-sample** statistic $t = (\bar x - \mu_0)/(s/\sqrt n)$ has Student's $t$ distribution with
$n - 1$ degrees of freedom when the data are normal with mean $\mu_0$. The **paired** test applies it to differences. To compare two independent
samples, **Student's** test pools the variances, $s_p^2 = ((n_1 - 1)s_1^2 + (n_2 - 1)s_2^2)/(n_1 + n_2 - 2)$, and assumes they are equal; **Welch's** test uses
$t = (\bar x_1 - \bar x_2)/\sqrt{s_1^2/n_1 + s_2^2/n_2}$ with the Welch-Satterthwaite degrees of freedom

$$
\nu = \frac{(s_1^2/n_1 + s_2^2/n_2)^2}{(s_1^2/n_1)^2/(n_1 - 1) + (s_2^2/n_2)^2/(n_2 - 1)}.
$$

Welch's test is the safer default: when the smaller sample has the larger variance, Student's test rejects a true null five
times as often as it should (table below), while Welch's loses almost nothing when the variances are equal.

### Chi-square tests for counts

For counts $O_i$ in $k$ categories with expected counts $E_i$ under $H_0$, **Pearson's statistic** $X^2 = \sum(O_i - E_i)^2/E_i$ is approximately $\chi^2$ with $k - 1$ degrees of
freedom (**goodness of fit**). For an $r \times c$ contingency table, the expected counts under **independence** are $E_{ij} = R_iC_j/N$ from the row and column totals,
and $X^2$ has $(r - 1)(c - 1)$ degrees of freedom. The approximation needs expected counts of about 5 or more.

### One-way analysis of variance

To compare the means of $k$ groups, ANOVA splits the total sum of squares into the spread **between** group means and the spread
**within** groups. The ratio $F = \frac{SS_B/(k - 1)}{SS_W/(N - k)}$ has the $F$ distribution with $(k - 1, N - k)$ degrees of freedom for normal groups with equal
variances and equal means. The library computes the upper tail $P(F > f) = I_{d_2/(d_2 + d_1 f)}(d_2/2, d_1/2)$ directly as an incomplete beta function; it used to
form $1 - \text{CDF}$, which lost every digit of p-values below $10^{-16}$.

### Kolmogorov-Smirnov tests

The **one-sample** statistic $D = \sup_x|F_n(x) - F(x)|$ is the largest gap between the empirical distribution function and a hypothesised
continuous $F$. Its null distribution does not depend on $F$, and $\sqrt nD$ converges to the **Kolmogorov distribution**,
$Q(\lambda) = 2\sum_{k\ge1}(-1)^{k-1}e^{-2k^2\lambda^2}$. The library uses Stephens' finite-sample correction $Q((\sqrt n + 0.12 + 0.11/\sqrt n)D)$, which is within a fraction of
a per cent of the exact p-value (table below). The **two-sample** test compares two empirical distribution functions, with the
effective size $nm/(n + m)$; the same correction is less accurate there, and for samples of 15 and 25 the p-value is 7% below the exact one.
The tests detect any difference in distribution, not just in location, at some cost in power against pure shifts.

### The Mann-Whitney $U$ test

Rank the combined samples; $U = R_1 - n_1(n_1 + 1)/2$, where $R_1$ is the rank sum of the first sample, counts the pairs in which the first sample's
value is larger. Under $H_0$ (identical continuous distributions) $U$ has mean $n_1n_2/2$ and variance $\frac{n_1n_2}{12}\big((n + 1) - \frac{\sum(t^3 - t)}{n(n - 1)}\big)$ with a
correction for tie groups of size $t$; the library uses the normal approximation with a continuity correction, as SciPy's
`method="asymptotic"`. Against normal shifts it has asymptotic efficiency $3/\pi \approx 0.955$ relative to the $t$ test, and it can be far more
powerful for heavy tails.

### Checking against SciPy

The cases below were computed with `scipy.stats` (version 1.17.1) by `book/generator/data/make_stats_refs.py`, and recomputed
with the library. The statistics agree to at least nine digits (the generator asserts it), and every p-value with an exact
reference distribution agrees to rounding:

{{#include ../generated/statistics/scipy_tests.md}}

### Size: do the tests reject a true null 5% of the time?

{{#include ../generated/statistics/size.md}}

The tests hold their size when their assumptions hold. Three departures stand out. The $t$ test on skewed (exponential) data of
size 10 rejects twice as often as it should; the central limit theorem repairs this by $n = 100$. Student's pooled test with unequal
variances and unequal sizes rejects a quarter of true nulls. And the discrete tests are somewhat conservative at small sizes.

### Power

| | |
|:---:|:---:|
| ![Power, normal](../generated/statistics/power_normal.svg) | ![Power, Cauchy](../generated/statistics/power_cauchy.svg) |

For normal data Welch's $t$ test is the most powerful, with Mann-Whitney a close second. For Cauchy data, whose variance is infinite,
the $t$ test barely gains power with the shift, while the rank and Kolmogorov-Smirnov tests detect it easily.

## Algorithm

All tests follow the same pattern: compute the statistic from sums, ranks or sorted samples; evaluate the tail of the reference
distribution ($t$, $\chi^2$, $F$, normal or Kolmogorov) with the [special functions](../special_functions/statistical.md), using the lower tail of symmetric
distributions to avoid cancellation.

## Implementation

```rust,ignore
{{#include ../../../crates/statistics/src/hypothesis_tests.rs:t_two_sample}}
```

```rust,ignore
{{#include ../../../crates/statistics/src/hypothesis_tests.rs:mann_whitney}}
```

- `anova_one_way` returns the `TestResult` (with the between-groups degrees of freedom) and the within-groups degrees of freedom.
- `chi_square_independence` applies no continuity correction, as `scipy.stats.chi2_contingency(correction=False)`.

## In practice

```rust,ignore
{{#include ../../../crates/statistics/examples/statistics_tests.rs}}
```

```text
Welch t = 0.266 on 10.0 df, p = 0.7954
Student t = 0.290 on 20 df, p = 0.7751
Mann-Whitney U = 97, p = 0.0161
Kolmogorov-Smirnov D = 0.600, p = 0.0220
chi-square = 5.263 on 1 df, p = 0.0218
```

One patient in the second group took 30 days to recover; that value inflates the variance and the $t$ tests see nothing, while
the rank tests find the shorter recovery times of the other nine.

**Strengths**

- The classical tests, with p-values matching SciPy; one-sided alternatives for the location tests.
- Rank and distribution tests for data that are not normal.

**Weaknesses**

- p-values of the rank and Kolmogorov-Smirnov tests are asymptotic; exact small-sample distributions are not implemented.
- No multiple-comparison corrections, post-hoc tests, or tests of normality beyond Kolmogorov-Smirnov.
- A p-value is not the probability that $H_0$ is true, and a non-significant result is not evidence of no effect; report effect sizes and
  confidence intervals as well.

## Exercises

1. Show that the paired $t$ test is more powerful than the two-sample test when the pairs are positively correlated.
2. Verify the Welch-Satterthwaite formula by matching the first two moments of $s_1^2/n_1 + s_2^2/n_2$ to a scaled $\chi^2$.
3. Prove that the null distribution of the Kolmogorov-Smirnov statistic does not depend on the continuous $F$.
4. Simulate the size of the $\chi^2$ goodness-of-fit test with 10 categories and expected counts of 1, 2 and 5.
5. Run 20 independent tests on null data and estimate the probability that at least one has $p < 0.05$. How does the Bonferroni correction help?

## Further reading

- E. L. Lehmann and J. P. Romano, *Testing Statistical Hypotheses*, 4th ed., Springer, 2022.
- M. A. Stephens, "EDF statistics for goodness of fit and some comparisons", *Journal of the American Statistical Association* 69 (1974) 730–737.
- B. L. Welch, "The generalization of 'Student's' problem when several different population variances are involved", *Biometrika* 34
  (1947) 28–35.
- R. L. Wasserstein and N. A. Lazar, "The ASA statement on p-values: context, process, and purpose", *The American Statistician* 70
  (2016) 129–133.
