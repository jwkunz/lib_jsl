# More hypothesis tests

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Test whether a sample is normal; whether groups have equal variances; whether groups or paired measurements differ, without assuming normality; compute exact p-values for small tables and counts; test any statistic by permutation; and control errors when many tests are made at once |
| **Devised by** | Samuel Shapiro and Martin Wilk (1965), Patrick Royston (1992, 1995); Theodore Anderson and Donald Darling (1952); Ralph D'Agostino (1970) and Egon Pearson (1973); Carlos Jarque and Anil Bera (1980); Howard Levene (1960), Morton Brown and Alan Forsythe (1974); Maurice Bartlett (1937); Michael Fligner and Timothy Killeen (1976); William Kruskal and Allen Wallis (1952); Milton Friedman (1937); Frank Wilcoxon (1945); Ronald Fisher (1922, 1935); Sture Holm (1979); Yoav Benjamini and Yosef Hochberg (1995), Benjamini and Daniel Yekutieli (2001) |
| **Accuracy** | Statistics and p-values agree with `scipy.stats` to $10^{-13}$ or better, except Shapiro-Wilk at $10^{-8}$ |
| **Cost** | $O(n\log n)$ for the rank tests; the exact Wilcoxon distribution in $O(n^3)$; Fisher's and the binomial test in $O(n)$; a permutation test costs one evaluation of the statistic per split |
| **Read first** | [Hypothesis tests](hypothesis-tests.md), [Probability distributions](distributions.md) |
| **API** | [`normality`](../../api/lib_jsl_statistics/normality/index.html) (`shapiro`, `anderson_normal`, `jarque_bera`, `normaltest`, `skewtest`, `kurtosistest`), [`comparisons`](../../api/lib_jsl_statistics/comparisons/index.html) (`levene`, `bartlett`, `fligner`, `kruskal`, `friedman`, `wilcoxon`, `fisher_exact`, `binomtest`, `permutation_test`, `adjust_p_values`) |

</div>

## Overview

The [hypothesis tests entry](hypothesis-tests.md) covered the classical tests of means and of goodness of fit. This one adds the
tests that are needed around them:

- **Normality tests** check the assumption behind t-tests and ANOVA.
- **Tests of equal variances** check the other assumption of pooled t-tests and ANOVA.
- **Rank tests** compare groups (Kruskal-Wallis), repeated measurements (Friedman) and paired differences (Wilcoxon) without
  assuming normality.
- **Exact tests** give p-values without large-sample approximations: Fisher's for 2 × 2 tables, and the binomial test.
- **Permutation tests** give an exact p-value for any statistic, by recomputing it under every relabelling of the data.
- **Multiple-comparison adjustments** control the error rate when many hypotheses are tested together.

Each follows the corresponding function of `scipy.stats`.

## History

Samuel Shapiro and Martin Wilk (1965) built their test from the regression of the ordered sample on the expected normal order
statistics. Patrick Royston (1992, 1995) extended it to large samples with the approximations used by every implementation today.
Theodore Anderson and Donald Darling (1952) weighted the distance between the empirical and the normal distribution function towards the tails.
Ralph D'Agostino and Egon Pearson combined tests of skewness and kurtosis into one statistic, and Carlos Jarque and Anil Bera (1980)
derived a simpler version for econometrics.

Maurice Bartlett's test of equal variances (1937) is sensitive to non-normality. Howard Levene (1960), Morton Brown and Alan Forsythe
(1974), and Michael Fligner and Timothy Killeen (1976) made robust alternatives. Rank tests began with Milton Friedman (1937) and
Frank Wilcoxon (1945). William Kruskal and Allen Wallis (1952) extended Wilcoxon's rank-sum test to many groups. Ronald Fisher
introduced both the exact test for 2 × 2 tables and the logic of permutation tests in his "lady tasting tea" (1935). Carlo Bonferroni's
inequality (1936) gave the first protection against multiple comparisons. Sture Holm (1979) sharpened it. Yoav Benjamini and Yosef
Hochberg (1995) changed the goal from avoiding any false rejection to controlling the false discovery rate, which suits studies of
thousands of hypotheses.

## The idea

### Tests of normality

Each test measures a different departure from the normal distribution:

- **Shapiro-Wilk** compares the ordered sample $x_{(1)} \le \cdots \le x_{(n)}$ with the expected order statistics of a normal sample. Its statistic
  $W = \big(\sum_i a_i x_{(i)}\big)^2 / \sum_i (x_i - \bar x)^2$ is the squared correlation of a normal probability plot. Near 1 means normal. Royston's algorithm approximates the
  coefficients $a_i$ and transforms $\ln(1 - W)$ to approximate normality to give the p-value.
- **Anderson-Darling** measures the squared distance between the empirical distribution function and the fitted normal, with weight
  $1/(F(1 - F))$, which emphasises the tails: $A^2 = -n - \frac1n\sum_i (2i - 1)\big[\ln\Phi(z_{(i)}) + \ln(1 - \Phi(z_{(n+1-i)}))\big]$. It is compared with Stephens's tabulated critical values.
- **D'Agostino-Pearson** transforms the sample skewness and kurtosis each to an approximately standard normal $Z$, and adds their squares: $K^2 = Z_{\text{skew}}^2 + Z_{\text{kurt}}^2 \sim \chi^2_2$.
- **Jarque-Bera** uses the same two moments without the transformations, $JB = \frac n6\big(S^2 + \tfrac14K^2\big) \sim \chi^2_2$, which is accurate only for large samples.

No test is best against every alternative. A simulation of 2000 samples of 50 shows how often each rejects at the 5% level:

<p class="caption" id="tab-10-7"><strong>Table 10.7.</strong> Power of the normality tests at the 5% level, n = 50.</p>

{{#include ../generated/statistics/more_tests_power.md}}

Shapiro-Wilk is the best general choice. The moment tests do well against heavy tails, but Jarque-Bera never detects the light-tailed
uniform distribution at this size: its kurtosis is below normal, and the $\chi^2$ approximation is poor for $n = 50$.

### Equal variances

**Bartlett's test** compares the pooled variance with the individual ones through their logarithms. It is optimal for normal data and
misleading otherwise, because heavy tails mimic unequal variances. **Levene's test** with Brown and Forsythe's median centring runs a
one-way ANOVA on the absolute deviations $|x_{ij} - \tilde x_j|$ from each group's median. The **Fligner-Killeen test** replaces those deviations by
normal scores of their ranks. Both are robust to non-normality.

### Rank tests

Replacing observations by their ranks makes a test's null distribution independent of the data's distribution:

- **Kruskal-Wallis** compares the mean ranks of $k$ groups, $H = \frac{12}{N(N+1)}\sum_j R_j^2/n_j - 3(N + 1)$, corrected for ties, approximately $\chi^2_{k-1}$. It is the rank version of
  one-way ANOVA.
- **Friedman's test** ranks $k$ treatments within each of $n$ blocks (subjects measured repeatedly), and compares the treatments' rank sums.
- The **Wilcoxon signed-rank test** ranks the absolute paired differences and sums the ranks of the positive ones, $W^+$. Under the null
  hypothesis of a distribution symmetric about zero, each rank is equally likely to carry either sign. The exact distribution of $W^+$ is that
  of the sum of a random subset of $\{1, \ldots, n\}$, which dynamic programming counts in $O(n^3)$. With ties, the library enumerates all $2^n$ sign changes for
  $n \le 13$ and otherwise uses the normal approximation, as SciPy does.

### Exact tests

With both margins of a 2 × 2 table fixed, the top-left count follows a **hypergeometric distribution**. **Fisher's exact test** sums the probabilities of all tables
at least as extreme as the observed one. For the two-sided test, these are the tables no more probable than it. The **binomial test** does the same for $k$ successes in $n$ trials.
Both are exact for any sample size, so they replace the $\chi^2$ tests when expected counts are small.

### Permutation tests

If the two samples come from the same distribution, every way of dividing the pooled observations into groups of the original sizes is
equally likely. A **permutation test** computes any statistic, such as the difference of means or medians or a ratio of variances, for each division. The
p-value is the fraction at least as extreme as the observed one. With up to `n_resamples` divisions the test enumerates them all and is
exact. Otherwise it samples random divisions and counts the observed one among them, so that the p-value is never zero.

### Many tests at once

Testing 1000 true null hypotheses at 5% rejects about 50 of them. Two kinds of protection exist:

- The **family-wise error rate**, the probability of any false rejection. Bonferroni multiplies each p-value by $m$. Holm's step-down
  version multiplies the $i$-th smallest by $m - i + 1$ and is uniformly more powerful.
- The **false discovery rate**, the expected share of false rejections among all rejections. Benjamini-Hochberg multiplies the $i$-th smallest
  p-value by $m/i$ (then enforces monotonicity). Benjamini-Yekutieli adds a factor $\sum_{i\le m} 1/i$ that makes it valid under any dependence.

With 1000 t-tests, of which 100 have a real effect:

<p class="caption" id="tab-10-8"><strong>Table 10.8.</strong> Rejections at 5% among 1000 tests, 100 of them with a real effect.</p>

{{#include ../generated/statistics/more_tests_fdr.md}}

Without adjustment a third of the discoveries are false. The family-wise methods find almost nothing. Benjamini-Hochberg finds many
true effects while keeping the share of false ones near the target.

## Algorithm

**Shapiro-Wilk** (Royston's AS R94), for $3 \le n \le 5000$:

1. Compute $m_i = \Phi^{-1}((i - 0.375)/(n + 0.25))$, the coefficients $a_n$ and $a_{n-1}$ by polynomials in $1/\sqrt n$, and the others proportional to $m_i$.
2. Compute $W$.
3. For $n \le 11$ transform $-\ln(\gamma - \ln(1 - W))$, otherwise $\ln(1 - W)$, by fitted mean and standard deviation polynomials. The p-value is a normal upper tail.

**Exact Wilcoxon**: counts $c_s$ of subsets of $\{1, \ldots, n\}$ summing to $s$, by adding the ranks one at a time (`c[s] += c[s − k]`); the p-value is a sum of $c_s/2^n$.

**Permutation test**: enumerate the $\binom{n_x + n_y}{n_x}$ subsets in lexicographic order, or shuffle by Fisher-Yates.

**Benjamini-Hochberg**: sort the p-values, form $p_{(i)}m/i$, take running minima from the largest down, and cap at 1.

## Implementation

```rust,ignore
{{#include ../../../crates/statistics/src/normality.rs:shapiro}}
```

```rust,ignore
{{#include ../../../crates/statistics/src/comparisons.rs:wilcoxon}}
```

- Every test returns the same `TestResult` as the [classical tests](hypothesis-tests.md): statistic, p-value and degrees of freedom.
- `anderson_normal` returns critical values, not a p-value, as SciPy did before 1.17 introduced tabulated interpolation.
- `adjust_p_values` covers SciPy's `false_discovery_control` (Benjamini-Hochberg and -Yekutieli) and adds Bonferroni and Holm.

## In practice

Every test is compared with `scipy.stats` from SciPy 1.17.1 on samples written by
`book/generator/data/make_more_tests_refs.py`. The differences are relative:

<p class="caption" id="tab-10-9"><strong>Table 10.9.</strong> Hypothesis tests against SciPy.</p>

{{#include ../generated/statistics/more_tests_scipy.md}}

Shapiro-Wilk differs at $10^{-8}$, because the normal quantiles inside Royston's approximation come from a different (slightly less
precise) inverse normal function. Every other test agrees to rounding, including the exact and permutation p-values.

**Strengths**

- Exact p-values for the Wilcoxon, Fisher and binomial tests and for permutation tests of any statistic.
- Robust alternatives for every classical assumption: rank tests for normality of the data, Levene and Fligner for equal variances.
- False-discovery-rate control for large families of tests.

**Weaknesses**

- No Cramér-von Mises test, Brunner-Munzel test, Tukey's or Dunnett's multiple-comparison procedures, or `monte_carlo_test`.
- Permutation tests are for two independent samples. Paired designs use `wilcoxon` or a permutation of signs written by the caller.
- Fisher's exact test is for 2 × 2 tables only.

## Exercises

1. Show that $W$ is the squared correlation between the ordered sample and the coefficients $a_i$, and therefore at most 1.
2. Count by hand the distribution of $W^+$ for $n = 4$, and find the exact two-sided p-value when all four differences are positive.
3. Show that the two-sided Fisher p-value for the table $[[a, 0], [0, a]]$ is $2/\binom{2a}{a}$.
4. Why must a permutation test that samples random splits count the observed split? What is its smallest possible p-value with $B$ resamples?
5. Show that Holm's procedure rejects every hypothesis Bonferroni rejects.

<details>
<summary>Hints and answers</summary>

1. The $a_i$ have $\sum a_i^2 = 1$ and $\sum a_i = 0$, so $W = \mathrm{corr}(a, x_{(\cdot)})^2$ by the Cauchy-Schwarz inequality.
2. The 16 subsets of $\{1, 2, 3, 4\}$ give sums 0 to 10 with counts 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1. All positive gives $W^+ = 10$, so $P(W^+ \ge 10) = 1/16$ and the two-sided p-value is $1/8$.
3. The top-left count $x$ has probability $\binom ax^2/\binom{2a}{a}$. The least probable tables are $x = a$ (observed) and $x = 0$, each with probability $1/\binom{2a}{a}$, and every other table is more probable, so the p-value is $2/\binom{2a}{a}$.
4. Otherwise a p-value of exactly 0 could occur, which is never valid for a finite sample. Counting the observed split makes the test exact for the random procedure. The smallest p-value is $1/(B + 1)$.
5. The smallest p-value is multiplied by $m$ in both. Holm then multiplies the others by smaller factors, and the running maximum never exceeds Bonferroni's $mp_{(i)}$.

</details>

## Further reading

- P. Royston, "Remark AS R94: A remark on algorithm AS 181: The W-test for normality", *Applied Statistics* 44 (1995) 547–551.
- R. B. D'Agostino and M. A. Stephens (eds.), *Goodness-of-Fit Techniques*, Marcel Dekker, 1986.
- M. Hollander, D. A. Wolfe and E. Chicken, *Nonparametric Statistical Methods*, 3rd ed., Wiley, 2014.
- P. Good, *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, 3rd ed., Springer, 2005.
- Y. Benjamini and Y. Hochberg, "Controlling the false discovery rate: a practical and powerful approach to multiple testing", *Journal of the Royal Statistical Society B* 57 (1995) 289–300.
