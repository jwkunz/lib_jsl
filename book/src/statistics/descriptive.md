# Descriptive statistics

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Summarise a sample by its location, spread and shape: means, variance, quantiles, robust measures, skewness and kurtosis |
| **Devised by** | Karl Pearson (1890s), moments, skewness and the standard deviation; B. P. Welford (1962), one-pass updating; Tony Chan, Gene Golub and Randall LeVeque (1983), analysis of variance algorithms; John Tukey (1977), the five-number summary; Frank Hampel (1974), robustness and the median absolute deviation; Rob Hyndman and Yanan Fan (1996), sample quantile definitions |
| **Accuracy** | Mean and variance to a few units in the last place regardless of the offset of the data (figure below); values agree with SciPy and NumPy to $10^{-15}$ |
| **Cost** | $O(n)$ for moments; $O(n \log n)$ for quantiles (a sort) |
| **API** | [`descriptive`](../../api/lib_jsl_statistics/descriptive/index.html): `mean`, `variance`, `standard_deviation`, `standard_error`, `weighted_mean`, `geometric_mean`, `harmonic_mean`, `quantile`, `median`, `interquartile_range`, `median_absolute_deviation`, `skewness`, `excess_kurtosis`, [`summary`](../../api/lib_jsl_statistics/descriptive/fn.summary.html) |

</div>

## Overview

Descriptive statistics compress a sample $x_1, \ldots, x_n$ into a few numbers. **Location** says where the data sit (mean,
median), **spread** how far they scatter (standard deviation, interquartile range, median absolute deviation), and **shape**
how they depart from symmetry and from normal tails (skewness, kurtosis). Two questions recur. Is the computation
**numerically sound**, given that data often carry a large common offset? And is the statistic **robust**, or can a single
wild value move it arbitrarily far?

## History

Karl Pearson introduced the term "standard deviation" (1893) and the moment measures of skewness and kurtosis in the 1890s,
building on Gauss's and Laplace's theory of errors. R. A. Fisher's $k$-statistics (1929) gave unbiased versions. As
computers arrived, the textbook "one-pass" variance formula $\sum x^2 - n\bar x^2$ was found to fail on real data; B. P. Welford
(1962) proposed a stable updating formula, and Tony Chan, Gene Golub and Randall LeVeque (1983) analysed all the
variants, recommending the corrected two-pass algorithm when the data can be read twice. John Tukey's *Exploratory Data
Analysis* (1977) popularised medians, quartiles and the box plot; Frank Hampel (1968, 1974) founded robust statistics
with the influence function and the breakdown point. Rob Hyndman and Yanan Fan (1996) catalogued nine competing
definitions of the sample quantile used by statistical packages.

## The idea

### The mean and compensated summation

The mean $\bar x = \frac1n\sum x_i$ is the least-squares location. Summing $n$ numbers naively accumulates a rounding error that grows
with $n$; the library uses **Neumaier's compensated summation**, which carries the rounding error of each addition in a second
variable, so the sum is accurate to a few units in the last place for any $n$. Weighted, geometric ($\exp$ of the mean log)
and harmonic (reciprocal of the mean reciprocal) means are also provided.

### The variance: why the textbook formula fails

The sample variance is $s^2 = \frac{1}{n - 1}\sum(x_i - \bar x)^2$; dividing by $n - 1$ rather than $n$ (`ddof = 1`) makes it unbiased. Algebraically
$\sum(x_i - \bar x)^2 = \sum x_i^2 - n\bar x^2$, which needs a single pass over the data, but when the mean is large compared with the spread
the two terms are nearly equal and their difference cancels catastrophically. The library uses the **corrected two-pass
algorithm**: compute $\bar x$, then $\sum(x_i - \bar x)^2 - \frac1n\big(\sum(x_i - \bar x)\big)^2$, where the second term removes the error left in $\bar x$.

![Variance accuracy](../generated/statistics/variance_accuracy.svg)

For 1000 values $c + U$ with $U$ uniform on $[0, 1)$, the one-pass formula loses all accuracy once $c \approx 10^8$ (its errors above $10^3$ are
clipped), Welford's one-pass update degrades more slowly, and the two-pass algorithm stays at rounding level for every offset.

### Quantiles

The $p$-quantile splits the data so that a fraction $p$ lies below it. For a finite sample there is no unique definition; the
library uses Hyndman and Fan's **type 7**, the default of R and NumPy: sort the data, set $h = (n - 1)p$, and interpolate linearly
between the order statistics $x_{(\lfloor h\rfloor + 1)}$ and $x_{(\lfloor h\rfloor + 2)}$. The median is the 0.5-quantile, the quartiles the 0.25 and 0.75 quantiles, and `summary` returns
Tukey's five-number summary with the count, mean and standard deviation.

### Robustness and the breakdown point

The **breakdown point** of an estimator is the largest fraction of the data that can be replaced by arbitrary values without
carrying the estimate arbitrarily far. The mean and standard deviation have breakdown point 0: a single value suffices. The
median and the **median absolute deviation** $\operatorname{MAD} = \operatorname{median}|x_i - \operatorname{median}(x)|$ have breakdown point 50%, the largest possible; the
interquartile range has 25%. Scaled by $1.4826 = 1/\Phi^{-1}(0.75)$, the MAD estimates $\sigma$ for normal data, as does $\operatorname{IQR}/1.349$.

| | |
|:---:|:---:|
| ![Robust location](../generated/statistics/robust_location.svg) | ![Robust scale](../generated/statistics/robust_scale.svg) |

As standard normal data are progressively replaced by values near 10, the mean and standard deviation drift at once; the
interquartile range holds until a quarter of the data are contaminated and then jumps; the median and MAD drift only slowly
until half are.

### Shape: skewness and kurtosis

With central moments $m_k = \frac1n\sum(x_i - \bar x)^k$, the **skewness** $g_1 = m_3/m_2^{3/2}$ measures asymmetry (positive for a long right tail) and the
**excess kurtosis** $g_2 = m_4/m_2^2 - 3$ measures tail weight relative to the normal distribution, for which it is 0. These are the biased
versions, as in `scipy.stats.skew` and `kurtosis`; both are very sensitive to outliers, since they raise deviations to the third
and fourth powers.

## Algorithm

1. **Mean**: Neumaier-compensated sum divided by $n$.
2. **Variance**: mean $\bar x$; then $S_2 = \sum(x_i - \bar x)^2$ and $S_1 = \sum(x_i - \bar x)$; return $(S_2 - S_1^2/n)/(n - \text{ddof})$.
3. **Quantile**: sort (rejecting NaN); interpolate at $h = (n - 1)p$.

## Implementation

```rust,ignore
{{#include ../../../crates/statistics/src/descriptive.rs:mean_variance}}
```

```rust,ignore
{{#include ../../../crates/statistics/src/descriptive.rs:quantile}}
```

Against SciPy 1.17 and NumPy 2.4 on a skewed sample of 20 values (the [kernel density](kernel-density.md) estimate is included for
completeness):

{{#include ../generated/statistics/scipy_descriptive.md}}

- Every function returns an error for an empty sample; quantile functions reject NaN; geometric and harmonic means need positive
  values.

## In practice

```rust,ignore
{{#include ../../../crates/statistics/examples/statistics_descriptive.rs}}
```

```text
n = 12, mean = 143.9, sd = 348.5
five-number summary: 28.0 38.0 42.8 53.4 1250.0
robust spread: IQR = 15.4, 1.4826 MAD = 12.4
skewness = 3.01, excess kurtosis = 7.08
geometric mean = 56.1, 90th percentile = 60.4
```

One income of 1250 makes the mean more than three times the median and the standard deviation 28 times the MAD.

**Strengths**

- Numerically robust mean and variance; quantiles compatible with R and NumPy.
- Robust alternatives (median, IQR, MAD) alongside the classical statistics.

**Weaknesses**

- Quantiles sort a copy of the data, $O(n \log n)$; selection algorithms would be $O(n)$.
- No one-pass (streaming) accumulator; Welford's update would be needed for data seen once.

## Exercises

1. Show that $E[s^2] = \sigma^2$ when dividing by $n - 1$, and find the bias when dividing by $n$.
2. Derive Welford's update $M_k = M_{k-1} + (x_k - M_{k-1})/k$, $S_k = S_{k-1} + (x_k - M_{k-1})(x_k - M_k)$.
3. Show that the correction term $S_1^2/n$ in the two-pass algorithm is zero in exact arithmetic, and estimate its size in floating point.
4. Compute the breakdown points of the trimmed mean that discards the top and bottom 10%, and of the interquartile range.
5. Compute the type 7 and type 6 ($h = (n + 1)p - 1$) quantiles of $1, 2, 3, 4$ at $p = 0.1$.

## Further reading

- T. F. Chan, G. H. Golub and R. J. LeVeque, "Algorithms for computing the sample variance: analysis and recommendations", *The American
  Statistician* 37 (1983) 242–247.
- B. P. Welford, "Note on a method for calculating corrected sums of squares and products", *Technometrics* 4 (1962) 419–420.
- R. J. Hyndman and Y. Fan, "Sample quantiles in statistical packages", *The American Statistician* 50 (1996) 361–365.
- P. J. Huber and E. M. Ronchetti, *Robust Statistics*, 2nd ed., Wiley, 2009.
- J. W. Tukey, *Exploratory Data Analysis*, Addison-Wesley, 1977.
