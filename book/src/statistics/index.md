# Statistics

Statistics turns data into statements about the process that produced them. **Descriptive**
statistics summarise a sample; **correlation** measures association between variables;
**hypothesis tests** ask whether an apparent effect could be chance; **density estimation**
recovers the shape of a distribution without assuming its family; and the **bootstrap**
estimates the uncertainty of any statistic by resampling. All of them rest on the
[random numbers](../random/index.md) and [statistical distribution functions](../special_functions/statistical.md)
of earlier chapters.

## The central questions

- **Is the computation sound?** Summaries of large or offset data lose accuracy to cancellation
  unless computed carefully (the variance is the classic example).
- **Is the statistic robust?** A single wild value can move a mean or a standard deviation arbitrarily far;
  medians, ranks and the median absolute deviation resist it.
- **Is the reference distribution right?** A p-value is only as good as the assumptions behind it: normality for
  $t$ tests and ANOVA, equal variances for Student's two-sample test, large expected counts for $\chi^2$. Rank tests and the
  bootstrap need fewer assumptions.
- **How much data?** Kernel density estimates converge like $n^{-4/5}$, and bootstrap intervals for statistics that depend on
  the tails under-cover badly for small samples.

## Testing the library

Every statistic and p-value was compared with SciPy 1.17 on fixed data sets
([hypothesis tests](hypothesis-tests.md#checking-against-scipy), [descriptive statistics](descriptive.md#implementation)):
all agree to about $10^{-14}$, except the Kolmogorov-Smirnov p-values, which are asymptotic approximations. Simulations
check that the tests reject a true null hypothesis at the stated rate.

## In the library

The crate is `lib_jsl_statistics` ([API reference](../../api/lib_jsl_statistics/index.html)). Functions take slices of `f64` and return
`Result`s; tests return a `TestResult` with the statistic, p-value and degrees of freedom.

| You need | Use | Entry |
|---|---|---|
| Mean, variance, quantiles, robust and shape measures | `descriptive::*`, `summary` | [Descriptive statistics](descriptive.md) |
| Probabilities, quantiles and moments of a named distribution; fitting one to data | `Normal`, `Gamma`, `Weibull`, … with `ContinuousDistribution`, `Fit` | [Probability distributions](distributions.md) |
| Association between two variables | `pearson`, `spearman`, `kendall`, `correlation_matrix` | [Correlation and covariance](correlation.md) |
| Compare means or distributions, test counts | `t_test_*`, `anova_one_way`, `mann_whitney_u`, `ks_test_*`, `chi_square_*` | [Hypothesis tests](hypothesis-tests.md) |
| A smooth density from a sample | `GaussianKde` | [Kernel density estimation](kernel-density.md) |
| Standard errors and confidence intervals for any statistic | `bootstrap`, `bca_interval` | [The bootstrap](bootstrap.md) |
