"""Reference values for the statistics chapter, from SciPy.

Run with a Python that has SciPy (the book used SciPy 1.17.1):

    python book/generator/data/make_stats_refs.py > book/generator/data/stats_reference.txt

Each output line is `case|function|options|x|y|statistic|p_value`, with comma-separated data. `y`
is empty for one-sample cases; for contingency tables and ANOVA groups it holds further rows
separated by `;`. The generator recomputes every case with the library and tabulates the
differences.
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(20251102)


def fmt(v):
    return ",".join(repr(float(t)) for t in v)


def out(case, function, options, x, y, statistic, p):
    print(f"{case}|{function}|{options}|{x}|{y}|{float(statistic)!r}|{float(p)!r}")


a = np.round(rng.normal(0.4, 1.0, 15), 3)
b = np.round(rng.normal(0.0, 2.0, 25), 3)
skewed = np.round(rng.exponential(1.0, 20), 3)
before = np.round(rng.normal(10, 2, 12), 2)
after = np.round(before - 0.6 + rng.normal(0, 1, 12), 2)
ties_x = rng.integers(0, 6, 30).astype(float)
ties_y = (ties_x + rng.integers(-2, 3, 30)).astype(float)

for alt, name in [("two-sided", "two"), ("less", "less"), ("greater", "greater")]:
    r = stats.ttest_1samp(a, 0.0, alternative=alt)
    out(f"one-sample t, {name}", "t1", alt, fmt(a), "", r.statistic, r.pvalue)
r = stats.ttest_ind(a, b, equal_var=True)
out("Student two-sample t", "t2", "pooled", fmt(a), fmt(b), r.statistic, r.pvalue)
r = stats.ttest_ind(a, b, equal_var=False)
out("Welch t", "t2", "welch", fmt(a), fmt(b), r.statistic, r.pvalue)
r = stats.ttest_rel(before, after)
out("paired t", "tp", "two-sided", fmt(before), fmt(after), r.statistic, r.pvalue)

counts = np.array([18.0, 25.0, 14.0, 22.0, 31.0, 10.0])
r = stats.chisquare(counts)
out("chi-square goodness of fit, uniform", "gof", "", fmt(counts), "", r.statistic, r.pvalue)
expected = np.array([0.1, 0.2, 0.1, 0.2, 0.3, 0.1])
r = stats.chisquare(counts, expected * counts.sum())
out("chi-square goodness of fit, given", "gof", "expected", fmt(counts), fmt(expected), r.statistic, r.pvalue)
table = np.array([[20.0, 15.0, 30.0], [25.0, 35.0, 18.0]])
chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
out("chi-square independence, 2 x 3", "ind", "", fmt(table[0]), fmt(table[1]), chi2, p)

g1, g2, g3 = np.round(rng.normal(5.0, 1, 8), 2), np.round(rng.normal(5.8, 1, 10), 2), np.round(rng.normal(6.5, 1, 7), 2)
r = stats.f_oneway(g1, g2, g3)
out("one-way ANOVA, 3 groups", "anova", "", fmt(g1), fmt(g2) + ";" + fmt(g3), r.statistic, r.pvalue)

normal_sample = np.round(rng.normal(0.3, 1.0, 40), 3)
r = stats.ks_1samp(normal_sample, stats.norm.cdf, method="asymp")
out("Kolmogorov-Smirnov vs N(0, 1), asymptotic", "ks1", "", fmt(normal_sample), "", r.statistic, r.pvalue)
r = stats.ks_1samp(normal_sample, stats.norm.cdf, method="exact")
out("Kolmogorov-Smirnov vs N(0, 1), exact", "ks1", "", fmt(normal_sample), "", r.statistic, r.pvalue)
r = stats.ks_2samp(a, b, method="asymp")
out("Kolmogorov-Smirnov two-sample, asymptotic", "ks2", "", fmt(a), fmt(b), r.statistic, r.pvalue)
r = stats.ks_2samp(a, b, method="exact")
out("Kolmogorov-Smirnov two-sample, exact", "ks2", "", fmt(a), fmt(b), r.statistic, r.pvalue)

for alt in ["two-sided", "less", "greater"]:
    r = stats.mannwhitneyu(a, b, alternative=alt, method="asymptotic", use_continuity=True)
    out(f"Mann-Whitney U, {alt}", "mw", alt, fmt(a), fmt(b), r.statistic, r.pvalue)
r = stats.mannwhitneyu(ties_x[:12], ties_y[12:], method="asymptotic", use_continuity=True)
out("Mann-Whitney U with ties", "mw", "two-sided", fmt(ties_x[:12]), fmt(ties_y[12:]), r.statistic, r.pvalue)

x = np.round(rng.normal(0, 1, 25), 3)
y = np.round(0.4 * x + rng.normal(0, 1, 25), 3)
r = stats.pearsonr(x, y)
out("Pearson r", "pearson", "", fmt(x), fmt(y), r.statistic, r.pvalue)
r = stats.spearmanr(x, y)
out("Spearman rho", "spearman", "", fmt(x), fmt(y), r.statistic, r.pvalue)
r = stats.spearmanr(ties_x, ties_y)
out("Spearman rho with ties", "spearman", "", fmt(ties_x), fmt(ties_y), r.statistic, r.pvalue)
r = stats.kendalltau(x, y, method="asymptotic")
out("Kendall tau_b", "kendall", "", fmt(x), fmt(y), r.statistic, r.pvalue)
r = stats.kendalltau(ties_x, ties_y, method="asymptotic")
out("Kendall tau_b with ties", "kendall", "", fmt(ties_x), fmt(ties_y), r.statistic, r.pvalue)

# Descriptive statistics and the kernel density estimate (statistic column only).
out("variance (ddof 1)", "var", "", fmt(skewed), "", np.var(skewed, ddof=1), float("nan"))
out("skewness", "skew", "", fmt(skewed), "", stats.skew(skewed), float("nan"))
out("excess kurtosis", "kurt", "", fmt(skewed), "", stats.kurtosis(skewed), float("nan"))
out("quantile 0.1 (type 7)", "quantile", "0.1", fmt(skewed), "", np.quantile(skewed, 0.1), float("nan"))
out("median absolute deviation (normal scale)", "mad", "", fmt(skewed), "", stats.median_abs_deviation(skewed, scale="normal"), float("nan"))
kde = stats.gaussian_kde(skewed, bw_method="scott")
for point in [0.0, 0.5, 1.5, 3.0]:
    out(f"Gaussian KDE (Scott) at {point}", "kde", repr(point), fmt(skewed), "", kde(point)[0], float("nan"))
