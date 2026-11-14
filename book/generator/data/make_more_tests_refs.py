"""Reference results for the entry on more hypothesis tests, from `scipy.stats` (the book used
SciPy 1.17.1).

    python book/generator/data/make_more_tests_refs.py > book/generator/data/more_tests_reference.txt

Samples are written as `data <name>|values` (rounded to 6 digits), results as
`<test> <case>|statistic,pvalue`.
"""

import warnings
import numpy as np
from scipy import stats

warnings.simplefilter("ignore")
rng = np.random.default_rng(404)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def data(name, x):
    x = np.round(np.asarray(x, dtype=float), 6)
    out(f"data {name}", x)
    return x


normal = data("normal", rng.normal(2.0, 1.5, 60))
skewed = data("skewed", rng.gamma(2.0, 1.0, 60))
small = data("small", rng.normal(0, 1, 9))
tiny = data("tiny", rng.normal(0, 1, 4))
for name, x in (("normal", normal), ("skewed", skewed), ("small", small), ("tiny", tiny)):
    r = stats.shapiro(x)
    out(f"shapiro {name}", [r.statistic, r.pvalue])
for name, x in (("normal", normal), ("skewed", skewed)):
    r = stats.anderson(x, "norm")
    out(f"anderson {name}", [r.statistic], r.critical_values)
    r = stats.jarque_bera(x)
    out(f"jarque_bera {name}", [r.statistic, r.pvalue])
    r = stats.normaltest(x)
    out(f"normaltest {name}", [r.statistic, r.pvalue])
    r = stats.skewtest(x)
    out(f"skewtest {name}", [r.statistic, r.pvalue])
    r = stats.kurtosistest(x)
    out(f"kurtosistest {name}", [r.statistic, r.pvalue])

g1 = data("g1", rng.normal(10, 1.0, 15))
g2 = data("g2", rng.normal(10.5, 2.0, 18))
g3 = data("g3", rng.normal(11, 1.2, 12))
for test in ("levene", "bartlett", "fligner", "kruskal"):
    r = getattr(stats, test)(g1, g2, g3)
    out(f"{test} groups", [r.statistic, r.pvalue])
blocks = data("blocks", np.round(rng.normal(0, 1, (10, 4)) + np.array([0, 0.3, 0.8, 0.4]), 1))
r = stats.friedmanchisquare(*blocks.T)
out("friedman blocks", [r.statistic, r.pvalue])

d = data("diffs", rng.normal(0.4, 1.0, 20))
for alt in ("two-sided", "less", "greater"):
    r = stats.wilcoxon(d, alternative=alt)
    out(f"wilcoxon exact {alt}", [r.statistic, r.pvalue])
dt = data("diffs_ties", np.round(rng.normal(0.4, 1.0, 12), 1))
r = stats.wilcoxon(dt)
out("wilcoxon ties", [r.statistic, r.pvalue])
dl = data("diffs_large_ties", np.round(rng.normal(0.3, 1.0, 40), 1))
r = stats.wilcoxon(dl)
out("wilcoxon asymptotic", [r.statistic, r.pvalue])

for name, table in (("a", [[8, 2], [1, 5]]), ("b", [[12, 5], [9, 14]]), ("c", [[0, 7], [6, 3]])):
    for alt in ("two-sided", "less", "greater"):
        r = stats.fisher_exact(table, alternative=alt)
        out(f"fisher {name} {alt}", [r.statistic, r.pvalue])
for k, n, p in ((7, 20, 0.5), (3, 40, 0.2), (61, 100, 0.5)):
    r = stats.binomtest(k, n, p)
    out(f"binomtest {k} {n} {p}", [r.statistic, r.pvalue])

x = data("perm_x", rng.normal(0, 1, 7))
y = data("perm_y", rng.normal(0.8, 1, 8))
diff = lambda a, b, axis: np.mean(a, axis=axis) - np.mean(b, axis=axis)
for alt in ("two-sided", "less", "greater"):
    r = stats.permutation_test((x, y), diff, vectorized=True, n_resamples=np.inf, alternative=alt)
    out(f"permutation {alt}", [r.statistic, r.pvalue])

ps = data("pvalues", [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
out("fdr bh", stats.false_discovery_control(ps, method="bh"))
out("fdr by", stats.false_discovery_control(ps, method="by"))
