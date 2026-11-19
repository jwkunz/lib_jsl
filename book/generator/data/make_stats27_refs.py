"""Reference results for release 2.7.0: the Rice and multivariate distributions, the
Cramér-von Mises and Brunner-Munzel tests, the studentized range and Tukey's HSD, odds ratios,
relative risk and association, from SciPy 1.17.1.

    python book/generator/data/make_stats27_refs.py > book/generator/data/stats27_reference.txt
"""

import numpy as np
from scipy import stats
from scipy.stats import contingency


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


# --- Rice ---
xs = [0.1, 0.8, 1.5, 2.6, 3.9, 6.0]
for b, scale in ((0.0, 1.3), (1.7, 0.8), (6.0, 0.5)):
    r = stats.rice(b, scale=scale)
    out(f"rice {b:g} {scale:g}", r.pdf(xs), r.cdf(xs), r.sf(xs), r.ppf([0.01, 0.5, 0.99]), r.stats(moments="mvsk"))
r = stats.rice(40.0, scale=0.5)
out("rice tails", r.cdf([18.0, 19.0]), r.sf([21.5, 22.0]))
# Far-tail survival functions by direct integration of the density (SciPy's sf loses digits).
import math
from scipy import integrate, special
tail = []
for b, scale in ((0.0, 1.3), (1.7, 0.8), (6.0, 0.5)):
    nu = b * scale
    dens = lambda t: t / scale**2 * math.exp(-(t - nu) ** 2 / (2 * scale * scale)) * special.i0e(t * nu / scale**2)
    tail.append(integrate.quad(dens, 6.0, 6.0 + 40 * scale, epsabs=1e-30, epsrel=1e-13, limit=200)[0])
out("rice tail quad", tail, [stats.rice(b, scale=s).sf(6.0) for b, s in ((0.0, 1.3), (1.7, 0.8), (6.0, 0.5))])
sample = stats.rice(2.0, scale=1.5).ppf((np.arange(60) + 0.5) / 60)
b, loc, scale = stats.rice.fit(sample, floc=0)
out("rice fit", [b * scale, scale], [stats.rice.logpdf(sample, b, scale=scale).sum()])

# --- Multivariate ---
mean = [1.0, -0.5, 2.0]
cov = [[2.0, 0.3, -0.4], [0.3, 1.5, 0.2], [-0.4, 0.2, 1.0]]
pts = [[0.0, 0.0, 0.0], [1.5, -1.0, 2.5], [3.0, 2.0, -1.0]]
mvn = stats.multivariate_normal(mean, cov)
out("mvn", mvn.logpdf(pts), [mvn.entropy()])
mvt = stats.multivariate_t(mean, cov, df=4.5)
out("mvt", mvt.logpdf(pts), [mvt.entropy()])
alpha = [1.5, 2.0, 0.7, 3.2]
d = stats.dirichlet(alpha)
x = [0.2, 0.3, 0.1, 0.4]
out("dirichlet", [d.logpdf(x)], d.mean(), d.var(), [d.entropy()])
S = [[2.0, 0.5], [0.5, 1.0]]
X = [[3.0, 1.0], [1.0, 2.5]]
w = stats.wishart(df=5.5, scale=S)
out("wishart", [w.logpdf(X)], w.var(), [w.entropy()])
iw = stats.invwishart(df=6.0, scale=S)
out("invwishart", [iw.logpdf(X)], iw.mean(), iw.mode())
m = stats.multinomial(10, [0.2, 0.5, 0.3])
out("multinomial", [m.logpmf([2, 5, 3]), m.logpmf([0, 10, 0])], m.cov())

# --- Cramér-von Mises ---
x10 = stats.norm.ppf((np.arange(10) + 0.3) / 10) * 1.1 + 0.2
x50 = np.sin(np.arange(50) * 1.7) * 1.5 + 0.1
for name, data in (("10", x10), ("50", x50)):
    res = stats.cramervonmises(data, "norm")
    out(f"cvm {name}", [res.statistic, res.pvalue])
a = [1.2, 3.4, 0.5, 2.2, 2.2, 4.1, 0.9]
b_ = [2.5, 3.9, 4.4, 1.8, 5.0, 3.3, 2.2, 6.1, 4.7]
res = stats.cramervonmises_2samp(a, b_)
out("cvm2 exact", [res.statistic, res.pvalue])
res = stats.cramervonmises_2samp(a, b_, method="asymptotic")
out("cvm2 asym small", [res.statistic, res.pvalue])
big_a = np.sin(np.arange(25) * 0.9) * 2
big_b = np.cos(np.arange(30) * 1.3) * 2 + 0.6
res = stats.cramervonmises_2samp(big_a, big_b)
out("cvm2 asym", [res.statistic, res.pvalue])

# --- Brunner-Munzel ---
x1 = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1]
y1 = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
for alt in ("two-sided", "less", "greater"):
    res = stats.brunnermunzel(x1, y1, alternative=alt)
    out(f"bm {alt}", [res.statistic, res.pvalue])
res = stats.brunnermunzel(x1, y1, distribution="normal")
out("bm normal", [res.statistic, res.pvalue])

# --- Studentized range and Tukey HSD ---
cases = [(3.5, 3, 10), (2.0, 5, 20), (4.8, 4, 60), (1.2, 2, 5), (5.5, 6, np.inf), (3.0, 10, 200)]
out("srange cdf", [stats.studentized_range.cdf(q, k, df) for q, k, df in cases])
out("srange pdf", [stats.studentized_range.pdf(q, k, df) for q, k, df in cases])
out("srange ppf", [stats.studentized_range.ppf(0.95, k, df) for _, k, df in cases])
g1 = [24.5, 23.5, 26.4, 27.1, 29.9]
g2 = [28.4, 34.2, 29.5, 32.2, 30.1, 33.0]
g3 = [26.1, 28.3, 24.3, 26.2, 27.8]
g4 = [31.0, 29.8, 33.5, 32.6]
for eq in (True, False):
    res = stats.tukey_hsd(g1, g2, g3, g4, equal_var=eq)
    ci = res.confidence_interval(0.95)
    out(f"tukey {eq}", res.statistic, res.pvalue, ci.low, ci.high)

# --- Contingency ---
tables = {"a": [[12, 5], [3, 15]], "b": [[7, 2], [9, 30]], "zero": [[0, 6], [5, 9]]}
for name, t in tables.items():
    for kind in ("conditional", "sample"):
        r = stats.contingency.odds_ratio(t, kind=kind)
        vals = [r.statistic]
        for alt in ("two-sided", "less", "greater"):
            ci = r.confidence_interval(0.95, alternative=alt)
            vals += [ci.low, ci.high]
        out(f"or {name} {kind}", np.nan_to_num(vals, nan=-1.0, posinf=1e300))
rr = stats.contingency.relative_risk(27, 122, 44, 487)
ci = rr.confidence_interval(0.9)
out("rr", [rr.relative_risk, ci.low, ci.high])
t34 = [[10, 4, 6, 2], [3, 12, 5, 8], [6, 2, 9, 11]]
out("association", [contingency.association(t34, method=m) for m in ("cramer", "tschuprow", "pearson")])
