"""Reference results for release 1.16.0 (FFT helpers, log-domain and inverse special functions,
robust regression, circular statistics and distances), from SciPy 1.17.1 and NumPy 2.

    python book/generator/data/make_utilities_refs.py > book/generator/data/utilities_reference.txt

Inputs built by formula are rebuilt by the generator; random inputs are written out first as
`input <name>|values`. Results are `case|values`.
"""

import numpy as np
from scipy import fft, special, stats
from scipy.spatial import distance

rng = np.random.default_rng(1160)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


# --- FFT helpers ---
for n in (7, 8):
    out(f"fftfreq {n}", fft.fftfreq(n, 0.1))
    out(f"rfftfreq {n}", fft.rfftfreq(n, 0.1))
targets = [7, 13, 97, 1000, 1021, 4097, 65537, 999983]
out("fast targets", targets)
out("next_fast_len real", [fft.next_fast_len(t, real=True) for t in targets])
out("next_fast_len complex", [fft.next_fast_len(t, real=False) for t in targets])
for shape in ((4, 5, 6), (3, 4, 7)):
    i, j, k = np.indices(shape)
    x = np.sin(0.3 * (i * 35 + j * 7 + k)) + 0.1 * i
    r = fft.rfftn(x)
    name = "x".join(map(str, shape))
    out(f"rfftn {name}", r.real, r.imag)
i, j = np.indices((5, 6))
x2 = np.cos(0.7 * (i * 6 + j)) + 0.1 * i
for kind in (1, 2, 3, 4):
    for norm in ("backward", "ortho"):
        out(f"dctn {kind} {norm}", fft.dctn(x2, type=kind, norm=norm))
        out(f"dstn {kind} {norm}", fft.dstn(x2, type=kind, norm=norm))

# --- Special functions ---
out("logsumexp", [special.logsumexp([1000.0, 999.0, -1000.0]), special.logsumexp([-1000.0, -1001.0]), special.logsumexp(np.linspace(-3, 3, 13))])
ls, sg = special.logsumexp([0.0, 1.0, 2.0], b=[1.0, -3.0, 1.0], return_sign=True)
out("logsumexp signed", [ls, sg])
grid = np.linspace(-50, 10, 61)
out("log_ndtr", special.log_ndtr(grid))
xs = np.array([-800.0, -40.0, -5.0, -0.3, 0.0, 0.7, 5.0, 40.0, 800.0])
out("expit", special.expit(xs))
out("log_expit", special.log_expit(xs))
ps = np.array([1e-12, 0.01, 0.3, 0.5, 0.9, 1 - 1e-9])
out("logit", special.logit(ps))
px = np.array([0.1, 0.5, 1.0, 3.7, 10.0, 55.5, -2.5, -0.3])
for n in (1, 2, 3, 4):
    out(f"polygamma {n}", special.polygamma(n, px))
out("multigammaln", [special.multigammaln(a, d) for a, d in ((3.5, 1), (2.0, 2), (7.25, 3), (20.0, 10))])
out("bernoulli", special.bernoulli(30))
out("euler", special.euler(30))


def exact_bernoulli(n):
    """Akiyama-Tanigawa, in exact rationals, with B1 = -1/2."""
    from fractions import Fraction
    a, b = [Fraction(0)] * (n + 1), []
    for m in range(n + 1):
        a[m] = Fraction(1, m + 1)
        for j in range(m, 0, -1):
            a[j - 1] = j * (a[j - 1] - a[j])
        b.append(a[0])
    b[1] = -b[1]
    return [float(v) for v in b]


def exact_euler(n):
    """E_n = -sum_{k<n, k even} C(n, k) E_k, in integers."""
    from math import comb
    e = [0] * (n + 1)
    e[0] = 1
    for m in range(2, n + 1, 2):
        e[m] = -sum(comb(m, k) * e[k] for k in range(0, m, 2))
    return [float(v) for v in e]


out("bernoulli exact", exact_bernoulli(30))
out("euler exact", exact_euler(30))
avals = [0.1, 0.5, 1.0, 2.5, 10.0, 100.0]
pvals = [1e-10, 0.01, 0.3, 0.5, 0.9, 0.999999]
out("gammaincinv", [special.gammaincinv(a, p) for a in avals for p in pvals])
out("gammainccinv", [special.gammainccinv(a, p) for a in avals for p in pvals])
abvals = [(0.2, 0.3), (0.5, 0.5), (1.0, 3.0), (2.0, 5.0), (30.0, 0.7), (100.0, 200.0)]
out("betaincinv", [special.betaincinv(a, b, p) for a, b in abvals for p in pvals])
bx = [0.5, 1.0, 3.2, 10.0]
bl = [-1.5, 0.0, 0.5, 2.0]
out("boxcox", [special.boxcox(x, l) for x in bx for l in bl])
out("boxcox1p", [special.boxcox1p(x, l) for x in (1e-10, 0.3, 5.0) for l in bl])

# --- Statistics ---
xr = np.arange(40) / 3.0
yr = 2.0 * xr + 1.0 + rng.normal(0, 0.8, 40)
yr[5] += 25.0
yr[22] -= 30.0
yr[31] += 18.0
out("input yr", yr)
lr = stats.linregress(xr, yr)
out("linregress", [lr.slope, lr.intercept, lr.rvalue, lr.pvalue, lr.stderr, lr.intercept_stderr])
for alpha in (0.95, 0.9):
    t = stats.theilslopes(yr, xr, alpha=alpha)
    out(f"theilslopes {alpha}", [t.slope, t.intercept, t.low_slope, t.high_slope])
s = stats.siegelslopes(yr, xr)
out("siegelslopes", [s.slope, s.intercept])
# Ties in x and y for the Theil-Sen interval.
xt = np.repeat(np.arange(10.0), 3)
yt = np.round(1.5 * xt + rng.normal(0, 1.0, 30))
out("input yt", yt)
t = stats.theilslopes(yt, xt)
out("theilslopes ties", [t.slope, t.intercept, t.low_slope, t.high_slope])

sk = rng.standard_t(3, 101)
out("input skewed", sk)
out("trim_mean", [stats.trim_mean(sk, p) for p in (0.0, 0.1, 0.25, 0.4)])
out("zscore", stats.zscore(sk[:10]), stats.zscore(sk[:10], ddof=1))
md = stats.mode(np.round(sk))
out("mode", [md.mode, md.count])
ang = np.mod(rng.vonmises(0.3, 2.0, 50), 2 * np.pi)
out("input angles", ang)
out("circular", [stats.circmean(ang), stats.circvar(ang), stats.circstd(ang)])
deg = np.degrees(ang)
out("circular degrees", [stats.circmean(deg, high=360), stats.circvar(deg, high=360), stats.circstd(deg, high=360)])
pos = rng.lognormal(1.0, 0.6, 80) + 0.5
out("input positive", pos)
bc, lam = stats.boxcox(pos)
out("boxcox_normmax", [lam, stats.boxcox_llf(lam, pos)])
out("boxcox data", bc)

pts = np.round(rng.normal(size=(7, 4)), 4)
pts[2, 1] = 0.0
pts[2, 3] = 0.0
pts[5, 0] = 0.0
out("input points", pts)
vi = np.linalg.inv(np.cov(pts.T))
out("input vi", vi)
var = pts.var(axis=0, ddof=1)
metrics = [("euclidean", {}), ("sqeuclidean", {}), ("cityblock", {}), ("chebyshev", {}), ("minkowski", {"p": 3}),
           ("seuclidean", {"V": var}), ("mahalanobis", {"VI": vi}), ("cosine", {}), ("correlation", {}),
           ("canberra", {}), ("braycurtis", {}), ("hamming", {}), ("jaccard", {})]
for name, kw in metrics:
    out(f"pdist {name}", distance.pdist(pts, name, **kw))
out("cdist euclidean", distance.cdist(pts[:3], pts[3:], "euclidean"))
