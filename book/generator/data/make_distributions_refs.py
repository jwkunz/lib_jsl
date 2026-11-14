"""Reference results for the entry on probability distributions, from `scipy.stats` (the book used
SciPy 1.17.1).

    python book/generator/data/make_distributions_refs.py > book/generator/data/distributions_reference.txt

For each distribution, in the library's parameterisation: `eval <name>|params;x;pdf;cdf;sf;q;ppf;
moments` (mean, variance, skewness, excess kurtosis). For fitting: `sample <name>|data` (rounded to
8 digits) and `fit <name>|params`, the maximum-likelihood estimate, polished by a tight
Nelder-Mead minimisation of SciPy's negative log-likelihood where SciPy has no closed form.
"""

import numpy as np
from scipy import stats, optimize

rng = np.random.default_rng(2026)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


# name: (library parameters, frozen SciPy distribution, evaluation points)
C = {
    "normal": ((1.0, 2.0), stats.norm(1.0, 2.0), [-3.0, 0.0, 1.5, 4.0, 9.0]),
    "uniform": ((-1.0, 3.0), stats.uniform(-1.0, 4.0), [-0.5, 0.0, 1.5, 2.9]),
    "exponential": ((0.7,), stats.expon(scale=1 / 0.7), [0.1, 1.0, 3.0, 12.0]),
    "gamma": ((2.5, 1.5), stats.gamma(2.5, scale=1.5), [0.2, 1.0, 3.0, 9.0, 20.0]),
    "beta": ((2.0, 3.5), stats.beta(2.0, 3.5), [0.05, 0.3, 0.6, 0.95]),
    "chi2": ((7.5,), stats.chi2(7.5), [0.5, 3.0, 7.0, 20.0]),
    "t": ((4.5, 1.0, 2.0), stats.t(4.5, 1.0, 2.0), [-5.0, 0.0, 1.0, 6.0, 20.0]),
    "f": ((5.0, 12.0), stats.f(5.0, 12.0), [0.1, 0.8, 2.0, 5.0]),
    "lognormal": ((0.3, 0.6), stats.lognorm(0.6, scale=np.exp(0.3)), [0.2, 1.0, 2.0, 5.0]),
    "weibull": ((1.7, 2.0), stats.weibull_min(1.7, scale=2.0), [0.1, 1.0, 2.5, 6.0]),
    "cauchy": ((0.5, 1.5), stats.cauchy(0.5, 1.5), [-10.0, 0.0, 1.0, 30.0]),
    "laplace": ((0.5, 1.2), stats.laplace(0.5, 1.2), [-3.0, 0.0, 1.0, 4.0]),
    "logistic": ((0.5, 1.2), stats.logistic(0.5, 1.2), [-3.0, 0.0, 1.0, 4.0]),
    "pareto": ((2.8, 1.5), stats.pareto(2.8, scale=1.5), [1.6, 2.0, 5.0, 30.0]),
    "rayleigh": ((1.3,), stats.rayleigh(scale=1.3), [0.2, 1.0, 2.0, 4.0]),
    "gumbel": ((1.0, 0.8), stats.gumbel_r(1.0, 0.8), [-1.0, 0.5, 2.0, 5.0]),
    "gev": ((0.2, 1.0, 0.8), stats.genextreme(-0.2, 1.0, 0.8), [-1.0, 0.5, 2.0, 5.0]),
    "gev_bounded": ((-0.3, 1.0, 0.8), stats.genextreme(0.3, 1.0, 0.8), [-1.0, 0.5, 2.0, 3.0]),
    "truncnorm": ((0.5, 1.2, -1.0, 2.5), stats.truncnorm((-1.0 - 0.5) / 1.2, (2.5 - 0.5) / 1.2, 0.5, 1.2), [-0.9, 0.0, 1.0, 2.4]),
    "vonmises": ((0.4, 2.5), stats.vonmises(2.5, loc=0.4), [-2.5, -0.5, 0.4, 3.0]),
    "skewnorm": ((3.0, 0.5, 1.5), stats.skewnorm(3.0, 0.5, 1.5), [-2.0, 0.5, 1.5, 5.0]),
}
q = [0.001, 0.1, 0.5, 0.9, 0.999]
for name, (params, d, xs) in C.items():
    m = d.stats(moments="mvsk")
    out(f"eval {name}", params, xs, d.pdf(xs), d.cdf(xs), d.sf(xs), q, d.ppf(q), [float(v) for v in m])

D = {
    "binomial": ((20, 0.3), stats.binom(20, 0.3), [0, 3, 6, 12, 20]),
    "poisson": ((4.2,), stats.poisson(4.2), [0, 2, 4, 9, 15]),
    "geometric": ((0.25,), stats.geom(0.25), [1, 2, 5, 12]),
    "nbinom": ((3.5, 0.4), stats.nbinom(3.5, 0.4), [0, 2, 5, 12, 25]),
    "hypergeom": ((50, 12, 15), stats.hypergeom(50, 12, 15), [0, 2, 4, 7, 12]),
}
for name, (params, d, ks) in D.items():
    m = d.stats(moments="mvsk")
    out(f"discrete {name}", params, ks, d.pmf(ks), d.cdf(ks), d.sf(ks), q, d.ppf(q), [float(v) for v in m])


def polish(dist, data, p0, fixed):
    """Tight Nelder-Mead on SciPy's negative log-likelihood over the free parameters."""
    free = [i for i in range(len(p0)) if i not in fixed]
    def nll(v):
        p = list(p0)
        for i, val in zip(free, v):
            p[i] = val
        r = dist.nnlf(p, data)
        return r if np.isfinite(r) else 1e300
    res = optimize.minimize(nll, [p0[i] for i in free], method="Nelder-Mead",
                            options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 40000, "maxfev": 80000})
    p = list(p0)
    for i, val in zip(free, res.x):
        p[i] = val
    return p


# Samples and fits. SciPy parameter order is (shapes..., loc, scale).
fits = {
    "normal": (stats.norm(1.0, 2.0), lambda x: stats.norm.fit(x)),
    "exponential": (stats.expon(scale=1 / 0.7), lambda x: [1 / stats.expon.fit(x, floc=0)[1]]),
    "gamma": (stats.gamma(2.5, scale=1.5), lambda x: (lambda p: [p[0], p[2]])(polish(stats.gamma, x, stats.gamma.fit(x, floc=0), {1}))),
    "beta": (stats.beta(2.0, 3.5), lambda x: polish(stats.beta, x, stats.beta.fit(x, floc=0, fscale=1), {2, 3})[:2]),
    "chi2": (stats.chi2(7.5), lambda x: polish(stats.chi2, x, stats.chi2.fit(x, floc=0, fscale=1), {1, 2})[:1]),
    "t": (stats.t(4.5, 1.0, 2.0), lambda x: polish(stats.t, x, stats.t.fit(x), set())),
    "f": (stats.f(5.0, 12.0), lambda x: polish(stats.f, x, stats.f.fit(x, floc=0, fscale=1), {2, 3})[:2]),
    "lognormal": (stats.lognorm(0.6, scale=np.exp(0.3)), lambda x: (lambda p: [np.log(p[2]), p[0]])(stats.lognorm.fit(x, floc=0))),
    "weibull": (stats.weibull_min(1.7, scale=2.0), lambda x: (lambda p: [p[0], p[2]])(polish(stats.weibull_min, x, stats.weibull_min.fit(x, floc=0), {1}))),
    "cauchy": (stats.cauchy(0.5, 1.5), lambda x: polish(stats.cauchy, x, stats.cauchy.fit(x), set())),
    "laplace": (stats.laplace(0.5, 1.2), lambda x: stats.laplace.fit(x)),
    "logistic": (stats.logistic(0.5, 1.2), lambda x: polish(stats.logistic, x, stats.logistic.fit(x), set())),
    "pareto": (stats.pareto(2.8, scale=1.5), lambda x: (lambda p: [p[0], p[2]])(stats.pareto.fit(x, floc=0))),
    "rayleigh": (stats.rayleigh(scale=1.3), lambda x: [stats.rayleigh.fit(x, floc=0)[1]]),
    "gumbel": (stats.gumbel_r(1.0, 0.8), lambda x: polish(stats.gumbel_r, x, stats.gumbel_r.fit(x), set())),
    "gev": (stats.genextreme(-0.2, 1.0, 0.8), lambda x: (lambda p: [-p[0], p[1], p[2]])(polish(stats.genextreme, x, stats.genextreme.fit(x), set()))),
    "vonmises": (stats.vonmises(2.5, loc=0.4), lambda x: (lambda p: [p[1], p[0]])(polish(stats.vonmises, x, stats.vonmises.fit(x, fscale=1), {2}))),
    "skewnorm": (stats.skewnorm(3.0, 0.5, 1.5), lambda x: polish(stats.skewnorm, x, stats.skewnorm.fit(x), set())),
}
for name, (d, fit) in fits.items():
    x = np.round(d.rvs(size=400, random_state=rng), 8)
    out(f"sample {name}", x)
    out(f"fit {name}", fit(x))
x = stats.nbinom(3.5, 0.4).rvs(size=400, random_state=rng)
out("sample nbinom", x)
r = optimize.minimize_scalar(lambda r: stats.nbinom.nnlf([r, r / (r + x.mean()), 0], x), bounds=(0.1, 50), method="bounded", options={"xatol": 1e-12})
out("fit nbinom", [r.x, r.x / (r.x + x.mean())])
