"""Reference values for the random numbers chapter, from SciPy (the book used SciPy 1.17.1).

    python book/generator/data/make_random_refs.py > book/generator/data/random_reference.txt

- `sobol` and `halton`: unscrambled points from `scipy.stats.qmc` (row-major).
- `bins <name>`: for continuous distributions, the 19 inner edges of 20 equally likely bins
  (`ppf` at 0.05, 0.10, ..., 0.95); the generator counts the library's samples in each bin.
- `pmf <name>`: for discrete distributions, the first value k0 and then P(k0), P(k0 + 1), ...,
  covering all but 10^-12 of the probability.
"""

import numpy as np
from scipy import stats
from scipy.stats import qmc


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


out("sobol", qmc.Sobol(10, scramble=False).random(256))
out("halton", qmc.Halton(10, scramble=False).random(257)[1:])

continuous = {
    "uniform U(-1, 3)": stats.uniform(-1, 4),
    "exponential, rate 2": stats.expon(scale=0.5),
    "Weibull, shape 1.5, scale 2": stats.weibull_min(1.5, scale=2),
    "normal N(1, 4)": stats.norm(1, 2),
    "log-normal, mu 0, sigma 0.5": stats.lognorm(0.5),
    "chi-square, k = 5": stats.chi2(5),
    "Student's t, nu = 5": stats.t(5),
    "gamma, shape 2.5, scale 2": stats.gamma(2.5, scale=2),
    "gamma, shape 0.3": stats.gamma(0.3),
    "beta (2, 5)": stats.beta(2, 5),
    "Cauchy": stats.cauchy(),
}
for name, d in continuous.items():
    out(f"bins {name}", d.ppf(np.arange(1, 20) / 20))

discrete = {
    "Bernoulli, p = 0.3": stats.bernoulli(0.3),
    "geometric, p = 0.2 (failures)": stats.geom(0.2, loc=-1),
    "Poisson, lambda = 4": stats.poisson(4),
    "Poisson, lambda = 300": stats.poisson(300),
    "binomial, n = 20, p = 0.3": stats.binom(20, 0.3),
    "binomial, n = 100000, p = 0.4": stats.binom(100000, 0.4),
}
for name, d in discrete.items():
    lo, hi = int(d.ppf(1e-12)), int(d.isf(1e-12))
    k = np.arange(max(lo, 0), hi + 1)
    out(f"pmf {name}", [k[0]], d.pmf(k))
