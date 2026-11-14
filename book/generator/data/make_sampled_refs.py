"""Reference results for the entries on integrating sampled data and on adaptive derivatives,
from `scipy.integrate` and `scipy.differentiate` (the book used SciPy 1.17.1).

    python book/generator/data/make_sampled_refs.py > book/generator/data/sampled_reference.txt
"""

import numpy as np
from scipy import integrate, differentiate

rng = np.random.default_rng(77)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


xu = np.linspace(0.0, 3.0, 25)
xn = np.round(np.sort(np.r_[0.0, rng.uniform(0, 3, 22), 3.0]), 6)
for name, x in (("uniform", xu), ("nonuniform", xn)):
    y = np.round(np.exp(-x) * np.sin(2 * x) + 0.1 * x ** 2, 6)
    out(f"data x {name}", x)
    out(f"data y {name}", y)
    out(f"trapezoid {name}", [integrate.trapezoid(y, x=x)])
    out(f"simpson {name}", [integrate.simpson(y, x=x)])
    out(f"simpson even {name}", [integrate.simpson(y[:-1], x=x[:-1])])
    out(f"cumulative_trapezoid {name}", integrate.cumulative_trapezoid(y, x=x, initial=0))
    out(f"cumulative_simpson {name}", integrate.cumulative_simpson(y, x=x, initial=0))
xr = np.linspace(0, 2, 33)
yr = np.round(np.exp(-xr) * np.sin(2 * xr) + 0.1 * xr ** 2, 6)
out("data y romb", yr)
out("romb", [integrate.romb(yr, dx=xr[1] - xr[0])])

# Derivatives: value and error estimate for several functions.
funcs = {
    "exp": (np.exp, np.exp),
    "sin3x": (lambda t: np.sin(3 * t), lambda t: 3 * np.cos(3 * t)),
    "log1p": (np.log1p, lambda t: 1 / (1 + t)),
    "atan10": (lambda t: np.arctan(10 * t), lambda t: 10 / (1 + 100 * t ** 2)),
}
pts = np.array([0.3, 1.0, 2.7])
for name, (f, df) in funcs.items():
    r = differentiate.derivative(f, pts)
    out(f"derivative {name}", r.df, r.error, r.nfev, df(pts))
