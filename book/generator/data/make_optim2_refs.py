"""Reference results for assignment, trust-region and global optimisation (release 1.19.0), from
SciPy 1.17.1.

    python book/generator/data/make_optim2_refs.py > book/generator/data/optim2_reference.txt

Random cost matrices are written as `input <name>|rows;cols;values`; results as `case|values`.
"""

import numpy as np
from scipy import optimize as so

rng = np.random.default_rng(1190)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def matrix(name, a):
    print(f"input {name}|{a.shape[0]};{a.shape[1]};" + ",".join(repr(float(v)) for v in a.ravel()))


# Assignment.
cases = {
    "square": rng.random((60, 60)),
    "wide": rng.random((40, 70)),
    "tall": rng.random((70, 40)),
    "ties": rng.integers(0, 5, (30, 30)).astype(float),
}
for name, c in cases.items():
    matrix(name, c)
    for maximize in (False, True):
        r, k = so.linear_sum_assignment(c, maximize=maximize)
        out(f"lsap {name} {int(maximize)}", r, k, [c[r, k].sum()])

# Trust-region methods.
def rosen_hess(x):
    return so.rosen_hess(x)


starts = {"r4": [-1.2, 1.0, -0.5, 0.8], "r10": [-1.2, 1.0] * 5, "near": [1.2, 1.3, 1.1, 1.0]}
for start, x0 in starts.items():
    for method in ("trust-ncg", "trust-exact", "dogleg"):
        r = so.minimize(so.rosen, x0, jac=so.rosen_der, hess=rosen_hess, method=method, options=dict(gtol=1e-8))
        out(f"tr {method} {start}", r.x, [r.fun, r.nit, r.nfev, r.njev, r.status])


def saddle(v):
    return v[0] ** 4 - 2 * v[0] ** 2 + v[1] ** 2 + 0.5 * v[0] * v[1]


def saddle_grad(v):
    return np.array([4 * v[0] ** 3 - 4 * v[0] + 0.5 * v[1], 2 * v[1] + 0.5 * v[0]])


def saddle_hess(v):
    return np.array([[12 * v[0] ** 2 - 4, 0.5], [0.5, 2.0]])


for method in ("trust-ncg", "trust-exact"):
    r = so.minimize(saddle, [0.01, 0.5], jac=saddle_grad, hess=saddle_hess, method=method, options=dict(gtol=1e-10))
    out(f"tr {method} saddle", r.x, [r.fun, r.nit, r.nfev, r.njev, r.status])

# Global optimisation on test functions.
def camel(x):
    a, b = x
    return (4 - 2.1 * a * a + a ** 4 / 3) * a * a + a * b + (-4 + 4 * b * b) * b * b


def branin(x):
    a, b = x
    return (b - 5.1 / (4 * np.pi ** 2) * a * a + 5 / np.pi * a - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(a) + 10


def rastrigin(x):
    x = np.asarray(x)
    return 10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x))


def ackley(x):
    a, b = x
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (a * a + b * b))) - np.exp(0.5 * (np.cos(2 * np.pi * a) + np.cos(2 * np.pi * b))) + np.e + 20


def eggholder(x):
    a, b = x
    return -(b + 47) * np.sin(np.sqrt(abs(a / 2 + b + 47))) - a * np.sin(np.sqrt(abs(a - (b + 47))))


def styblinski(x):
    x = np.asarray(x)
    return 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)


problems = [
    ("camel", camel, [(-3, 3), (-2, 2)], -1.031628453489877),
    ("branin", branin, [(-5, 10), (0, 15)], 0.39788735772973816),
    ("rastrigin", rastrigin, [(-5.12, 5.12)] * 4, 0.0),
    ("ackley", ackley, [(-5, 5)] * 2, 0.0),
    ("eggholder", eggholder, [(-512, 512)] * 2, -959.6406627208506),
    ("styblinski", styblinski, [(-5, 5)] * 3, -39.16616570377142 * 3),
]
for name, f, bounds, fstar in problems:
    ok = lambda v: v - fstar <= 1e-4 * max(1.0, abs(fstar))
    d = so.direct(f, bounds)
    out(f"direct {name}", [d.fun, d.nfev, d.nit, int(ok(d.fun))])
    da = [so.dual_annealing(f, bounds, rng=s) for s in range(20)]
    out(f"dual {name}", [sum(ok(r.fun) for r in da), np.median([r.nfev for r in da])])
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    bh = []
    for s in range(20):
        x0 = np.random.default_rng(s).uniform(lo, hi)
        bh.append(so.basinhopping(f, x0, niter=100, rng=s))
    out(f"basin {name}", [sum(ok(r.fun) for r in bh), np.median([r.nfev for r in bh])])
