"""SciPy references for release 2.18.0: shgo (Sobol sampling) and quadratic_assignment (FAQ and
2-opt from the identity). Writes glob218_reference.txt."""
import numpy as np
from scipy.optimize import shgo, quadratic_assignment

out = []
def put(key, *lists):
    out.append(key + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))

def himmelblau(x): return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
def eggholder(x): return -(x[1] + 47) * np.sin(np.sqrt(abs(x[0] / 2 + (x[1] + 47)))) - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))
def ackley(x): return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20
def rosen(x): return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
cases = [
    ("himmelblau", himmelblau, [(-5, 5)] * 2, 128, 1, []),
    ("eggholder", eggholder, [(-512, 512)] * 2, 256, 3, []),
    ("ackley", ackley, [(-5, 5)] * 2, 128, 1, []),
    ("rosen4", rosen, [(-2, 2)] * 4, 128, 1, []),
    # SciPy's documented constrained example (g >= 0 in SciPy's convention).
    ("constrained", lambda x: (x[0] - 1)**2 + (x[1] - 2)**2, [(-3, 3)] * 2, 128, 1,
     [{"type": "ineq", "fun": lambda x: 1.5 - x[0]**2 - x[1]**2}, {"type": "ineq", "fun": lambda x: x[0] + x[1]}]),
]
for name, f, b, n, it, cons in cases:
    r = shgo(f, b, n=n, iters=it, sampling_method="sobol", constraints=cons or None, options={"ftol": 1e-12})
    xl = np.atleast_2d(r.xl)
    put(f"shgo {name}", r.x, [r.fun, r.nfev, r.nlfev], xl, r.funl)

rng = np.random.default_rng(218)
for n in [10, 20, 40]:
    a = rng.integers(0, 10, size=(n, n)).astype(float)
    b = rng.integers(0, 10, size=(n, n)).astype(float)
    np.fill_diagonal(a, 0); np.fill_diagonal(b, 0)
    for maximize in [False, True]:
        f = quadratic_assignment(a, b, method="faq", options={"maximize": maximize})
        t = quadratic_assignment(a, b, method="2opt", options={"maximize": maximize, "partial_guess": np.c_[np.arange(n), np.arange(n)]})
        put(f"qap {n} {maximize}", a, b, f.col_ind, [f.fun, f.nit], t.col_ind, [t.fun, t.nit])
open("data/glob218_reference.txt", "w").write("\n".join(out) + "\n")
