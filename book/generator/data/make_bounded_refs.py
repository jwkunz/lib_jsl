"""Reference results for the entry on bound-constrained optimisation, from `scipy.optimize`
(the book used SciPy 1.17.1).

    python book/generator/data/make_bounded_refs.py > book/generator/data/bounded_reference.txt

Inputs are written as `data <name>|values` (matrices as `rows;cols;values`), results as
`case|values` (`;` between lists).
"""

import numpy as np
from scipy.optimize import minimize, least_squares, lsq_linear, nnls

rng = np.random.default_rng(71)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def matrix(name, a):
    a = np.round(np.atleast_2d(a), 6)
    print(f"data {name}|{a.shape[0]};{a.shape[1]};" + ",".join(repr(float(v)) for v in a.ravel()))
    return a


# L-BFGS-B on Rosenbrock (n = 10) with bounds that cut off the minimum, and on a convex quadratic.
def rosen(x):
    f = np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    g = np.zeros_like(x)
    g[:-1] = -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
    g[1:] += 200 * (x[1:] - x[:-1] ** 2)
    return f, g


x0 = np.array([-1.2, 1.0] * 5)
lb = np.full(10, -2.0)
ub = np.array([1.5, 0.6, 1.5, 0.6, 1.5, 1.5, 0.7, 1.5, 1.5, 1.5])
out("data rosen ub", ub)
r = minimize(rosen, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lb, ub)))
out("lbfgsb rosen", r.x, [r.fun, r.nit, r.nfev])
M = rng.normal(size=(30, 30))
H = matrix("H", M.T @ M / 30 + 0.1 * np.eye(30))
c = matrix("c", rng.normal(size=(1, 30)))[0]
quad = lambda x: (0.5 * x @ H @ x + c @ x, H @ x + c)
r = minimize(quad, np.zeros(30), jac=True, method="L-BFGS-B", bounds=[(-0.5, 0.5)] * 30, options={"ftol": 1e-15, "gtol": 1e-10})
out("lbfgsb quadratic", r.x, [r.fun, r.nit, r.nfev])

# Nonlinear least squares: y = a e^{-k t} + c with outliers, k bounded.
t = np.linspace(0, 10, 60)
y = 3.0 * np.exp(-0.6 * t) + 0.5 + 0.05 * rng.normal(size=t.size)
y[[7, 23, 41]] += np.array([2.0, -1.5, 2.5])
y = np.round(y, 6)
out("data t", t)
out("data y", y)
fun = lambda p: p[0] * np.exp(-p[1] * t) + p[2] - y
jac = lambda p: np.c_[np.exp(-p[1] * t), -p[0] * t * np.exp(-p[1] * t), np.ones_like(t)]
for loss in ("linear", "soft_l1", "huber", "cauchy", "arctan"):
    r = least_squares(fun, [1.0, 0.1, 0.0], jac=jac, bounds=([0, 0, -1], [10, 2, 1]), loss=loss, f_scale=0.2)
    out(f"trf {loss}", r.x, [r.cost, r.nfev, r.njev, r.optimality])
r = least_squares(fun, [1.0, 0.1, 0.0], jac=jac, bounds=([0, 0, -1], [10, 0.4, 1]))
out("trf bounded", r.x, [r.cost, r.nfev, r.njev, r.optimality], r.active_mask)

# Linear problems.
A = matrix("A", rng.normal(size=(40, 12)))
b = matrix("b", rng.normal(size=(1, 40)))[0]
r = lsq_linear(A, b, bounds=(-0.15, 0.15), method="bvls", tol=1e-12)
out("lsq_linear", r.x, [r.cost], r.active_mask)
x, rn = nnls(A, b)
out("nnls", x, [rn])
