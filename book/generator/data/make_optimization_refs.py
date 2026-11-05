"""Reference optima for the optimization chapters, from SciPy (the book used SciPy 1.17.1).

    python book/generator/data/make_optimization_refs.py > book/generator/data/optimization_reference.txt

Each output line is `case|values` (comma-separated, `;` between lists).
"""

import numpy as np
from scipy import optimize


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


# Unconstrained minimisation of the Rosenbrock function in 2 and 5 dimensions.
for n, x0 in [(2, [-1.2, 1.0]), (5, [-1.2, 1.0, -0.5, 0.8, 1.5])]:
    for method in ["BFGS", "L-BFGS-B", "CG", "Newton-CG", "Nelder-Mead", "Powell"]:
        kw = {}
        if method in ("BFGS", "L-BFGS-B", "CG", "Newton-CG"):
            kw["jac"] = optimize.rosen_der
        if method == "Newton-CG":
            kw["hess"] = optimize.rosen_hess
        opts = {"Nelder-Mead": dict(xatol=1e-12, fatol=1e-14, maxiter=200000, maxfev=200000, adaptive=True), "Powell": dict(xtol=1e-12, ftol=1e-14, maxiter=100000)}.get(method, dict(gtol=1e-12) if method != "Newton-CG" else dict(xtol=1e-12))
        if method == "L-BFGS-B":
            opts = dict(gtol=1e-12, ftol=1e-15)
        r = optimize.minimize(optimize.rosen, x0, method=method, options=opts, **kw)
        out(f"rosenbrock {n} {method}", r.x, [r.fun])

# One-dimensional minimisation: the surface area 2 pi r^2 + 2/r of a can of unit volume.
area = lambda r: 2 * np.pi * r * r + 2 / r
for method in ["golden", "brent"]:
    r = optimize.minimize_scalar(area, bracket=(0.1, 0.5, 2.0), method=method, options=dict(xtol=1e-12))
    out(f"scalar {method}", [r.x, r.fun])

# Weighted curve fitting: radioactive decay, N(t) = N0 exp(-t / tau), with Poisson errors.
t = 10.0 * np.arange(10)
counts = np.array([1031.0, 735.0, 589.0, 409.0, 331.0, 226.0, 187.0, 128.0, 109.0, 70.0])
p, cov = optimize.curve_fit(lambda t, n0, tau: n0 * np.exp(-t / tau), t, counts, p0=[1000, 50], sigma=np.sqrt(counts), absolute_sigma=True)
out("curve_fit decay", p, np.sqrt(np.diag(cov)))
p, cov = optimize.curve_fit(lambda t, n0, tau: n0 * np.exp(-t / tau), t, counts, p0=[1000, 50])
out("curve_fit decay unweighted", p, np.sqrt(np.diag(cov)))

# Linear programs with HiGHS: the diet problem, and a random feasible bounded program.
r = optimize.linprog([2, 3.5, 8], A_ub=-np.array([[65, 120, 106], [2, 8, 7]]), b_ub=[-300, -20], bounds=[(0, None), (0, 2), (0, None)], method="highs")
out("linprog diet", r.x, [r.fun])
rng = np.random.default_rng(7)
A = np.round(rng.uniform(0, 1, (8, 10)), 3)
b = np.round(rng.uniform(2, 5, 8), 3)
c = -np.round(rng.uniform(0.5, 2, 10), 3)
print("linprog random data|" + ";".join([",".join(repr(float(v)) for v in A.ravel()), ",".join(repr(float(v)) for v in b), ",".join(repr(float(v)) for v in c)]))
r = optimize.linprog(c, A_ub=A, b_ub=b, method="highs")
out("linprog random", r.x, [r.fun])

# Constrained minimisation with SLSQP: the largest box of surface area 6, and Hock-Schittkowski 71.
box = optimize.minimize(lambda x: -x[0] * x[1] * x[2], [0.5, 1.0, 2.0], method="SLSQP", constraints=[dict(type="eq", fun=lambda x: 2 * (x[0] * x[1] + x[1] * x[2] + x[2] * x[0]) - 6)], bounds=[(0, None)] * 3, options=dict(ftol=1e-14))
out("slsqp box", box.x, [box.fun])
hs71 = optimize.minimize(lambda x: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2], [1, 5, 5, 1], method="SLSQP", constraints=[dict(type="ineq", fun=lambda x: x[0] * x[1] * x[2] * x[3] - 25), dict(type="eq", fun=lambda x: (x**2).sum() - 40)], bounds=[(1, 5)] * 4, options=dict(ftol=1e-14))
out("slsqp hs71", hs71.x, [hs71.fun])

# A 0/1 knapsack as a mixed-integer program (HiGHS branch and bound).
weights = np.array([23, 31, 29, 44, 53, 38, 63, 85, 89, 82])
values = np.array([92, 57, 49, 68, 60, 43, 67, 84, 87, 72], float)
r = optimize.milp(-values, constraints=optimize.LinearConstraint(weights[None, :], ub=165), integrality=np.ones(10), bounds=optimize.Bounds(0, 1))
out("knapsack", [-r.fun], np.flatnonzero(np.round(r.x)))

# Global minimisation of Rastrigin's function in 3 dimensions with differential evolution.
rastrigin = lambda x: 10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x))
r = optimize.differential_evolution(rastrigin, [(-5.12, 5.12)] * 3, seed=1, tol=1e-12, polish=True)
out("de rastrigin", r.x, [r.fun])
