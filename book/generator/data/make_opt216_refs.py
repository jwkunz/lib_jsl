"""SciPy references for release 2.16.0: trust-krylov, COBYLA, trust-constr, least_squares with
tr_solver='lsmr', and scipy.sparse.linalg.lsmr. Constrained optima from SLSQP at tol 1e-14.
Writes opt216_reference.txt."""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize, least_squares, rosen, rosen_der, rosen_hess_prod, BFGS
from scipy.sparse.linalg import lsmr

out = []
def put(key, *lists):
    out.append(key + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))

# trust-krylov on Rosenbrock.
for name, x0 in [("rosen4", [-1.2, 1.0, -0.5, 0.8]), ("rosen30", [-1.2 if i % 2 == 0 else 1.0 for i in range(30)])]:
    r = minimize(rosen, x0, method="trust-krylov", jac=rosen_der, hessp=rosen_hess_prod, options={"gtol": 1e-8})
    put(f"krylov {name}", r.x, [r.fun, r.nit, r.nfev, r.njev, r.nhev])

# Constrained problems (SciPy convention c(x) >= 0).
problems = {
    "tutorial": (lambda x: x[0] * x[1], [{"type": "ineq", "fun": lambda x: 1 - x[0] ** 2 - x[1] ** 2}, {"type": "ineq", "fun": lambda x: x[1]}], [1.0, 1.0]),
    "rosendisc": (rosen, [{"type": "ineq", "fun": lambda x: 1.5 - x[0] ** 2 - x[1] ** 2}], [-1.0, 0.0]),
    "hs071": (lambda x: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
              [{"type": "ineq", "fun": lambda x: np.prod(x) - 25.0}, {"type": "eq", "fun": lambda x: np.sum(x ** 2) - 40.0}]
              + [{"type": "ineq", "fun": (lambda i: lambda x: x[i] - 1.0)(i)} for i in range(4)]
              + [{"type": "ineq", "fun": (lambda i: lambda x: 5.0 - x[i])(i)} for i in range(4)], [1.0, 5.0, 5.0, 1.0]),
}
for name, (f, cons, x0) in problems.items():
    if name == "tutorial":
        # SLSQP from (1, 1) stops at the stationary point (0, 0); the minimum is known exactly.
        put(f"{name} optimum", [-np.sqrt(0.5), np.sqrt(0.5)], [-0.5])
    else:
        ref = minimize(f, x0, method="SLSQP", constraints=cons, tol=1e-14, options={"maxiter": 1000})
        put(f"{name} optimum", ref.x, [ref.fun])
    ineq = [c for c in cons if c["type"] == "ineq"]
    if name != "hs071":
        r = minimize(f, x0, method="COBYLA", constraints=ineq, tol=1e-8, options={"maxiter": 1000})
        put(f"{name} cobyla", r.x, [r.fun, r.nfev, r.maxcv])
        r = minimize(f, x0, method="COBYLA", constraints=ineq, tol=1e-8, options={"maxiter": 10000})
        put(f"{name} cobyla10k", r.x, [r.fun, r.nfev, r.maxcv])
    r = minimize(f, x0, method="trust-constr", constraints=cons, hess=BFGS(), options={"gtol": 1e-8, "xtol": 1e-12, "maxiter": 3000})
    put(f"{name} trust-constr", r.x, [r.fun, r.nit, r.constr_violation])

# least_squares with tr_solver='lsmr': the Broyden tridiagonal system (sparse) and a bounded fit.
n = 100
def broyden(x):
    f = (3 - 2 * x) * x + 1
    f[1:] -= x[:-1]
    f[:-1] -= 2 * x[1:]
    return f
def broyden_jac(x):
    return sp.diags([np.full(n - 1, -1.0), 3 - 4 * x, np.full(n - 1, -2.0)], [-1, 0, 1], format="csr")
r = least_squares(broyden, -np.ones(n), jac=broyden_jac, tr_solver="lsmr")
put("broyden", r.x, [r.cost, r.nfev, r.njev, r.status])
t = np.linspace(0, 5, 21)
y = 2 * np.exp(-0.7 * t) + 0.01 * np.sin(3 * t)
fun = lambda p: p[0] * np.exp(-p[1] * t) - y
jac = lambda p: np.column_stack([np.exp(-p[1] * t), -p[0] * t * np.exp(-p[1] * t)])
for ub, loss in [(5.0, "linear"), (0.5, "linear"), (5.0, "soft_l1")]:
    r = least_squares(fun, [1.0, 0.1], jac=jac, bounds=([0, 0], [10, ub]), tr_solver="lsmr", loss=loss)
    put(f"expfit {ub} {loss}", r.x, [r.cost, r.nfev, r.status])

# lsmr itself.
rng = np.random.default_rng(216)
a = sp.random(60, 20, density=0.2, random_state=216, format="csr") + sp.eye(60, 20)
b = rng.normal(size=60)
put("lsmr a", a.toarray())
put("lsmr b", b)
for damp in [0.0, 0.5]:
    x, istop, itn, normr, normar, norma, conda, normx = lsmr(a, b, damp=damp, atol=1e-12, btol=1e-12)
    put(f"lsmr {damp}", x, [istop, itn, normr, normar])

open("data/opt216_reference.txt", "w").write("\n".join(out) + "\n")
