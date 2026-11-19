"""SciPy references for release 2.11.0: least_squares(method="dogbox"), leastsq and
root(method="lm") (MINPACK), milp, and minimize(method="TNC"). Writes opt211_reference.txt."""
import warnings
import numpy as np
from scipy.optimize import least_squares, leastsq, root, milp, LinearConstraint, Bounds, minimize, rosen, rosen_der

out = []
def put(key, *lists):
    out.append(key + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))

# dogbox
f = lambda x: np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])
j = lambda x: np.array([[-20 * x[0], 10], [-1, 0]])
for name, x0, b in [("rosen1", [2, 2], ([-np.inf, 1.5], np.inf)), ("rosen2", [-1.2, 1.0], ([-2, -2], [0.5, 2]))]:
    r = least_squares(f, x0, jac=j, bounds=b, method="dogbox")
    put(f"dogbox {name}", r.x, [r.cost, r.nfev, r.njev, r.status], r.active_mask)
t = np.linspace(0, 4, 30)
y = 2.5 * np.exp(-1.3 * t) + 0.5 + 0.02 * np.sin(9 * t)
y[[5, 17]] += [1.0, -0.8]  # outliers
fe = lambda p: p[0] * np.exp(-p[1] * t) + p[2] - y
je = lambda p: np.stack([np.exp(-p[1] * t), -p[0] * t * np.exp(-p[1] * t), np.ones_like(t)], axis=1)
for name, loss, b in [("expfit", "linear", ([0, 0, 0], [10, 1.2, 1])), ("expfit soft_l1", "soft_l1", ([0, 0, 0], [10, 5, 1]))]:
    r = least_squares(fe, [1, 0.5, 0.1], jac=je, bounds=b, method="dogbox", loss=loss, f_scale=0.1)
    put(f"dogbox {name}", r.x, [r.cost, r.nfev, r.njev, r.status], r.active_mask)
put("expfit y", y)

# MINPACK
r = leastsq(f, [-1.2, 1], Dfun=j, full_output=True)
put("leastsq rosen jac", r[0], [r[2]["nfev"], r[2]["njev"], r[4]])
r = leastsq(f, [-1.2, 1], full_output=True)
put("leastsq rosen fd", r[0], [r[2]["nfev"], r[4]])
r = leastsq(fe, [1, 0.5, 0.1], full_output=True)
put("leastsq expfit", r[0], [r[2]["nfev"], r[4]])
h = lambda x: [x[0] ** 2 + x[1] ** 2 - 4, np.exp(x[0]) + x[1] - 1]
r = leastsq(h, [1, -1], full_output=True)
put("leastsq system", r[0], [r[2]["nfev"], r[4]])
s = root(h, [1, -1], method="lm")
put("root lm", s.x)
# Powell's badly scaled function as a root problem.
pb = lambda x: [1e4 * x[0] * x[1] - 1, np.exp(-x[0]) + np.exp(-x[1]) - 1.0001]
r = leastsq(pb, [0, 1], full_output=True, xtol=1e-12, ftol=1e-12)
put("leastsq powell", r[0], [r[2]["nfev"], r[4]])

# milp
c = [0, -1]
A = [[-1, 1], [3, 2], [2, 3]]
r = milp(c, constraints=LinearConstraint(A, -np.inf, [1, 12, 12]), integrality=[1, 1])
put("milp doc", r.x, [r.fun])
rng = np.random.default_rng(211)
v = rng.integers(10, 60, 20).astype(float); w = rng.integers(5, 30, 20).astype(float)
r = milp(-v, constraints=LinearConstraint([w], -np.inf, w.sum() * 0.4), integrality=np.ones(20), bounds=Bounds(0, 1), options=dict(mip_rel_gap=0))
put("milp knapsack v", v); put("milp knapsack w", w)
put("milp knapsack", r.x, [r.fun])
# A mixed problem: integer production quantities, continuous overtime, with equality.
cm = np.array([3.0, 2.5, 4.0, 1.5])
Am = np.array([[1, 1, 1, 0], [2, 1, 3, -1], [1, 2, 0, 0]])
r = milp(cm, constraints=[LinearConstraint(Am[:2], [7.5, 10], [np.inf, np.inf]), LinearConstraint(Am[2:], [6], [6])], integrality=[1, 1, 1, 0], bounds=Bounds([0, 0, 0, 0], [10, 10, 10, 4]), options=dict(mip_rel_gap=0))
put("milp mixed", r.x, [r.fun])
# Random integer program, 8 variables, 6 constraints.
Ar = rng.integers(1, 9, (6, 8)).astype(float); br = rng.integers(20, 40, 6).astype(float); cr = -rng.integers(1, 10, 8).astype(float)
r = milp(cr, constraints=LinearConstraint(Ar, -np.inf, br), integrality=np.ones(8), bounds=Bounds(0, 5), options=dict(mip_rel_gap=0))
put("milp random A", Ar); put("milp random b", br); put("milp random c", cr)
put("milp random", r.x, [r.fun])

# TNC
for n in [2, 6, 20]:
    r = minimize(rosen, [-1.2] * n, jac=rosen_der, method="TNC")
    put(f"tnc rosen {n}", r.x, [r.fun, r.nfev])
b = [(None, 0.5)] + [(None, None)] * 5
r = minimize(rosen, [-1.2] * 6, jac=rosen_der, method="TNC", bounds=b)
put("tnc rosen bounded", r.x, [r.fun, r.nfev])
r2 = minimize(rosen, [-1.2] * 6, jac=rosen_der, method="L-BFGS-B", bounds=b, options=dict(ftol=1e-15, gtol=1e-12))
put("lbfgsb rosen bounded", r2.x, [r2.fun])
open("opt211_reference.txt", "w").write("\n".join(out) + "\n")
