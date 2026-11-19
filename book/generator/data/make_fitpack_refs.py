"""SciPy references for release 2.9.0: smoothing splines with automatic knots (make_splrep,
make_splprep, generate_knots, and FITPACK's own splrep and splprep) and Bernstein-basis
piecewise polynomials (BPoly). Writes fitpack_reference.txt."""
import warnings
import numpy as np
from scipy import interpolate as si

out = []
def put(key, *lists):
    out.append(key + "|" + ";".join(",".join(repr(float(v)) for v in l) for l in lists))

rng = np.random.default_rng(29)
m = 80
x = np.sort(rng.uniform(0, 10, m))
x[0], x[-1] = 0.0, 10.0
y = np.sin(x) + 0.3 * np.cos(3 * x) + rng.normal(0, 0.15, m)
w = 1 / rng.uniform(0.1, 0.2, m)
put("x", x)
put("y", y)
put("w", w)
cases = [
    ("cubic s=0.5", dict(k=3, s=0.5)),
    ("cubic s=2", dict(k=3, s=2.0)),
    ("cubic s=8", dict(k=3, s=8.0)),
    ("linear s=1", dict(k=1, s=1.0)),
    ("quadratic s=1", dict(k=2, s=1.0)),
    ("quartic s=1", dict(k=4, s=1.0)),
    ("quintic s=1", dict(k=5, s=1.0)),
    ("weighted s=m", dict(k=3, s=float(m), w=w)),
    ("interval [-1, 11]", dict(k=3, s=2.0, xb=-1.0, xe=11.0)),
    ("nest=14", dict(k=3, s=0.01, nest=14)),
    ("s huge", dict(k=3, s=1e4)),
]
for name, kw in cases:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spl = si.make_splrep(x, y, **kw)
        fw = kw.get("w", np.ones(m))
        fp = float(np.sum((fw * (y - spl(x))) ** 2))
        # FITPACK's Fortran curfit.
        # (splrep has no nest argument.)
        tck = si.splrep(x, y, **kw) if "nest" not in kw else ([], [])
    n = len(tck[0])
    put(f"splrep {name}", spl.t, spl.c, [fp], tck[0], tck[1][: max(n - kw["k"] - 1, 0)])
put("generate_knots", [len(t) for t in si.generate_knots(x, y, k=3, s=0.5)])

# A noisy closed-ish curve in 3-D for make_splprep.
mc = 60
a = np.linspace(0, 1.8 * np.pi, mc)
pts = np.array([np.cos(a), np.sin(2 * a) / 2, a / 6]) + rng.normal(0, 0.02, (3, mc))
put("curve", *pts)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    spl, u = si.make_splprep(pts, s=0.05)
    tck, u2 = si.splprep(pts, s=0.05)
put("splprep", u, spl.t, *spl.c.T, tck[0])

# BPoly.
xi = [0.0, 1.0, 2.5, 4.0]
yi = [[1.0, 0.0, -2.0], [0.5, -1.0], [2.0, 0.3, 0.1], [-1.0, 1.5]]
ev = np.linspace(-0.5, 4.5, 23)
for name, orders in [("none", None), ("3", 3), ("1", 1)]:
    bp = si.BPoly.from_derivatives(xi, yi, orders=orders)
    put(f"bpoly {name}", bp.c.ravel(), bp(ev), bp(ev, 1), bp.antiderivative(2)(ev), [bp.integrate(-0.3, 3.7)])
cs = si.CubicSpline([0, 1, 2.5, 4, 5], [1, -1, 0.5, 2, 0])
bp = si.BPoly.from_power_basis(cs)
pp = si.PPoly.from_bernstein_basis(bp)
put("power", bp.c.ravel(), pp.c.ravel())

open("fitpack_reference.txt", "w").write("\n".join(out) + "\n")
