"""Reference results for the entry on B-splines and smoothing splines, from `scipy.interpolate`
(the book used SciPy 1.17.1).

    python book/generator/data/make_bspline_refs.py > book/generator/data/bspline_reference.txt

Inputs are written first (`data <name>|values`), then results as `case|values`.
"""

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline, make_lsq_spline, make_smoothing_spline, splprep, splev

rng = np.random.default_rng(23)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


x = np.round(np.sort(rng.uniform(0.0, 6.0, 15)), 6)
x[0], x[-1] = 0.0, 6.0
y = np.round(np.sin(x) + 0.3 * np.cos(2.3 * x), 6)
xe = np.linspace(-0.5, 6.5, 141)
out("data x", x)
out("data y", y)
out("data xe", xe)

cases = [
    ("cubic not-a-knot", 3, None),
    ("quadratic not-a-knot", 2, None),
    ("quartic not-a-knot", 4, None),
    ("quintic not-a-knot", 5, None),
    ("cubic natural", 3, "natural"),
    ("cubic clamped", 3, "clamped"),
    ("cubic given slopes", 3, ([(1, 0.5)], [(1, -1.0)])),
    ("quintic given derivatives", 5, ([(1, 0.5), (2, 0.0)], [(1, -1.0), (2, 0.25)])),
    ("linear", 1, None),
]
for name, k, bc in cases:
    s = make_interp_spline(x, y, k, bc_type=bc)
    out("interp " + name, s(xe))
    if k >= 1:
        out("derivative " + name, s.derivative(1)(xe))
    out("integral " + name, [s.integrate(0.3, 5.1), s.integrate(-0.5, 6.5)])
    out("antiderivative " + name, s.antiderivative(1)(xe))

yp = y.copy(); yp[-1] = yp[0]
out("data yp", yp)
for k in (3, 4):
    s = make_interp_spline(x, yp, k, bc_type="periodic")
    out(f"periodic {k}", s(xe))
    out(f"periodic integral {k}", [s.integrate(-0.5, 13.0)])

# Least squares on given knots, with weights.
xd = np.round(np.sort(rng.uniform(0.0, 6.0, 60)), 6)
yd = np.round(np.sin(xd) + 0.1 * rng.normal(size=60), 6)
wd = np.round(rng.uniform(0.5, 2.0, 60), 6)
out("data xd", xd)
out("data yd", yd)
out("data wd", wd)
t = np.r_[[xd[0]] * 4, [1.0, 2.0, 2.5, 4.0, 5.0], [xd[-1]] * 4]
out("data t", t)
s = make_lsq_spline(xd, yd, t, k=3, w=wd)
out("lsq", s(xe))

# Smoothing splines: a fixed lambda and GCV.
for lam in (0.05, 2.0):
    out(f"smoothing {lam}", make_smoothing_spline(xd, yd, w=wd, lam=lam)(xe))
s = make_smoothing_spline(xd, yd)
out("smoothing gcv", s(xe))

# A parametric curve: splprep with s = 0 against chord-length make_interp_spline.
theta = np.linspace(0, 1.7 * np.pi, 12)
pts = np.round(np.c_[np.cos(theta) * (1 + 0.2 * theta), np.sin(theta) * (1 + 0.2 * theta)], 6)
out("data curve", pts)
tck, u = splprep(pts.T, s=0, k=3)
out("splprep u", u)
uu = np.linspace(0, 1, 101)
cx, cy = splev(uu, tck)
out("splprep curve", cx, cy)
