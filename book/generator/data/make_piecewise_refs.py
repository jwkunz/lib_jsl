"""Reference results for the entry on piecewise polynomials, Hermite data and scattered
interpolation (release 1.17.0), from SciPy 1.17.1.

    python book/generator/data/make_piecewise_refs.py > book/generator/data/piecewise_reference.txt

Random inputs are written as `input <name>|values`; results as `case|values`.
"""

import numpy as np
from scipy import interpolate as si
from scipy.interpolate import _interpnd as interpnd
from scipy.spatial import Delaunay

rng = np.random.default_rng(1170)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


xs = np.linspace(-0.2, 4.2, 45)

# PPoly from a not-a-knot cubic spline.
x = np.arange(9) * 0.5
y = np.sin(1.3 * x)
pp = si.PPoly.from_spline(si.make_interp_spline(x, y, k=3))
out("ppoly spline values", pp(xs))
out("ppoly spline d1", pp(xs, 1))
out("ppoly spline d2", pp(xs, 2))
out("ppoly spline integral", [pp.integrate(0.3, 3.7), pp.integrate(-0.5, 4.5)])
out("ppoly spline anti2", pp.antiderivative(2)(xs))
out("ppoly spline roots", pp.roots())
out("ppoly spline solve", pp.solve(0.5))

# Cubic Hermite spline.
xh = np.array([0.0, 0.4, 1.1, 1.5, 2.6, 3.0, 4.0])
yh = np.array([1.0, -0.5, 0.3, 2.0, -1.0, 0.0, 0.8])
dh = np.array([0.0, -2.0, 3.0, 1.0, -0.5, 2.0, 0.0])
ch = si.CubicHermiteSpline(xh, yh, dh)
out("hermite values", ch(xs))
out("hermite d1", ch(xs, 1))
out("hermite integral", [ch.integrate(0.1, 3.9)])
out("hermite roots", ch.roots())
out("hermite solve", ch.solve(0.25))

# Akima and makima, on Akima's data and on data with flat stretches.
xa = np.arange(11.0)
ya = np.array([10, 10, 10, 10, 10, 10, 10.5, 15, 50, 60, 85])
xq = np.linspace(0, 10, 101)
out("akima classic", si.Akima1DInterpolator(xa, ya)(xq))
out("makima classic", si.Akima1DInterpolator(xa, ya, method="makima")(xq))
xb = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
yb = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0])
xq2 = np.linspace(0, 7, 71)
out("akima flat", si.Akima1DInterpolator(xb, yb)(xq2))
out("makima flat", si.Akima1DInterpolator(xb, yb, method="makima")(xq2))

# Krogh with repeated nodes (derivatives).
xi = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.5, 3.0])
yi = np.array([1.0, 2.0, 0.0, -1.0, 4.0, 0.5, -0.3])
kr = si.KroghInterpolator(xi, yi)
xk = np.linspace(-0.3, 3.3, 37)
out("krogh values", kr(xk))
out("krogh derivatives", kr.derivatives(xk, der=4))

# Tensor-product splines.
gx = np.array([0.0, 0.3, 0.7, 1.2, 1.6, 2.1, 2.5, 3.0])
gy = np.array([-1.0, -0.6, -0.1, 0.4, 0.8, 1.3, 2.0])
gz = np.sin(gx[:, None]) * np.cos(1.5 * gy[None, :]) + 0.2 * gx[:, None] * gy[None, :]
qx = np.linspace(0.05, 2.95, 9)
qy = np.linspace(-0.95, 1.95, 8)
for kx, ky in ((3, 3), (1, 1), (5, 3)):
    rb = si.RectBivariateSpline(gx, gy, gz, kx=kx, ky=ky, s=0)
    out(f"rbs {kx}{ky} values", rb(qx, qy))
    if kx > 1 and ky > 1:  # FITPACK needs dx < kx
        out(f"rbs {kx}{ky} dx", rb(qx, qy, dx=1))
        out(f"rbs {kx}{ky} dxdy", rb(qx, qy, dx=1, dy=1))
    out(f"rbs {kx}{ky} integral", [rb.integral(0.2, 2.8, -0.7, 1.5)])

# Scattered 2-D data.
pts = rng.random((60, 2))
pts = np.vstack([pts, [[0, 0], [1, 0], [0, 1], [1, 1]]])
out("input points", pts)
vals = np.sin(3 * pts[:, 0]) * np.cos(2 * pts[:, 1]) + pts[:, 0]
g = np.linspace(0.02, 0.98, 13)
qq = np.array([[a, b] for a in g for b in g])
out("linear values", si.LinearNDInterpolator(pts, vals)(qq))
out("cubic values", si.CloughTocher2DInterpolator(pts, vals)(qq))
out("nearest values", si.NearestNDInterpolator(pts, vals)(qq))
grads = interpnd.estimate_gradients_2d_global(Delaunay(pts), vals)
out("cubic gradients", grads)
out("griddata outside", si.griddata(pts, vals, np.array([[1.2, 0.5], [-0.1, 0.3]]), method="linear"))

# RBFs in three dimensions.
p3 = rng.random((50, 3)) * 2 - 1
out("input points3", p3)
f3 = np.exp(-(p3 ** 2).sum(axis=1)) + p3[:, 0] * p3[:, 2]
q3 = rng.random((20, 3)) * 1.6 - 0.8
out("input queries3", q3)
cases = [
    ("tps", dict()),
    ("cubic", dict(kernel="cubic")),
    ("quintic", dict(kernel="quintic")),
    ("linear", dict(kernel="linear")),
    ("gaussian", dict(kernel="gaussian", epsilon=2.0)),
    ("multiquadric", dict(kernel="multiquadric", epsilon=1.5)),
    ("inverse_quadratic", dict(kernel="inverse_quadratic", epsilon=1.0)),
    ("tps smooth", dict(smoothing=0.1)),
    ("tps degree2", dict(degree=2)),
]
for name, kw in cases:
    out(f"rbf {name}", si.RBFInterpolator(p3, f3, **kw)(q3))
