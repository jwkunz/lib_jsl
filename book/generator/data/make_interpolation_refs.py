"""Reference values for the interpolation chapter, from SciPy and NumPy (the book used SciPy 1.17.1
and NumPy 2.4).

    python book/generator/data/make_interpolation_refs.py > book/generator/data/interpolation_reference.txt

Each output line is `case|values` (comma-separated, `;` between lists). The data are defined by
formulas that the generator repeats exactly.
"""

import numpy as np
from scipy import interpolate, signal


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


# One-dimensional data: uneven nodes, a smooth curve and a curve with flat stretches.
i = np.arange(12)
x = i + 0.3 * np.sin(i)
smooth = np.sin(x) + 0.05 * x**2
steps = np.array([0, 0, 0.5, 1, 1, 1.2, 3, 3.1, 3.1, 4, 6, 6.2])
xq = np.linspace(x[0], x[-1], 61)
for name, y in [("smooth", smooth), ("steps", steps)]:
    out(f"natural spline {name}", interpolate.CubicSpline(x, y, bc_type="natural")(xq))
    out(f"pchip {name}", interpolate.PchipInterpolator(x, y)(xq))
    out(f"akima {name}", interpolate.Akima1DInterpolator(x, y)(xq))
    out(f"linear {name}", np.interp(xq, x, y))

# Polynomial and rational interpolation of Runge's function.
runge = lambda t: 1 / (1 + 25 * t * t)
tq = np.linspace(-1, 1, 101)
cheb = np.cos(np.pi * np.arange(9) / 8)
out("polynomial chebyshev 9", interpolate.BarycentricInterpolator(cheb, runge(cheb))(tq))
eq = np.linspace(-1, 1, 21)
out("floater-hormann 21 d3", interpolate.FloaterHormannInterpolator(eq, runge(eq), d=3)(tq))
z = np.linspace(-1, 1, 1000)
f = lambda t: np.tan(np.pi * t / 2.2)
out("aaa tan", interpolate.AAA(z, f(z), rtol=1e-13)(tq))
taylor = [1 / np.prod(np.arange(1, k + 1)) for k in range(9)]
p, q = interpolate.pade(taylor, 4, 4)
out("pade exp 4 4", p.coeffs[::-1] / q.coeffs[-1], q.coeffs[::-1] / q.coeffs[-1])
out("chebyshev exp 12", np.polynomial.chebyshev.chebinterpolate(np.exp, 11))

# Savitzky-Golay coefficients and filtering, with SciPy's "interp" end treatment.
out("savgol coeffs 11 3 0", signal.savgol_coeffs(11, 3, deriv=0))
out("savgol coeffs 11 3 1", signal.savgol_coeffs(11, 3, deriv=1, delta=0.1))
k = np.arange(100)
noisy = np.sin(k / 8) + 0.1 * np.sin(k * 2.3)
out("savgol filter 11 3 0", signal.savgol_filter(noisy, 11, 3))
out("savgol filter 11 3 1", signal.savgol_filter(noisy, 11, 3, deriv=1, delta=0.1))

# Fourier interpolation: periodic samples resampled onto a four times finer grid.
for n in [15, 16]:
    t = 2 * np.pi * np.arange(n) / n
    out(f"fourier {n}", signal.resample(np.exp(np.sin(t)) + 0.3 * np.cos(3 * t), 4 * n))

# Gridded data in two and three dimensions.
gx, gy = np.array([0, 0.5, 1.3, 2, 3.1]), np.array([0, 1, 1.5, 2.5])
g2 = np.sin(gx[:, None]) * np.cos(gy[None, :]) + gx[:, None] * gy[None, :] / 4
q2 = np.array([[0.05 + 3.0 * ((7 * j) % 41) / 41, 0.05 + 2.4 * ((11 * j) % 37) / 37] for j in range(40)])
out("grid2 linear", interpolate.RegularGridInterpolator((gx, gy), g2, method="linear")(q2))
out("grid2 nearest", interpolate.RegularGridInterpolator((gx, gy), g2, method="nearest")(q2))
ax = np.linspace(0, 2, 6)
g3 = np.sin(ax[:, None, None]) + np.cos(1.3 * ax[None, :, None]) * ax[None, None, :] ** 2 / 4
q3 = np.array([[2.0 * ((5 * j) % 29) / 29, 2.0 * ((13 * j) % 31) / 31, 2.0 * ((17 * j) % 23) / 23] for j in range(40)])
out("grid3 linear", interpolate.RegularGridInterpolator((ax, ax, ax), g3, method="linear")(q3))


def natural_tensor(v, point):
    """The tensor-product natural cubic spline, by natural splines along each axis in turn."""
    for axis_value in point:
        v = interpolate.CubicSpline(ax, v, axis=0, bc_type="natural")(axis_value)
    return v


out("grid3 natural spline", [natural_tensor(g3, p) for p in q3])

# Scattered data and radial basis functions (all with a linear polynomial term).
sp = np.array([[((37 * j) % 53) / 53, ((61 * j) % 47) / 47] for j in range(30)])
sf = np.sin(3 * sp[:, 0]) * np.cos(2 * sp[:, 1]) + sp[:, 0] ** 2
sq = np.array([[((29 * j) % 43) / 43, ((19 * j) % 41) / 41] for j in range(40)])
for kernel, eps in [("thin_plate_spline", 1.0), ("cubic", 1.0), ("gaussian", 1.5), ("multiquadric", 1.5), ("inverse_multiquadric", 1.5)]:
    out(f"rbf {kernel}", interpolate.RBFInterpolator(sp, sf, kernel=kernel, epsilon=eps, degree=1)(sq))
