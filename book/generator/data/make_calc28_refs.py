"""SciPy references for release 2.8.0: Faddeeva, Voigt, Struve, Kelvin, Hankel and Bessel zeros
(scipy.special), and Halley's method, fixed points, isotonic regression, finite-difference
gradients and brute-force search (scipy.optimize). Writes calc28_reference.txt."""
import numpy as np
from scipy import special as sp, optimize as so

out = []
def k(v):
    """A number as Rust's Display writes it."""
    return repr(float(v)).removesuffix(".0")
def put(key, *lists):
    out.append(key + "|" + ";".join(",".join(repr(float(v)) for v in l) for l in lists))

# wofz on a grid covering both half planes, the real axis and |z| beyond 6.
xs = [-8.0, -3.0, -0.5, 0.0, 0.7, 2.0, 5.5, 12.0]
ys = [-2.0, -0.3, 0.0, 1e-3, 0.5, 2.0, 6.0, 20.0]
for y in ys:
    w = sp.wofz(np.array(xs) + 1j * y)
    put(f"wofz {k(y)}", w.real, w.imag)
vx = [-6.0, -2.0, -0.5, 0.0, 0.3, 1.0, 3.0, 10.0]
for s, g in [(1.0, 0.0), (0.0, 1.0), (1.0, 0.5), (0.3, 2.0), (2.0, 0.05)]:
    put(f"voigt {k(s)} {k(g)}", [sp.voigt_profile(x, s, g) for x in vx])

sx = [0.1, 1.0, 3.0, 7.0, 12.0, 20.0, 30.0, 50.0, 100.0]
for v in [0.0, 1.0, 2.5, -0.5, 5.0, 10.0]:
    put(f"struve {k(v)}", sp.struve(v, sx), sp.modstruve(v, sx))

kx = [0.01, 0.1, 0.5, 1.0, 2.8, 5.0, 8.0, 10.0, 20.0, 30.0, 50.0]
for name in ["ber", "bei", "ker", "kei", "berp", "beip", "kerp", "keip"]:
    put(f"kelvin {name}", getattr(sp, name)(kx))

hx = [0.2, 2.0, 15.0, 60.0]
for v in [0.0, 1.5, 7.3]:
    h1 = sp.hankel1(v, hx)
    h2 = sp.hankel2(v, hx)
    put(f"hankel {k(v)}", h1.real, h1.imag, h2.real, h2.imag)

for n in [0, 1, 5, 20]:
    put(f"zeros {n}", sp.jn_zeros(n, 10), sp.jnp_zeros(n, 10), sp.yn_zeros(n, 10), sp.ynp_zeros(n, 10))

# Halley: SciPy's newton with fprime2, default tol.
cases = {
    "cbrt": (lambda x: x**3 - 10, lambda x: 3 * x**2, lambda x: 6 * x, 1.0),
    "kepler": (lambda E: E - 0.9 * np.sin(E) - 0.3, lambda E: 1 - 0.9 * np.cos(E), lambda E: 0.9 * np.sin(E), 1.0),
    "lambert": (lambda w: w * np.exp(w) - 5, lambda w: (1 + w) * np.exp(w), lambda w: (2 + w) * np.exp(w), 1.0),
}
for name, (f, fp, fpp, x0) in cases.items():
    put(f"halley {name}", [so.newton(f, x0, fprime=fp, fprime2=fpp)])
fp_cases = {
    "sqrt": (lambda x: np.sqrt(10 / (x + 2)), 1.2),
    "cos": (lambda x: np.cos(x), 0.5),
    "logistic": (lambda x: 3.5 * x * (1 - x), 0.7),
}
for name, (g, x0) in fp_cases.items():
    put(f"fixed_point {name}", [so.fixed_point(g, x0)])

rng = np.random.default_rng(28)
y = np.cumsum(rng.normal(0.2, 1.0, 60))
w = rng.uniform(0.5, 2.0, 60)
put("isotonic y", y)
put("isotonic w", w)
for inc in [True, False]:
    r = so.isotonic_regression(y, weights=w, increasing=inc)
    put(f"isotonic {str(inc).lower()}", r.x, r.weights, r.blocks)

def rosen(x):
    return sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
x0 = np.array([-1.2, 1.0, 0.7, -0.4])
eps = np.sqrt(np.finfo(float).eps)
put("approx_fprime", so.approx_fprime(x0, rosen, eps))
put("check_grad", [so.check_grad(rosen, so.rosen_der, x0)])

def himmel(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
rr = so.brute(himmel, ((-5, 5), (-5, 5)), Ns=20, full_output=True, finish=None)
put("brute grid", rr[0], [rr[1]])
rf = so.brute(himmel, ((-5, 5), (-5, 5)), Ns=20, full_output=True)
put("brute finish", rf[0], [rf[1]])

open("calc28_reference.txt", "w").write("\n".join(out) + "\n")
