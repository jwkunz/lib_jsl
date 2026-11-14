"""Reference results for release 2.1.0: explicit Runge-Kutta methods in solve_ivp (RK45, DOP853),
quad_vec, and weighted quadrature (quad with weight=...), from SciPy 1.17.1.

    python book/generator/data/make_quadode_refs.py > book/generator/data/quadode_reference.txt
"""

import numpy as np
from scipy import integrate


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


# --- Explicit Runge-Kutta ---
vdp = lambda t, y: [y[1], 2 * (1 - y[0] ** 2) * y[1] - y[0]]
lorenz = lambda t, y: [10 * (y[1] - y[0]), y[0] * (28 - y[2]) - y[1], y[0] * y[1] - 8 / 3 * y[2]]
kepler = lambda t, y: [y[2], y[3], -y[0] / (y[0] ** 2 + y[1] ** 2) ** 1.5, -y[1] / (y[0] ** 2 + y[1] ** 2) ** 1.5]
problems = {
    "vdp": (vdp, (0.0, 20.0), [2.0, 0.0]),
    "lorenz": (lorenz, (0.0, 2.0), [1.0, 1.0, 1.0]),
    "kepler": (kepler, (0.0, 20.0), [0.4, 0.0, 0.0, 2.0]),
    "vdp back": (vdp, (20.0, 0.0), [2.0, 0.0]),
}
dense_t = {"vdp": [3.3, 11.7], "lorenz": [0.77, 1.5], "kepler": [5.5, 13.1], "vdp back": [15.2, 4.4]}
for method in ("RK45", "DOP853"):
    for p in (3, 6, 10):
        rtol = 10.0 ** -p
        for name, (f, span, y0) in problems.items():
            s = integrate.solve_ivp(f, span, y0, method=method, rtol=rtol, atol=rtol * 1e-3, dense_output=True)
            out(f"ivp {method} {p} {name}", [len(s.t), s.nfev], s.y[:, -1], s.sol(dense_t[name]))

# Work-precision data on the Kepler problem (exact solution known after one period, T = 2π·a^1.5).
e = 0.6
a = 1.0
T = 2 * np.pi
y0 = [1 - e, 0.0, 0.0, np.sqrt((1 + e) / (1 - e))]
for method in ("RK45", "DOP853"):
    rows = []
    for k in range(3, 13):
        rtol = 10.0 ** (-k)
        s = integrate.solve_ivp(kepler, (0, T), y0, method=method, rtol=rtol, atol=rtol, dense_output=True)
        err = np.max(np.abs(s.y[:, -1] - y0))
        rows += [rtol, s.nfev, err]
    out(f"kepler work {method}", rows)

# Events and per-component tolerances.
ground = lambda t, y: y[0]
ground.terminal = True
ground.direction = -1
s = integrate.solve_ivp(lambda t, y: [y[1], -9.81 - 0.1 * y[1] * abs(y[1])], (0, 10), [10.0, 5.0], method="DOP853", rtol=1e-10, atol=1e-12, events=ground)
out("event ground", [s.t_events[0][0]], s.y_events[0][0], [len(s.t), s.nfev])
s = integrate.solve_ivp(vdp, (0, 20), [2.0, 0.0], method="RK45", rtol=1e-6, atol=[1e-9, 1e-3])
out("atol vector", [len(s.t), s.nfev], s.y[:, -1])

# --- quad_vec ---
f5 = lambda x: np.array([np.sin(k * x) / np.sqrt(1 + x * x) + abs(x - 0.3) ** 0.5 for k in range(1, 6)])
g3 = lambda x: np.array([np.exp(-x * x * k) * (1 + np.sin(x)) for k in range(1, 4)])
qv_cases = [
    ("f5 max", f5, 0, 3, dict(norm="max")),
    ("f5 2", f5, 0, 3, dict(norm="2")),
    ("f5 points", f5, 0, 3, dict(points=[0.3])),
    ("f5 trapezoid", f5, 0, 3, dict(quadrature="trapezoid", epsrel=1e-5)),
    ("g3 semi", g3, 0, np.inf, dict()),
    ("g3 neg semi", g3, -np.inf, 0, dict()),
    ("g3 double", g3, -np.inf, np.inf, dict()),
    ("bessel", lambda x: np.array([np.cos(n * np.sin(x) - x * n) / np.pi for n in range(8)]), 0, np.pi, dict(epsrel=1e-12)),
]
for name, f, a, b, kw in qv_cases:
    r, err, info = integrate.quad_vec(f, a, b, full_output=True, **kw)
    out(f"qv {name}", r, [err, info.neval, len(info.intervals), info.status])

# --- Weighted quadrature ---
wcases = [
    ("cos finite", lambda x: np.exp(-x) * x, 0, 10, "cos", 25.0),
    ("sin finite", lambda x: 1 / (1 + x * x), -2, 7, "sin", 40.0),
    ("cos fourier", lambda x: 1 / (1 + x * x), 0, np.inf, "cos", 3.0),
    ("sin fourier", lambda x: np.exp(-0.1 * x) / (1 + x), 1, np.inf, "sin", 2.5),
    ("sin fourier slow", lambda x: 1 / np.sqrt(x), 1, np.inf, "sin", 1.0),
    ("alg", np.cos, 0, 2, "alg", (-0.5, 0.3)),
    ("alg-loga", np.exp, 0, 1, "alg-loga", (0.0, 0.0)),
    ("alg-logb", lambda x: 1 + x * x, -1, 1, "alg-logb", (0.5, -0.7)),
    ("alg-log", np.cos, 0, 3, "alg-log", (-0.3, -0.3)),
    ("cauchy", np.exp, -1, 2, "cauchy", 0.5),
    ("cauchy asym", lambda x: 1 / (1 + x ** 2), 0, 10, "cauchy", 1.0),
]
for name, f, a, b, w, wvar in wcases:
    r = integrate.quad(f, a, b, weight=w, wvar=wvar, full_output=1)
    out(f"w {name}", [r[0], r[1], r[2]["neval"] if "neval" in r[2] else -1])
