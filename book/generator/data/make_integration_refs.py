"""Reference values for the numerical integration chapter, from SciPy, with Gauss-Legendre weights
also from mpmath (the book used SciPy 1.17.1 and mpmath 1.3).

    python book/generator/data/make_integration_refs.py > book/generator/data/integration_reference.txt

Each output line is `case|values` (comma-separated, `;` between lists). The generator recomputes
every case with the library and tabulates the differences.
"""

import numpy as np
from scipy import integrate, special

np.set_printoptions(precision=17)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


# Gaussian rules: nodes and weights, with Gauss-Legendre weights also from mpmath at 40 digits
# (Newton's method on the Legendre polynomial), which shows SciPy's own error.
import mpmath as mp

mp.mp.dps = 40


def legendre_weights(n):
    weights = []
    for k in range(1, n + 1):
        x = mp.cos(mp.pi * (k - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
        for _ in range(100):
            dp = n * (x * mp.legendre(n, x) - mp.legendre(n - 1, x)) / (x * x - 1)
            step = mp.legendre(n, x) / dp
            x -= step
            if abs(step) < mp.mpf(10) ** -35:
                break
        dp = n * (x * mp.legendre(n, x) - mp.legendre(n - 1, x)) / (x * x - 1)
        weights.append((x, 2 / ((1 - x * x) * dp * dp)))
    return [w for _, w in sorted(weights)]


for n in [5, 20, 100]:
    out(f"gauss_legendre {n}", *special.roots_legendre(n), legendre_weights(n))
for n in [10, 40]:
    out(f"gauss_hermite {n}", *special.roots_hermite(n))
for n, alpha in [(10, 0.0), (20, 1.5)]:
    out(f"gauss_laguerre {n} {alpha}", *special.roots_genlaguerre(n, alpha))

# One-dimensional integrals: QUADPACK (quad) and SciPy's own tanh-sinh.
integrands = {
    "exp(-x^2) on [0, 1]": (lambda x: np.exp(-x * x), 0.0, 1.0),
    "1/(1 + 25 x^2) on [-1, 1]": (lambda x: 1 / (1 + 25 * x * x), -1.0, 1.0),
    "sqrt(x) ln(x) on [0, 1]": (lambda x: np.sqrt(x) * np.log(x) if x > 0 else 0.0, 0.0, 1.0),
    "1/sqrt(1 - x^2) on [-1, 1]": (lambda x: 1 / np.sqrt(1 - x * x) if abs(x) < 1 else 0.0, -1.0, 1.0),
    "cos(30 x) exp(x) on [0, 2]": (lambda x: np.cos(30 * x) * np.exp(x), 0.0, 2.0),
}
for name, (f, a, b) in integrands.items():
    q, _ = integrate.quad(f, a, b, epsabs=0, epsrel=1e-13, limit=500)
    t = integrate.tanhsinh(np.vectorize(f), a, b, rtol=1e-14).integral
    out(f"quad {name}", [q, t])

# Cubature over boxes.
cubes = {
    "exp(-r^2) on [0, 1]^3": (lambda x: np.exp(-(x**2).sum(axis=-1)), [0, 0, 0], [1, 1, 1]),
    "cos(x + 2y) on [0, 1] x [0, 2]": (lambda x: np.cos(x[..., 0] + 2 * x[..., 1]), [0, 0], [1, 2]),
}
for name, (f, a, b) in cubes.items():
    res = integrate.cubature(f, a, b, rtol=1e-12, atol=0)
    out(f"cubature {name}", [res.estimate])


# Initial value problems: the final state from tightly converged SciPy solutions.
def vdp(mu):
    return lambda t, y: [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]


sol = integrate.solve_ivp(vdp(1.0), (0, 10), [2.0, 0.0], method="DOP853", rtol=1e-13, atol=1e-13)
out("ivp van der Pol mu 1, t = 10", sol.y[:, -1])
sol = integrate.solve_ivp(vdp(100.0), (0, 100), [2.0, 0.0], method="Radau", rtol=1e-11, atol=1e-11)
out("ivp van der Pol mu 100, t = 100", sol.y[:, -1])


def robertson(t, y):
    return [-0.04 * y[0] + 1e4 * y[1] * y[2], 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2, 3e7 * y[1] ** 2]


sol = integrate.solve_ivp(robertson, (0, 1e3), [1.0, 0.0, 0.0], method="Radau", rtol=1e-11, atol=1e-16)
out("ivp Robertson, t = 1000", sol.y[:, -1])

# A boundary value problem: Bratu's equation y'' + exp(y) = 0, y(0) = y(1) = 0, lower branch.
x = np.linspace(0, 1, 11)
sol = integrate.solve_bvp(lambda x, y: np.vstack([y[1], -np.exp(y[0])]), lambda ya, yb: np.array([ya[0], yb[0]]), x, np.zeros((2, x.size)), tol=1e-10, max_nodes=100000)
out("bvp Bratu", sol.sol(x)[0])
