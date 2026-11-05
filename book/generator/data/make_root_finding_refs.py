"""Reference roots for the root-finding chapter, from SciPy and NumPy (the book used SciPy 1.17.1
and NumPy 2.4).

    python book/generator/data/make_root_finding_refs.py > book/generator/data/root_finding_reference.txt

Each output line is `case|values` (comma-separated, `;` between lists).
"""

import numpy as np
from scipy import optimize


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


# Scalar equations: (f, f', bracket, Newton start).
functions = {
    "cos(x) = x": (lambda x: np.cos(x) - x, lambda x: -np.sin(x) - 1, (0.0, 1.0), 0.5),
    "x^3 - 2x - 5 = 0": (lambda x: x**3 - 2 * x - 5, lambda x: 3 * x**2 - 2, (2.0, 3.0), 2.0),
    "Kepler: E - 0.9 sin E = 0.3": (lambda x: x - 0.9 * np.sin(x) - 0.3, lambda x: 1 - 0.9 * np.cos(x), (0.0, np.pi), 1.0),
    "x e^-x = 0.1": (lambda x: x * np.exp(-x) - 0.1, lambda x: (1 - x) * np.exp(-x), (0.0, 0.5), 0.0),
    "ln x + x^2 = 3": (lambda x: np.log(x) + x * x - 3, lambda x: 1 / x + 2 * x, (1.0, 2.0), 1.5),
}
tol = dict(xtol=1e-15, rtol=4 * np.finfo(float).eps)
for name, (f, df, (a, b), x0) in functions.items():
    out(
        f"scalar {name}",
        [
            optimize.brentq(f, a, b, **tol),
            optimize.bisect(f, a, b, **tol),
            optimize.ridder(f, a, b, **tol),
            optimize.toms748(f, a, b, **tol),
            optimize.newton(f, x0, fprime=df, tol=1e-15),
            optimize.newton(f, x0, x1=x0 + 0.1, tol=1e-15),
        ],
    )

# Polynomials, highest power first; roots sorted by real part, then imaginary part.
polynomials = {
    "Wilkinson-like, roots 1 to 10": np.poly(np.arange(1, 11)),
    "x^5 - x - 1": [1, 0, 0, 0, -1, -1],
    "(x - 1)^2 (x + 2) (x^2 + 1)": np.polymul(np.polymul([1, -2, 1], [1, 2]), [1, 0, 1]),
    "degree 12, coefficients 1 to 13": np.arange(1, 14, dtype=float),
}
for name, c in polynomials.items():
    r = np.roots(c)
    r = r[np.lexsort((r.imag, r.real))]
    out(f"poly {name}", r.real, r.imag)


# Nonlinear systems solved by MINPACK's hybrid method.
def circle_exp(v):
    x, y = v
    return [x * x + y * y - 4, np.exp(x) + y - 1]


def trig3(v):
    x, y, z = v
    return [3 * x - np.cos(y * z) - 0.5, x * x - 81 * (y + 0.1) ** 2 + np.sin(z) + 1.06, np.exp(-x * y) + 20 * z + (10 * np.pi - 3) / 3]


def chandrasekhar_like(v):
    # A five-variable tridiagonal system: (3 - 2 x_i) x_i - x_{i-1} - 2 x_{i+1} + 1 = 0 (Broyden).
    x = np.concatenate([[0], v, [0]])
    return [(3 - 2 * x[i]) * x[i] - x[i - 1] - 2 * x[i + 1] + 1 for i in range(1, 6)]


for name, f, x0 in [("circle and exponential", circle_exp, [1.0, -1.0]), ("three trigonometric equations", trig3, [0.1, 0.1, -0.1]), ("Broyden tridiagonal, 5 variables", chandrasekhar_like, [-1.0] * 5)]:
    out(f"system {name}", optimize.fsolve(f, x0, xtol=1e-14))
