"""Reference derivatives from SciPy's adaptive finite differences, `scipy.differentiate` (the book
used SciPy 1.17.1).

    python book/generator/data/make_derivatives_refs.py > book/generator/data/derivatives_reference.txt

Each output line is `case|values` (comma-separated, `;` between rows of a matrix).
"""

import numpy as np
from scipy import differentiate


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


functions = {
    "exp(x) at 1": (np.exp, 1.0),
    "sin(x) at 1": (np.sin, 1.0),
    "ln(x) at 0.1": (np.log, 0.1),
    "1/(1 + 25 x^2) at 0.3": (lambda x: 1 / (1 + 25 * x * x), 0.3),
    "x^(5/2) at 2": (lambda x: x**2.5, 2.0),
}
import warnings

warnings.simplefilter("ignore")  # SciPy's default first step crosses zero for ln(x) at 0.1
for name, (f, x) in functions.items():
    # SciPy's default (a first step of 0.5) and a first step of 0.01.
    out(f"d1 {name}", [differentiate.derivative(f, x).df, differentiate.derivative(f, x, initial_step=0.01).df])
    # The second derivative as the derivative of SciPy's own first derivative.
    out(f"d2 {name}", [differentiate.derivative(lambda t: differentiate.derivative(f, t, initial_step=0.01).df, x, initial_step=0.01).df])


def rosen(v):
    return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2


x = np.array([-1.2, 1.0])
out("gradient rosenbrock", differentiate.jacobian(rosen, x).df)
out("hessian rosenbrock", *differentiate.hessian(rosen, x).ddf)
out("jacobian polar", *differentiate.jacobian(lambda v: np.array([v[0] * np.cos(v[1]), v[0] * np.sin(v[1])]), np.array([2.0, 0.5])).df)
