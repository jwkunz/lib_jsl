"""Reference responses for the controls entry, from `scipy.signal.dlsim` (the book used SciPy 1.17.1).

    python book/generator/data/make_controls_refs.py > book/generator/data/controls_reference.txt

The library's `StateSpace::step` updates the state before reading the output,
y[k] = C x[k+1] + D u[k], which is SciPy's convention y[k] = C' x[k] + D' u[k] with C' = C A and
D' = C B + D. Each output line is `case|values`.
"""

import numpy as np
from scipy import linalg, signal


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


A = np.array([[0.9, 0.2, 0.0], [-0.1, 0.8, 0.3], [0.0, -0.2, 0.7]])
B = np.array([[1.0, 0.0], [0.5, 0.2], [0.0, 1.0]])
C = np.array([[1.0, 0.0, -1.0], [0.3, 1.0, 0.0]])
D = np.array([[0.0, 0.1], [0.2, 0.0]])
x0 = np.array([0.5, -0.2, 0.1])
k = np.arange(60)
u = np.column_stack([np.sin(0.2 * k), np.where(k % 20 < 10, 1.0, -1.0)])
_, y, _ = signal.dlsim((A, B, C @ A, C @ B + D, 1.0), u, x0=x0)
out("dlsim", y)
e = linalg.eigvals(A)
e = e[np.lexsort((e.imag, e.real))]
out("poles", e.real, e.imag)
