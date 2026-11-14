"""Reference results for the entry on matrix functions and matrix equations, from `scipy.linalg`
(the book used SciPy 1.17.1).

    python book/generator/data/make_matrix_equations_refs.py > book/generator/data/matrix_equations_reference.txt

Matrices are written first, as `matrix <name>|rows;cols;values` (row-major, rounded to 6 decimals,
complex ones as `re;im` halves), then results as `case|values` (`;` between lists; complex results
as real parts then imaginary parts).
"""

import numpy as np
from scipy import linalg

rng = np.random.default_rng(57)


def out(name, a):
    a = np.asarray(a)
    if np.iscomplexobj(a):
        print(name + "|" + ",".join(repr(float(v)) for v in a.real.ravel()) + ";" + ",".join(repr(float(v)) for v in a.imag.ravel()))
    else:
        print(name + "|" + ",".join(repr(float(v)) for v in a.ravel()))


def matrix(name, a):
    a = np.round(a, 6)
    if np.iscomplexobj(a):
        print(f"matrix {name}|{a.shape[0]};{a.shape[1]};" + ",".join(repr(float(v)) for v in a.real.ravel()) + ";" + ",".join(repr(float(v)) for v in a.imag.ravel()))
    else:
        print(f"matrix {name}|{a.shape[0]};{a.shape[1]};" + ",".join(repr(float(v)) for v in a.ravel()))
    return a


n, m = 20, 4
A = matrix("A", rng.normal(size=(n, n)) / np.sqrt(n))
Big = matrix("Big", rng.normal(size=(n, n)) * 3.0)
Ac = matrix("Ac", (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2 * n))
B = matrix("B", rng.normal(size=(n, m)))
S = matrix("S", rng.normal(size=(15, 15)) / np.sqrt(15) + 2.0 * np.eye(15))
F = matrix("F", rng.normal(size=(n, 15)))
M = rng.normal(size=(n, n))
Q = matrix("Q", M @ M.T / n + np.eye(n))
Rm = rng.normal(size=(m, m))
R = matrix("R", Rm @ Rm.T / m + np.eye(m))
Stable = matrix("Stable", A - 1.5 * np.eye(n))
Contr = matrix("Contr", 0.8 * A / max(abs(linalg.eigvals(A))))
K = np.round(rng.normal(size=(n, n)) * 0.1, 6)
Qh = matrix("Qh", Q + 1j * (K - K.T))

out("expm A", linalg.expm(A))
out("expm Big", linalg.expm(Big))
out("expm Ac", linalg.expm(Ac))
out("sylvester", linalg.solve_sylvester(A, S, F))
out("continuous lyapunov", linalg.solve_continuous_lyapunov(Stable, Q))
out("discrete lyapunov", linalg.solve_discrete_lyapunov(Contr, Q))
out("continuous lyapunov complex", linalg.solve_continuous_lyapunov(Ac - 1.5 * np.eye(n), Qh))
out("care", linalg.solve_continuous_are(A, B, Q, R))
out("dare", linalg.solve_discrete_are(A, B, Q, R))
out("care complex", linalg.solve_continuous_are(Ac, B.astype(complex), Qh, R))
