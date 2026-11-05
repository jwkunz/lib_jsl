"""Reference results for the linear algebra chapters, from SciPy's `scipy.linalg` (LAPACK through
OpenBLAS; the book used SciPy 1.17.1).

    python book/generator/data/make_linear_algebra_refs.py > book/generator/data/linear_algebra_reference.txt

Matrices are written first, as `matrix <name>|rows;cols;values` (row-major, rounded to 6 decimals,
complex ones as `re;im` halves), then results as `case|values` (`;` between lists).
"""

import numpy as np
from scipy import linalg

rng = np.random.default_rng(31)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def matrix(name, a):
    a = np.round(a, 6)
    if np.iscomplexobj(a):
        print(f"matrix {name}|{a.shape[0]};{a.shape[1]};" + ",".join(repr(float(v)) for v in a.real.ravel()) + ";" + ",".join(repr(float(v)) for v in a.imag.ravel()))
    else:
        print(f"matrix {name}|{a.shape[0]};{a.shape[1]};" + ",".join(repr(float(v)) for v in a.ravel()))
    return a


A = matrix("A", rng.normal(size=(40, 40)))
b = matrix("b", rng.normal(size=(40, 1)))[:, 0]
B = rng.normal(size=(30, 30))
S = matrix("S", B @ B.T + 30 * np.eye(30))
H = rng.normal(size=(30, 30))
H = matrix("H", H + H.T)
N = matrix("N", rng.normal(size=(25, 25)))
R = matrix("R", rng.normal(size=(50, 20)))
c = matrix("c", rng.normal(size=(50, 1)))[:, 0]
# Integer factors keep the rank exactly 12 after rounding.
D = matrix("D", rng.integers(-3, 4, size=(50, 12)).astype(float) @ rng.integers(-3, 4, size=(12, 20)).astype(float))
Hc = rng.normal(size=(20, 20)) + 1j * rng.normal(size=(20, 20))
Hc = matrix("Hc", Hc + Hc.conj().T)
Ac = matrix("Ac", rng.normal(size=(20, 20)) + 1j * rng.normal(size=(20, 20)))
bc = matrix("bc", rng.normal(size=(20, 1)) + 1j * rng.normal(size=(20, 1)))[:, 0]

out("solve", linalg.solve(A, b))
out("inv", linalg.inv(A))
out("det", [linalg.det(A)])
out("cholesky", linalg.cholesky(S, lower=True))
w, v = linalg.eigh(H)
out("eigh", w, np.abs(v))
e = linalg.eigvals(N)
e = e[np.lexsort((e.imag, e.real))]
out("eigvals", e.real, e.imag)
q, r = linalg.qr(R, mode="economic")
out("qr r", np.abs(r))
u, s, vt = linalg.svd(R)
out("svd", s, np.abs(u[:, :20]), np.abs(vt))
out("lstsq", linalg.lstsq(R, c)[0])
x, _, rank, sv = linalg.lstsq(D, c)
out("lstsq rank deficient", x, [rank])
w, _ = linalg.eigh(Hc)
out("eigh complex", w)
x = linalg.solve(Ac, bc)
out("solve complex", x.real, x.imag)
