"""Reference results for release 2.2.0: eigs and its shift-invert mode, from SciPy 1.17.1.

    python book/generator/data/make_eigs_refs.py > book/generator/data/eigs_reference.txt
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, LinearOperator


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def scattered(n):
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(np.sin(i * 0.37))
        for t in range(1, 5):
            rows.append(i); cols.append((i * 37 + 11 * t + 7 * t * t) % n); vals.append(np.sin(i + 3.0 * t + 1.0))
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def ramp(n):
    main = 2.0 * np.arange(n) / n
    return sp.diags([main, np.ones(n - 1), -np.ones(n - 1)], [0, 1, -1]).tocsr()


def convection(m, gamma):
    t = sp.diags([-1 - gamma, 2.0, -1 + gamma], [-1, 0, 1], shape=(m, m))
    i = sp.identity(m)
    return (sp.kron(t, i) + sp.kron(i, t)).tocsr()


def counted(a):
    count = [0]
    def mv(x):
        count[0] += 1
        return a @ x
    return LinearOperator(a.shape, matvec=mv, dtype=float), count


def emit(name, vals, nmv):
    out(name, np.real(vals), np.imag(vals), [nmv])


matrices = {"scattered": scattered(300), "ramp": ramp(400), "convection": convection(30, 0.2)}
# Full spectra (the exact values for the convection operator are known in closed form).
for mat in ("scattered", "ramp"):
    emit(f"spectrum {mat}", np.linalg.eigvals(matrices[mat].toarray()), 0)
cases = [("scattered", "LM"), ("scattered", "LR"), ("scattered", "SR"), ("scattered", "LI"),
         ("ramp", "LM"), ("ramp", "LR"), ("ramp", "LI"), ("convection", "LM"), ("convection", "SR")]
for mat, which in cases:
    a = matrices[mat]
    op, count = counted(a)
    vals = eigs(op, k=6, which=which, tol=0, return_eigenvectors=False, v0=np.ones(a.shape[0]))
    emit(f"eigs {mat} {which}", vals, count[0])

# Shift-invert: the eigenvalues nearest sigma.
for mat, sigma in [("scattered", 0.3), ("convection", 1.0)]:
    a = matrices[mat]
    vals = eigs(a, k=6, sigma=sigma, tol=0, return_eigenvectors=False, v0=np.ones(a.shape[0]))
    emit(f"shift {mat} {sigma:g}", vals, 0)
