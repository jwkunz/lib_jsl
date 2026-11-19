"""Reference results for release 2.6.0: generalised eigenproblems (scipy.linalg.eigh(a, b),
eig(a, b), qz), structured matrices and their solvers (toeplitz, hankel, circulant, hadamard,
block_diag, companion, pascal, invpascal, hilbert, invhilbert, leslie, dft, convolution_matrix,
fiedler; solve_toeplitz, solve_circulant, solve_banded, solveh_banded), from SciPy 1.17.1.

    python book/generator/data/make_genstruct_refs.py > book/generator/data/genstruct_reference.txt
"""

import numpy as np
from scipy import linalg


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def cout(name, *arrays):
    lists = []
    for a in arrays:
        a = np.asarray(a)
        lists += [a.real, a.imag]
    out(name, *lists)


def pencil(n, seed):
    a = np.array([[np.sin(seed + 1.3 * i + 0.7 * j * j + 0.37 * i * j) for j in range(n)] for i in range(n)])
    b = np.array([[np.cos(seed + 0.5 * i - 0.9 * j + 0.21 * i * j) + (3.0 if i == j else 0.0) for j in range(n)] for i in range(n)])
    return a, b


# Symmetric-definite pencils.
a, b = pencil(7, 1.0)
sa, sb = a + a.T, b.T @ b
w, v = linalg.eigh(sa, sb)
out("eigh real", w, v)
ca, cb = pencil(6, 2.0)
ha = (ca + 1j * cb) + (ca + 1j * cb).conj().T
hb = (cb + 1j * ca).conj().T @ (cb + 1j * ca) + 6 * np.eye(6)
w, v = linalg.eigh(ha, hb)
cout("eigh complex", w, v)

# General pencils.
for n, seed in ((6, 3.0), (12, 4.0)):
    a, b = pencil(n, seed)
    w, v = linalg.eig(a, b)
    cout(f"eig {n}", w, v)
a, b = pencil(5, 5.0)
za, zb = a + 1j * b.T, b + 0.3j * a.T
w, v = linalg.eig(za, zb)
cout("eig complex", w, v)
# A singular B: one infinite eigenvalue.
a = np.array([[2.0, 1.0, 0.5], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]])
b = np.diag([1.0, 1.0, 0.0])
w = linalg.eig(a, b, right=False)
cout("eig singular", np.where(np.isinf(w.real), 1e300, w))
# QZ (complex output): the generalised eigenvalues from the diagonals.
a, b = pencil(8, 6.0)
AA, BB, Q, Z = linalg.qz(a, b, output="complex")
cout("qz 8", np.diag(AA) / np.diag(BB))

# Structured constructors.
c = [1.0, 2.5, -1.0, 4.0]
r = [9.0, -3.0, 0.5]
out("toeplitz", linalg.toeplitz(c, r))
cc = [1 + 1j, 2 - 1j, 0.5j]
cout("toeplitz hermitian", linalg.toeplitz(cc))
out("hankel", linalg.hankel(c, r))
out("hankel zeros", linalg.hankel(c))
out("circulant", linalg.circulant(c))
out("hadamard 16", linalg.hadamard(16))
out("block_diag", linalg.block_diag([[1, 2], [3, 4]], [[5]], [[6, 7, 8]]))
out("companion", linalg.companion([2.0, -3.0, 0.5, 4.0]))
for kind in ("symmetric", "lower", "upper"):
    out(f"pascal {kind}", linalg.pascal(6, kind=kind, exact=False))
    out(f"invpascal {kind}", linalg.invpascal(6, kind=kind, exact=False))
out("hilbert", linalg.hilbert(5))
out("invhilbert", linalg.invhilbert(8))
out("leslie", linalg.leslie([0.1, 2.0, 1.0, 0.1], [0.2, 0.8, 0.7]))
for scale in (None, "sqrtn", "n"):
    cout(f"dft {scale}", linalg.dft(5, scale=scale))
for mode in ("full", "same", "valid"):
    out(f"convolution {mode}", linalg.convolution_matrix([1.0, -2.0, 3.0, 0.5], 6, mode=mode))
    out(f"convolution short {mode}", linalg.convolution_matrix([1.0, -2.0, 3.0, 0.5, 2.0, 1.0], 4, mode=mode))
out("fiedler", linalg.fiedler([1.0, 4.0, 12.0, 45.0, 77.0]))

# Fast solvers.
n = 40
c = 4.0 / (1.0 + np.arange(n)); c[0] += 2.0
r = -0.7 / (1.0 + np.arange(n) ** 2); r[0] = c[0]
bvec = np.sin(0.7 * np.arange(n)) + 1.0
out("solve_toeplitz", linalg.solve_toeplitz((c, r), bvec))
ch = c + 1j * np.r_[0.0, 0.3 / np.arange(1, n)]
cout("solve_toeplitz hermitian", linalg.solve_toeplitz(ch, bvec - 1j * bvec))
out("solve_circulant", linalg.solve_circulant(c, bvec))
cout("solve_circulant complex", linalg.solve_circulant(ch, bvec + 0.5j))
l, u = 3, 2
ab = np.array([[np.sin(0.37 * (7 * (i + 1) + 3 * j)) + (0.01 if i == u else 0.0) for j in range(n)] for i in range(l + u + 1)])
out("solve_banded", linalg.solve_banded((l, u), ab, bvec))
w = 2
upper = np.array([[(6.0 if k == w else 1.0 / (1 + j + (j - (w - k)))) if j >= w - k else 0.0 for j in range(n)] for k in range(w + 1)])
out("solveh_banded", linalg.solveh_banded(upper, bvec))
