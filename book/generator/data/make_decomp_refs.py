"""Reference results for release 2.5.0: scipy.linalg's qr (with and without pivoting), rq,
hessenberg, schur (real), ldl, polar, subspace_angles and funm, from SciPy 1.17.1.

    python book/generator/data/make_decomp_refs.py > book/generator/data/decomp_reference.txt

The matrices are built by formula; the generator builds the same ones.
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


def real(m, n, seed):
    return np.array([[np.sin(seed + 1.3 * i + 0.7 * j * j + 0.37 * i * j) + (2.0 if i == j else 0.0) for j in range(n)] for i in range(m)])


def cplx(m, n, seed):
    return real(m, n, seed) + 1j * np.array([[np.cos(seed + 0.4 * i * i - 0.9 * j) for j in range(n)] for i in range(m)])


mats = {
    "square": real(6, 6, 0.0),
    "tall": real(7, 4, 1.0),
    "wide": real(4, 7, 2.0),
    "complex": cplx(5, 5, 3.0),
    "complex wide": cplx(3, 5, 4.0),
}
# Rank 3 in 6 columns: columns 3..5 are combinations of 0..2.
rd = real(6, 3, 5.0)
mats["rank deficient"] = np.hstack([rd, rd @ np.array([[1.0, 0.5, -2.0], [0.0, 1.0, 1.0], [3.0, -1.0, 0.5]])])

for name, a in mats.items():
    for mode in ("full", "economic"):
        q, r = linalg.qr(a, mode=mode)
        cout(f"qr {name} {mode}", q, r)
        q, r, p = linalg.qr(a, mode=mode, pivoting=True)
        cout(f"qrp {name} {mode}", q, r, p)
        r, q = linalg.rq(a, mode=mode)
        cout(f"rq {name} {mode}", r, q)
    for side in ("right", "left"):
        u, p = linalg.polar(a, side=side)
        cout(f"polar {name} {side}", u, p)

for name in ("square", "complex"):
    h, q = linalg.hessenberg(mats[name], calc_q=True)
    cout(f"hess {name}", h, q)
big = real(30, 30, 6.0)
h, q = linalg.hessenberg(big, calc_q=True)
cout("hess big", h, q)

# Real Schur: eigenvalues of the diagonal blocks and the reconstruction.
for name, a in (("square", mats["square"]), ("big", big)):
    t, z = linalg.schur(a, output="real")
    out(f"schur {name}", np.sort_complex(np.linalg.eigvals(t)).real, np.sort_complex(np.linalg.eigvals(t)).imag)


def sym(n, seed, small_diag):
    s = np.array([[np.sin(seed + 0.9 * (i + j) + 0.3 * i * j) for j in range(n)] for i in range(n)])
    s = s + s.T
    np.fill_diagonal(s, small_diag * np.cos(np.arange(n) + seed))
    return s


def herm(n, seed, small_diag):
    s = sym(n, seed, small_diag) + 1j * np.array([[np.cos(seed + 0.5 * i - 0.8 * j) for j in range(n)] for i in range(n)])
    s = (s + s.conj().T) / 2
    np.fill_diagonal(s, small_diag * np.cos(np.arange(n) + seed))
    return s


ldl_mats = {
    "spd": sym(6, 0.0, 0.0) + 8 * np.eye(6),
    "indefinite": sym(7, 1.0, 0.05),
    "zero diagonal": sym(6, 2.0, 0.0),
    "hermitian": herm(6, 3.0, 0.1),
    "complex symmetric": sym(5, 4.0, 0.02) + 1j * sym(5, 5.0, 0.03),
}
for name, a in ldl_mats.items():
    herm_flag = name != "complex symmetric"
    for lower in (True, False):
        l, d, p = linalg.ldl(a, lower=lower, hermitian=herm_flag)
        cout(f"ldl {name} {'lower' if lower else 'upper'}", l, d, p)

# Subspace angles.
A = real(8, 3, 7.0)
B = real(8, 4, 8.0)
out("angles general", linalg.subspace_angles(A, B))
Bnear = A + 1e-7 * real(8, 3, 9.0)
out("angles near", linalg.subspace_angles(A, Bnear))
Ac = cplx(6, 2, 10.0)
Bc = cplx(6, 3, 11.0)
out("angles complex", linalg.subspace_angles(Ac, Bc))

# funm.
for name, f in (("exp", np.exp), ("sin", np.sin), ("cube", lambda x: x ** 3)):
    F, err = linalg.funm(mats["square"], f, disp=False)
    cout(f"funm square {name}", F)
    out(f"funm square {name} err", [err])
F, err = linalg.funm(mats["complex"], np.exp, disp=False)
cout("funm complex exp", F)
