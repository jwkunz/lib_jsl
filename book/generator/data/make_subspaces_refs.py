"""Reference results for the entry on matrix functions, subspaces and norms, from `scipy.linalg`
and `numpy.linalg` (the book used SciPy 1.17.1 and NumPy 2).

    python book/generator/data/make_subspaces_refs.py > book/generator/data/subspaces_reference.txt

Matrices are written as `matrix <name>|rows;cols;re;im`, results as `case|values` (complex
results as real parts then imaginary parts).
"""

import numpy as np
from scipy import linalg

rng = np.random.default_rng(321)


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def matrix(name, a, digits=6):
    a = np.asarray(a, dtype=complex)
    a = np.round(a, digits) if digits is not None else a
    print(f"matrix {name}|{a.shape[0]};{a.shape[1]};" + ",".join(repr(float(v)) for v in a.real.ravel()) + ";" + ",".join(repr(float(v)) for v in a.imag.ravel()))
    return a if np.iscomplexobj(a) and np.any(a.imag) else a.real


M = rng.normal(size=(12, 12))
spd = matrix("spd", M @ M.T / 12 + np.eye(12))
gen = matrix("gen", rng.normal(size=(12, 12)) + 4 * np.eye(12))
cplx = matrix("cplx", rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10)) + 4 * np.eye(10))
for name, a in (("spd", spd), ("gen", gen), ("cplx", cplx)):
    s = linalg.sqrtm(a)
    out(f"sqrtm {name}", np.real(s), np.imag(s))
    l = linalg.logm(a)
    out(f"logm {name}", np.real(l), np.imag(l))
    f = linalg.fractional_matrix_power(a, 0.3)
    out(f"fmp {name}", np.real(f), np.imag(f))

# The low-rank matrix is written at full precision: rounding would make it full rank.
low = matrix("lowrank", rng.normal(size=(15, 4)) @ rng.normal(size=(4, 9)), digits=None)
out("pinv lowrank", linalg.pinv(low))
out("rank lowrank", [np.linalg.matrix_rank(low)])
ns = linalg.null_space(low)
out("null projector", ns @ ns.T)
o = linalg.orth(low)
out("orth projector", o @ o.T)
tall = matrix("tall", rng.normal(size=(20, 7)))
out("pinv tall", linalg.pinv(tall))
for name, a in (("gen", gen), ("lowrank", low), ("tall", tall)):
    out(f"norms {name}", [np.linalg.norm(a, o) for o in ("fro", "nuc", 1, -1, 2, -2, np.inf, -np.inf)])
for name, a in (("gen", gen), ("spd", spd)):
    out(f"cond {name}", [np.linalg.cond(a, p) for p in (2, -2, 1, np.inf, "fro")])
