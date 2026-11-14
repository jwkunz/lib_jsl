"""Reference results for sparse solvers at scale (release 1.18.0): fill under SuperLU's
orderings, GMRES with SciPy's `spilu` preconditioner, `expm_multiply` and shift-invert `eigsh`,
from SciPy 1.17.1.

    python book/generator/data/make_sparse2_refs.py > book/generator/data/sparse2_reference.txt

The matrices are built by formula (the generator builds the same ones); the random permutations
are written as `perm <name>|indices`.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def grid(m, east, west, north, south, centre=4.0):
    """The 5-point stencil on an m × m grid, row-major (as in make_sparse_refs.py)."""
    n = m * m
    rows, cols, vals = [], [], []
    for i in range(m):
        for j in range(m):
            k = i * m + j
            for (cond, kk, v) in ((True, k, centre), (j + 1 < m, k + 1, east), (j > 0, k - 1, west),
                                  (i + 1 < m, k + m, north), (i > 0, k - m, south)):
                if cond:
                    rows.append(k)
                    cols.append(kk)
                    vals.append(v)
    return sparse.csr_array((vals, (rows, cols)), shape=(n, n))


def fill(a, spec):
    lu = sla.splu(a.tocsc(), permc_spec=spec)
    return lu.L.nnz + lu.U.nnz


# Fill of shuffled Laplacians and of a convection-diffusion matrix.
for m in (30, 60):
    perm = np.random.default_rng(100 + m).permutation(m * m)
    print(f"perm {m}|" + ",".join(str(int(v)) for v in perm))
    a = grid(m, -1, -1, -1, -1)[perm][:, perm]
    out(f"fill laplacian {m}", [fill(a, s) for s in ("NATURAL", "MMD_AT_PLUS_A", "COLAMD")])
cd = grid(40, -1 + 0.3, -1 - 0.3, -1 + 0.1, -1 - 0.1)
out("fill convdiff 40", [fill(cd, s) for s in ("NATURAL", "MMD_AT_PLUS_A", "COLAMD")])

# GMRES(30) with incomplete LU preconditioning, rtol 1e-8.
n = 40 * 40
b = np.array([1.0 + 0.5 * np.sin(0.37 * k) for k in range(n)])
hard = grid(40, -1 + 1.4, -1 - 1.4, -1 + 0.9, -1 - 0.9)
for name, a in (("convdiff", cd), ("hard", hard)):
    def run(M=None):
        it = [0]
        x, info = sla.gmres(a, b, rtol=1e-8, restart=30, M=M, maxiter=2000,
                            callback=lambda _: it.__setitem__(0, it[0] + 1), callback_type="pr_norm")
        return it[0], info, np.linalg.norm(b - a @ x) / np.linalg.norm(b)
    rows = []
    rows.append(run())
    for drop in (1e-2, 1e-4):
        ilu = sla.spilu(a.tocsc(), drop_tol=drop)
        M = sla.LinearOperator(a.shape, ilu.solve)
        rows.append(run(M) + (ilu.L.nnz + ilu.U.nnz,))
    out(f"gmres {name} none", rows[0])
    out(f"gmres {name} spilu 1e-2", rows[1])
    out(f"gmres {name} spilu 1e-4", rows[2])

# expm_multiply.
lap = grid(30, -1, -1, -1, -1)
b2 = np.array([np.cos(0.11 * k) for k in range(900)])
for t in (0.01, 1.0, 10.0, 40.0):
    out(f"expm heat {t:g}", sla.expm_multiply(-t * lap, b2))
cd30 = grid(30, -1 + 0.3, -1 - 0.3, -1 + 0.1, -1 - 0.1)
for t in (0.5, 3.0):
    out(f"expm convdiff {t:g}", sla.expm_multiply(-t * cd30, b2))

# Shift-invert eigsh on the 30 × 30 Laplacian.
for sigma in (2.0, 5.5):
    out(f"eigsh sigma {sigma:g}", np.sort(sla.eigsh(lap, k=6, sigma=sigma, return_eigenvectors=False)))
