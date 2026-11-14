"""Reference results for the entry on sparse matrices and iterative solvers, from `scipy.sparse`
and `scipy.sparse.linalg` (the book used SciPy 1.17.1).

    python book/generator/data/make_sparse_refs.py > book/generator/data/sparse_reference.txt

The square test matrices are built by formula (the generator builds the same ones); the random
rectangular matrix is written as `triplets rect|nrows;ncols;rows;cols;values`. Results are
`case|values`, iteration counts as `iters <solver> <matrix>|count`.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla
from scipy.sparse.csgraph import reverse_cuthill_mckee

M = 30  # grid side: n = 900


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


def grid(m, east, west, north, south, centre=4.0, extra=None):
    """The 5-point stencil on an m × m grid, row-major, Dirichlet boundaries."""
    n = m * m
    a = sparse.lil_array((n, n))
    for i in range(m):
        for j in range(m):
            k = i * m + j
            a[k, k] = centre + (extra(k) if extra else 0.0)
            if j + 1 < m:
                a[k, k + 1] = east
            if j > 0:
                a[k, k - 1] = west
            if i + 1 < m:
                a[k, k + m] = north
            if i > 0:
                a[k, k - m] = south
    return a.tocsr()


poisson = grid(M, -1, -1, -1, -1)
# Convection-diffusion by central differences, cell Péclet numbers 0.3 (x) and 0.1 (y).
convdiff = grid(M, -1 + 0.3, -1 - 0.3, -1 + 0.1, -1 - 0.1)
# A diffusion operator with a strongly varying diagonal, where Jacobi preconditioning pays.
varied = grid(M, -1, -1, -1, -1, extra=lambda k: 50.0 * ((k * 13) % 17) / 17.0)
n = M * M
b = np.array([1.0 + 0.5 * np.sin(0.37 * k) for k in range(n)])

for name, a in (("poisson", poisson), ("convdiff", convdiff), ("varied", varied)):
    out(f"spsolve {name}", sla.spsolve(a.tocsc(), b))

rtol = 1e-8


def count(solver, a, **kw):
    it = [0]

    def cb(_):
        it[0] += 1

    x, info = solver(a, b, rtol=rtol, callback=cb, **kw)
    return x, it[0], info


for name, a in (("poisson", poisson), ("varied", varied)):
    for sname, solver in (("cg", sla.cg), ("minres", sla.minres)):
        x, it, info = count(solver, a)
        out(f"x {sname} {name}", x)
        out(f"iters {sname} {name}", [it])
jac = sparse.diags_array(1.0 / varied.diagonal())
x, it, info = count(sla.cg, varied, M=jac)
out("iters pcg varied", [it])
for name, a in (("poisson", poisson), ("convdiff", convdiff)):
    x, it, info = count(sla.bicgstab, a)
    out(f"x bicgstab {name}", x)
    out(f"iters bicgstab {name}", [it])
    x, it, info = count(sla.gmres, a, restart=30, callback_type="pr_norm")
    out(f"x gmres {name}", x)
    out(f"iters gmres {name}", [it])

rng = np.random.default_rng(2026)
rect = sparse.random_array((200, 80), density=0.05, rng=rng, format="coo")
rect = sparse.coo_array((np.round(rect.data - 0.5, 6), (rect.row, rect.col)), shape=rect.shape)
print("triplets rect|200;80;" + ",".join(str(int(v)) for v in rect.row) + ";" + ",".join(str(int(v)) for v in rect.col) + ";" + ",".join(repr(float(v)) for v in rect.data))
rect = rect.tocsr()
brect = np.array([np.cos(0.21 * k) for k in range(200)])
res = sla.lsqr(rect, brect, atol=1e-10, btol=1e-10)
out("x lsqr rect", res[0])
out("iters lsqr rect", [res[2]])
out("lstsq rect", np.linalg.lstsq(rect.toarray(), brect, rcond=None)[0])

for which in ("LA", "SA", "LM"):
    vals = sla.eigsh(poisson, k=4, which=which, return_eigenvectors=False)
    out(f"eigsh {which}", np.sort(vals))
out("svds rect", np.sort(sla.svds(rect, k=4, return_singular_vectors=False))[::-1])
out("svd rect dense", np.linalg.svd(rect.toarray(), compute_uv=False)[:4])

# Bandwidth and LU fill (nnz of L + U) for the 2-D Laplacian with its rows in a random order.
perm = np.random.default_rng(7).permutation(n)
print("perm|" + ",".join(str(int(v)) for v in perm))
shuffled = poisson[perm][:, perm].tocsr()
rcm = reverse_cuthill_mckee(shuffled, symmetric_mode=True)


def bw(a):
    c = a.tocoo()
    return int(np.max(np.abs(c.row - c.col)))


out("bandwidth", [bw(shuffled), bw(shuffled[rcm][:, rcm])])
for spec in ("NATURAL", "MMD_AT_PLUS_A", "COLAMD"):
    lu = sla.splu(shuffled.tocsc(), permc_spec=spec)
    out(f"fill {spec}", [lu.L.nnz + lu.U.nnz])
lu = sla.splu(shuffled[rcm][:, rcm].tocsc(), permc_spec="NATURAL")
out("fill RCM then NATURAL", [lu.L.nnz + lu.U.nnz])
