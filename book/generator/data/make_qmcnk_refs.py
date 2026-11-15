"""Reference results for releases 2.3.0 and 2.4.0: scipy.stats.qmc (Sobol, Halton, Latin
hypercube, discrepancy), scipy.optimize.newton_krylov and scipy.sparse.linalg.lgmres, from
SciPy 1.17.1.

    python book/generator/data/make_qmcnk_refs.py > book/generator/data/qmcnk_reference.txt
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import LinearOperator, splu
from scipy.stats import qmc


def out(name, *lists):
    print(name + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))


# --- Unscrambled sequences (deterministic; compared point by point) ---
full = qmc.Sobol(21201, scramble=False).random(64)
cols = list(range(0, 21201, 211)) + [21200]
out("sobol full", full[:, cols])
out("halton plain", qmc.Halton(20, scramble=False).random(65)[1:])

# --- Discrepancies of deterministic sets ---
sets = {
    "sobol 128x5": qmc.Sobol(5, scramble=False).random(128),
    "halton 100x3": qmc.Halton(3, scramble=False).random(101)[1:],
    "grid 64x2": np.array([[(i + 0.5) / 8, (j + 0.5) / 8] for i in range(8) for j in range(8)]),
    "formula 50x6": np.array([[(np.sin(i * 1.7 + k * 0.3) + 1) / 2 for k in range(6)] for i in range(50)]),
}
for name, pts in sets.items():
    out(f"disc {name}", [qmc.discrepancy(pts, method=m) for m in ("CD", "WD", "MD", "L2-star")])


# --- Randomised samplers: statistics over independent replicates ---
def gfun(x):
    a = np.arange(1, x.shape[1] + 1)
    return np.prod((np.abs(4 * x - 2) + a) / (1 + a), axis=1)


d, reps = 5, 50
rng = np.random.default_rng(2024)
samplers = {
    "mc": lambda n: rng.random((n, d)),
    "lhs": lambda n: qmc.LatinHypercube(d, rng=rng).random(n),
    "sobol": lambda n: qmc.Sobol(d, rng=rng).random(n),
    "halton": lambda n: qmc.Halton(d, rng=rng).random(n),
}
for name, draw in samplers.items():
    rmse = []
    for m in range(4, 13):
        errs = [gfun(draw(2 ** m)).mean() - 1.0 for _ in range(reps)]
        rmse.append(np.sqrt(np.mean(np.square(errs))))
    cd = [qmc.discrepancy(draw(256)) for _ in range(reps)]
    out(f"rmse {name}", rmse)
    out(f"cd {name}", [np.mean(cd), np.std(cd)])


# --- Newton-Krylov on the Bratu problem ---
def bratu(m, lam):
    h2 = 1.0 / (m + 1) ** 2

    def F(u):
        U = np.pad(u.reshape(m, m), 1)
        return ((U[:-2, 1:-1] + U[2:, 1:-1] + U[1:-1, :-2] + U[1:-1, 2:] - 4 * U[1:-1, 1:-1]) / h2 + lam * np.exp(U[1:-1, 1:-1])).ravel()

    return F


def laplacian(m):
    h2 = 1.0 / (m + 1) ** 2
    t = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(m, m))
    i = sp.identity(m)
    return ((sp.kron(t, i) + sp.kron(i, t)) / h2).tocsc()


for m in (16, 32, 64):
    for method, precond in (("gmres", False), ("bicgstab", False), ("gmres", True), ("lgmres", False)):
        F = bratu(m, 6.0)
        count = [0]
        hist = []

        def G(u):
            count[0] += 1
            return F(u)

        kw = {}
        if precond:
            lu = splu(laplacian(m))
            kw["inner_M"] = LinearOperator((m * m, m * m), matvec=lu.solve)
        try:
            x = newton_krylov(G, np.zeros(m * m), method=method, callback=lambda x, f: hist.append(np.abs(f).max()), **kw)
            ok = 1
        except Exception:
            x, ok = np.full(m * m, np.nan), 0
        name = f"nk {m} {method}{' lu' if precond else ''}"
        out(name, [len(hist), count[0], ok, np.abs(F(x)).max(), x.max()], [np.abs(F(np.zeros(m * m))).max()] + hist)


# --- LGMRES on nonsymmetric systems ---
from scipy.sparse.linalg import lgmres


def convection(m, g):
    t = sp.diags([-1 - g, 2.0, -1 + g], [-1, 0, 1], shape=(m, m))
    i = sp.identity(m)
    return (sp.kron(t, i) + sp.kron(i, t)).tocsr()


for g, outer_k in ((0.5, 3), (0.9, 3), (0.5, 0), (0.9, 0)):
    a = convection(30, g)
    b = 1 + 0.5 * np.sin(0.37 * np.arange(900))
    count = [0]

    def mv(x, a=a):
        count[0] += 1
        return a @ x

    op = LinearOperator(a.shape, matvec=mv, dtype=float)
    x, info = lgmres(op, b, rtol=1e-8, atol=0, outer_k=outer_k)
    out(f"lgmres {g:g} {outer_k}", [count[0], info, np.linalg.norm(a @ x - b) / np.linalg.norm(b)], x)
