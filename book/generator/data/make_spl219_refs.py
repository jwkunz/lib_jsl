"""SciPy references for release 2.19.0: periodic smoothing splines (make_splrep with
bc_type='periodic'), smoothing surfaces (bisplrep/bisplev) and stiff ODEs with sparse
Jacobians. Writes spl219_reference.txt."""
import numpy as np
from scipy.interpolate import make_splrep, bisplrep, bisplev
from scipy.interpolate._fitpack_repro import generate_knots

out = []
def put(key, *lists):
    out.append(key + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))

rng = np.random.default_rng(219)
x = np.sort(rng.uniform(0, 2 * np.pi, 60)); x[0] = 0.0; x[-1] = 2 * np.pi
y = np.sin(x) + 0.5 * np.cos(3 * x) + 0.1 * rng.normal(size=x.size); y[-1] = y[0]
xe = np.linspace(0, 2 * np.pi, 101)
put("periodic data", x, y, xe)
for k in [3, 2]:
    for s in [0.0, 0.3, 0.6, 2.0]:
        spl = make_splrep(x, y, k=k, s=s, bc_type="periodic")
        put(f"periodic {k} {s}", spl.t, spl(xe))
        if s > 0:
            put(f"periodic knots {k} {s}", [len(t) for t in generate_knots(x, y, k=k, s=s, bc_type="periodic")])
# Scattered surfaces.
xs = rng.uniform(0, 1, 400); ys = rng.uniform(0, 1, 400)
sigma = 0.05
zs = np.sin(6 * xs) * np.cos(4 * ys) + sigma * rng.normal(size=400)
gx = np.linspace(0, 1, 21)
put("surface data", xs, ys, zs, gx)
tx = np.r_[[0.0] * 4, 0.25, 0.5, 0.75, [1.0] * 4]
lsq = bisplrep(xs, ys, zs, task=-1, tx=tx, ty=tx, kx=3, ky=3)
put("surface lsq", lsq[2], bisplev(gx, gx, lsq))
tck, fp, ier, msg = bisplrep(xs, ys, zs, w=np.full(400, 1 / sigma), kx=3, ky=3, full_output=1)
put("surface smooth", [fp, len(tck[0]), len(tck[1]), ier], bisplev(gx, gx, tck))

# Stiff ODEs with sparse Jacobians: a reaction-diffusion chain.
from scipy.integrate import solve_ivp
import scipy.sparse as sp
n = 200; D = 400.0
def rd(t, y):
    lap = -2 * y; lap[1:] += y[:-1]; lap[:-1] += y[1:]
    return D * lap - y ** 3
y0 = np.sin(np.pi * np.arange(1, n + 1) / (n + 1))
pattern = sp.diags([1, 1, 1], [-1, 0, 1], shape=(n, n))
for method in ["Radau", "LSODA"]:
    for sparsity in [None, pattern]:
        r = solve_ivp(rd, (0, 0.5), y0, method=method, rtol=1e-8, atol=1e-10, jac_sparsity=sparsity)
        put(f"ode {method} {sparsity is not None}", r.y[:, -1], [r.nfev, r.njev, r.nlu, r.t.size - 1])
open("data/spl219_reference.txt", "w").write("\n".join(out) + "\n")
