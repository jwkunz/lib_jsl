"""SciPy references for release 2.10.0: rotation fitting and representations, Procrustes,
geometric slerp, Hausdorff distance and k-d tree extras (scipy.spatial), and Johnson's
algorithm, Yen's k shortest paths, bipartite matchings, structural rank and Laplacians
(scipy.sparse.csgraph). Writes geograph_reference.txt."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import procrustes, geometric_slerp, cKDTree
from scipy.spatial.distance import directed_hausdorff
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph as cg

out = []
def put(key, *lists):
    out.append(key + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))

def quat(r):
    q = r.as_quat()  # x, y, z, w with w >= 0 canonical
    return q if q[3] >= 0 else -q

rng = np.random.default_rng(210)
# align_vectors
b = rng.normal(size=(12, 3))
truth = R.from_rotvec([0.4, -1.1, 0.7])
a = truth.apply(b) + rng.normal(0, 0.05, (12, 3))
w = rng.uniform(0.5, 2, 12)
put("align a", a)
put("align b", b)
put("align w", w)
for name, kw in [("plain", {}), ("weighted", {"weights": w})]:
    r, rssd = R.align_vectors(a, b, **kw)
    put(f"align {name}", quat(r), [rssd])
winf = w.copy(); winf[3] = np.inf
r, rssd = R.align_vectors(a, b, weights=winf)
put("align inf", quat(r), [rssd])
r, rssd = R.align_vectors(a[:1], b[:1])
put("align single", quat(r), [rssd])
# mean
rots = R.from_rotvec(rng.normal(0, 0.6, (10, 3)) + [0.5, 0.2, -0.3])
mw = rng.uniform(0.2, 1, 10)
put("mean rotvecs", rots.as_rotvec())
put("mean w", mw)
put("mean", quat(rots.mean()), quat(rots.mean(weights=mw)))
# representations
rv = np.array([[1e-6, -2e-6, 3e-6], [0.3, 0.2, -0.1], [2.0, -1.0, 0.5], [0, 0, np.pi - 1e-9], [-1.5, 2.2, 0.4]])
put("rotvec in", rv)
put("rotvec quat", np.array([quat(R.from_rotvec(v)) for v in rv]))
put("rotvec back", np.array([R.from_rotvec(v).as_rotvec() for v in rv]))
put("mrp", np.array([R.from_rotvec(v).as_mrp() for v in rv]))
put("from mrp", np.array([quat(R.from_mrp(v / 3)) for v in rv]))
# procrustes
d1 = rng.normal(size=(15, 3))
d2 = d1 @ R.from_rotvec([1.0, 0.3, -0.2]).as_matrix().T * 2.5 + [1, -2, 0.5] + rng.normal(0, 0.1, (15, 3))
m1, m2, disp = procrustes(d1, d2)
put("procrustes d1", d1)
put("procrustes d2", d2)
put("procrustes", m2, [disp])
# geometric slerp in 4-D
s0 = np.array([1.0, 0, 0, 0]); s1 = np.array([0.2, 0.5, -0.3, 0.6]); s1 /= np.linalg.norm(s1)
t = np.linspace(0, 1, 7)
put("slerp end", s1)
put("slerp", geometric_slerp(s0, s1, t))
# Hausdorff
u = rng.normal(size=(200, 3)); v = rng.normal(size=(150, 3)) + 0.3
dh = directed_hausdorff(u, v)
put("hausdorff u", u)
put("hausdorff v", v)
put("hausdorff", [dh[0], dh[1], dh[2]], [directed_hausdorff(v, u)[0]])
# k-d tree extras
pts = rng.uniform(0, 1, (500, 3))
tree = cKDTree(pts)
pairs = np.array(sorted(tree.query_pairs(0.08)))
put("kd pts", pts)
put("kd pairs", pairs)
other = rng.uniform(0, 1, (300, 3))
put("kd other", other)
put("kd count", [tree.count_neighbors(cKDTree(other), r) for r in [0.05, 0.1, 0.2]])
ptree = cKDTree(pts, boxsize=1.0)
q = rng.uniform(0, 1, (20, 3))
dd, ii = ptree.query(q, k=4)
put("kd q", q)
put("kd periodic", dd, ii)
put("kd periodic pairs", [len(ptree.query_pairs(0.08))])

# Graphs: a random sparse directed graph with some negative edges but no negative cycles
# (potentials make every cycle non-negative).
n = 40
pot = rng.uniform(0, 5, n)
rows, cols, vals = [], [], []
for i in range(n):
    for j in rng.choice(n, 4, replace=False):
        if i != j:
            rows.append(i); cols.append(j)
            vals.append(rng.uniform(0.5, 3) + pot[j] - pot[i] - 0.5)  # can be negative
vals = np.array(vals)
# Ensure the reduced weights w + pot[i] - pot[j] stay positive: they are uniform(0, 2.5) by design.
G = csr_matrix((vals, (rows, cols)), shape=(n, n))
put("graph edges", np.array([rows, cols, vals]).T)
put("johnson", cg.johnson(G, directed=True))
Gp = csr_matrix((np.abs(vals) + 0.1, (rows, cols)), shape=(n, n))
put("yen", cg.yen(Gp, 0, 17, 8, directed=True))
put("yen undirected", cg.yen(Gp, 0, 17, 8, directed=False))
for name, kw in [("plain", {}), ("normed", {"normed": True}), ("out", {"use_out_degree": True})]:
    L, d = cg.laplacian(Gp.toarray(), return_diag=True, **kw)
    put(f"laplacian {name}", L, d)
# Bipartite
B = (rng.uniform(size=(30, 40)) < 0.07).astype(float)
br, bc = np.nonzero(B)
put("bip edges", np.array([br, bc]).T)
m = cg.maximum_bipartite_matching(csr_matrix(B), perm_type="column")
put("bip matching", [np.sum(m >= 0)], [cg.structural_rank(csr_matrix(B))])
W = rng.uniform(1, 10, (25, 35)) * (rng.uniform(size=(25, 35)) < 0.3)
wr, wc = np.nonzero(W)
put("mw edges", np.array([wr, wc, W[wr, wc]]).T)
ri, ci = cg.min_weight_full_bipartite_matching(csr_matrix(W))
put("mw", ri, ci, [W[ri, ci].sum()])
ri, ci = cg.min_weight_full_bipartite_matching(csr_matrix(W), maximize=True)
put("mw max", ri, ci, [W[ri, ci].sum()])
open("geograph_reference.txt", "w").write("\n".join(out) + "\n")
