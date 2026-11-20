"""SciPy references for release 2.17.0: ConvexHull, Delaunay and Voronoi in several dimensions,
HalfspaceIntersection, SphericalVoronoi and LinearNDInterpolator in 3-D and 4-D (all Qhull).
Writes geo217_reference.txt."""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi, HalfspaceIntersection, SphericalVoronoi
from scipy.interpolate import LinearNDInterpolator

out = []
def put(key, *lists):
    out.append(key + "|" + ";".join(",".join(repr(float(v)) for v in np.ravel(l)) for l in lists))

rng = np.random.default_rng(217)
for d, n in [(2, 200), (3, 200), (4, 150), (5, 120), (6, 80)]:
    p = rng.normal(size=(n, d))
    h = ConvexHull(p)
    put(f"hull {d}", p, [h.volume, h.area, len(h.simplices)], np.sort(h.vertices))
for d, n in [(2, 60), (3, 50), (4, 40)]:
    p = rng.uniform(size=(n, d))
    t = Delaunay(p)
    s = np.sort(t.simplices, axis=1)
    s = s[np.lexsort(s.T[::-1])]
    q = rng.uniform(0.1, 0.9, size=(20, d))
    found = t.find_simplex(q)
    fs = np.where(found[:, None] >= 0, np.sort(t.simplices[found], axis=1), -1)
    put(f"delaunay {d}", p, s, q, fs)
for d, n in [(2, 40), (3, 30)]:
    p = rng.uniform(size=(n, d))
    v = Voronoi(p)
    verts = v.vertices[np.lexsort(v.vertices.T[::-1])]
    rp = np.sort(v.ridge_points, axis=1)
    rp = rp[np.lexsort(rp.T[::-1])]
    put(f"voronoi {d}", p, verts, rp)
# Half-spaces: tangent planes to the unit sphere at random directions (3-D), plus redundant ones.
u = rng.normal(size=(30, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
hs = np.hstack([u, -np.ones((30, 1))])
hs = np.vstack([hs, np.hstack([u[:5], -3 * np.ones((5, 1))])])
hi = HalfspaceIntersection(hs, np.array([0.05, -0.02, 0.01]))
pts = hi.intersections[np.lexsort(hi.intersections.T[::-1])]
put("halfspace 3", hs, pts, np.sort(hi.dual_vertices))
# 2-D polygon for the figure.
a = rng.normal(size=(12, 2)); a /= np.linalg.norm(a, axis=1)[:, None]
hs2 = np.hstack([a, -rng.uniform(0.6, 1.2, size=(12, 1))])
hi2 = HalfspaceIntersection(hs2, np.zeros(2))
put("halfspace 2", hs2, hi2.intersections[np.lexsort(hi2.intersections.T[::-1])])
# Spherical Voronoi.
s = rng.normal(size=(60, 3)); s /= np.linalg.norm(s, axis=1)[:, None]
sv = SphericalVoronoi(s, 1.0, np.zeros(3))
sv.sort_vertices_of_regions()
vv = sv.vertices[np.lexsort(sv.vertices.T[::-1])]
put("spherical", s, vv, sv.calculate_areas())
# LinearNDInterpolator.
for d in [3, 4]:
    p = rng.uniform(size=(80, d))
    vals = np.sin(p.sum(axis=1)) + p[:, 0] ** 2
    q = rng.uniform(-0.1, 1.1, size=(100, d))
    r = LinearNDInterpolator(p, vals)(q)
    put(f"linearnd {d}", p, vals, q, np.nan_to_num(r, nan=1e300))
open("data/geo217_reference.txt", "w").write("\n".join(out) + "\n")
