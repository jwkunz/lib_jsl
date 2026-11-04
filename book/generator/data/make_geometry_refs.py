"""Reference values for the chapter on computational geometry, from SciPy's `spatial` module
(Qhull for Delaunay triangulations, convex hulls and Voronoi diagrams; `cKDTree`; `Rotation` and
`Slerp`) and Shapely's polygon overlay (the book used SciPy 1.17.1 and Shapely 2.2.0).

    python book/generator/data/make_geometry_refs.py > book/generator/data/geometry_reference.txt

Inputs are written as `points <name>|x,y,...` lines (rounded to six decimals so that both sides
read the same numbers), followed by results as `<case>|values`.
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi, cKDTree
from scipy.spatial.transform import Rotation, Slerp
from shapely.geometry import Polygon

rng = np.random.default_rng(2026)


def out(name, values):
    print(name + "|" + ",".join(f"{float(v):.15g}" for v in np.ravel(values)))


def points(name, array):
    array = np.round(array, 6)
    out("points " + name, array)
    return array


# Delaunay triangulation and 2D hull of uniform points: the triangles as sorted index triples.
p = points("uniform", rng.uniform(0.0, 1.0, (500, 2)))
tri = Delaunay(p)
out("delaunay uniform", sorted(tuple(sorted(int(i) for i in s)) for s in tri.simplices))
h = ConvexHull(p)
out("hull2 uniform", [h.volume] + sorted(int(v) for v in h.vertices))

# 3D hulls: points in a ball and on a sphere.
for name, n in [("ball", 1000), ("sphere", 400)]:
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    if name == "ball":
        v *= rng.uniform(0.0, 1.0, (n, 1)) ** (1 / 3)
    q = points(name, v)
    h = ConvexHull(q)
    out("hull3 " + name, [h.volume, h.area] + sorted(int(i) for i in h.vertices))

# Voronoi cells clipped to the unit square, by mirroring the sites across its four sides.
s = points("sites", rng.uniform(0.0, 1.0, (300, 2)))
mirrored = np.vstack([s, s * [-1, 1], s * [-1, 1] + [2, 0], s * [1, -1], s * [1, -1] + [0, 2]])
vor = Voronoi(mirrored)
out("voronoi areas", [Polygon(vor.vertices[vor.regions[vor.point_region[i]]]).area for i in range(len(s))])

# k nearest neighbours in 3D and radius counts in 2D.
k = points("cloud", rng.uniform(0.0, 1.0, (4000, 3)))
queries = points("queries", rng.uniform(-0.1, 1.1, (200, 3)))
d, i = cKDTree(k).query(queries, 8)
out("knn index", i)
out("knn distance", d)
tree2 = cKDTree(s)
centres = points("centres", rng.uniform(0.0, 1.0, (100, 2)))
out("radius counts", [len(tree2.query_ball_point(c, 0.08)) for c in centres])

# Boolean operations on pairs of random star-shaped polygons.
def star(cx, m):
    t = 2 * np.pi * (np.arange(m) + 0.8 * rng.uniform(size=m)) / m
    r = 0.3 + 0.7 * rng.uniform(size=m)
    return np.round(np.column_stack([cx + r * np.cos(t), r * np.sin(t)]), 6)


areas = []
for trial in range(200):
    a = star(0.0, 5 + trial % 9)
    b = star(rng.uniform(0.0, 0.8), 4 + trial % 7)
    out(f"polygon a{trial}", a)
    out(f"polygon b{trial}", b)
    pa, pb = Polygon(a), Polygon(b)
    areas += [pa.intersection(pb).area, pa.union(pb).area, pa.difference(pb).area]
out("boolean areas", areas)

# Rotations: matrices from intrinsic Euler angles, the angles back from matrices, and slerp.
angles = points("euler angles", rng.uniform(-np.pi, np.pi, (100, 3)))
for seq in ["XYZ", "ZYX", "ZXZ", "YZY"]:
    r = Rotation.from_euler(seq, angles)
    out("euler matrix " + seq, r.as_matrix())
    out("euler angles " + seq, r.as_euler(seq))
ends = Rotation.from_quat(points("slerp ends", rng.normal(size=(2, 4))))
times = np.linspace(0.0, 1.0, 11)
out("slerp", Slerp([0.0, 1.0], ends)(times).as_quat(scalar_first=True))
