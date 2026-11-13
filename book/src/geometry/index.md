# Computational geometry: triangulations, hulls, spatial search and meshes

Computational geometry turns questions about shapes into algorithms: which points are nearest, which triangles to draw, where a ray lands, what a part looks like in cross-section. It underlies computer graphics, CAD and 3D printing, robotics, geographic information systems, and scientific computing. This chapter covers `lib_jsl_geometry`:
- the primitive types: points, vectors, shapes and the tables that let them share vertices;
- rotations, Euler angles and quaternions;
- Delaunay triangulations and Voronoi diagrams;
- convex hulls in two and three dimensions, and boolean operations on polygons;
- k-d trees, quadtrees and octrees for nearest-neighbour and range search;
- intersection tests and closest points;
- smoothing, decimating and slicing triangle meshes;
- four-vectors, Lorentz boosts and the Christoffel symbols of curved space-time.

## The central questions

- **Exact or approximate predicates?** Geometric algorithms branch on signs: is this point left of that line, inside that circle? Rounding can make the signs inconsistent, and the algorithm then loops, crashes or returns overlapping triangles. The library compares each predicate with a bound on its rounding error and treats values below it as zero (collinear or cocircular). Most of the defects fixed in this chapter were of this kind or close to it.
- **Plain arrays or connected shapes?** The algorithms take plain `[f64; 2]` and `[f64; 3]` arrays and index triples. The [primitive types](types.md) keep shared vertices in tables, which suits editing connected models.
- **Static or dynamic?** A k-d tree or a triangulation is built once; a quadtree accepts insertions and removals.

## Testing against SciPy and Shapely

The references come from SciPy 1.17.1's `spatial` module (Qhull for triangulations, hulls and Voronoi diagrams; `cKDTree`; `Rotation` and `Slerp`) and from Shapely 2.2.0, on random inputs written out by `book/generator/data/make_geometry_refs.py` and read back by the generator:

{{#include ../generated/geometry/reference.md}}

## In the library

All of these are in `lib_jsl_geometry` ([API reference](../../api/lib_jsl_geometry/index.html)).

| You need | Use | Entry |
|---|---|---|
| Points, vectors and shapes that share vertices; coordinate systems | `GeometryTableRegistry`, `CoordinateVector3D`, `Triangle3D`, `SurfaceMesh3D`, … | [Geometric types](types.md) |
| To represent, convert or interpolate orientations | `Rotation3`, `EulerSequence`, `Quaternion` | [Rotations](rotations.md) |
| A triangle mesh of scattered points; nearest-site regions | `delaunay_triangulation`, `voronoi_diagram` | [Delaunay and Voronoi](delaunay-voronoi.md) |
| The convex outline of points; unions and intersections of polygons | `convex_hull_2d`, `convex_hull_3d`, `polygon_boolean` | [Hulls and booleans](hulls-booleans.md) |
| Nearest neighbours, points in a radius or a box | `KdTree`, `QuadTree`, `OctTree` | [Spatial trees](spatial-trees.md) |
| Ray casting, collision tests, nearest points of shapes | `ray_triangle`, `segment_segment_2d`, `closest_point_on_triangle`, … | [Intersections](intersections.md) |
| Cleaning, simplifying or slicing a mesh | `taubin_smoothing`, `decimate`, `slice_mesh` | [Meshes](meshes.md) |
| Relativistic kinematics; free fall in curved space-time | `SpaceTimeVector4D`, `ChristoffelSymbols4D` | [Space-time](spacetime.md) |
