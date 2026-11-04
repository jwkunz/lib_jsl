# Geometric types: points, vectors, shapes and shared tables

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Represent points, vectors, lines, planes, polygons, polyhedra and meshes in one to three dimensions so that shapes can share vertices, change coordinate systems and be transformed in place |
| **Devised by** | René Descartes (1637), coordinates; Hermann Grassmann (1844) and Josiah Willard Gibbs (1881), vector algebra; Bruce Baumgart (1972), shared-vertex boundary representations for computer models |
| **Accuracy** | Single precision (`f32`, about seven significant digits) for the primitive types; the algorithms of the other entries work in `f64` on plain arrays |
| **Cost** | $O(1)$ per point lookup in a hash table; $O(k)$ for a shape with $k$ vertices |
| **API** | [`concrete`](../../api/lib_jsl_geometry/concrete/index.html) (`CoordinateVector1D`–`3D`, `FreeVector2D`–`3D`, `UnitVector1D`–`3D`, `Line1D`–`3D`, `Plane1D`–`3D`, `Triangle2D`–`3D`, `PolygonFace2D`–`3D`, `Tetrahedron3D`, `Mesh2D`, `SurfaceMesh3D`, `VolumeMesh3D`), [`GeometryTableRegistry`](../../api/lib_jsl_geometry/registry/struct.GeometryTableRegistry.html), [`HashGeometryTable`](../../api/lib_jsl_geometry/tables/struct.HashGeometryTable.html), [`common`](../../api/lib_jsl_geometry/common/index.html) (ids and traits), [`coordinate_systems`](../../api/lib_jsl_geometry/coordinate_systems/index.html), [`transformation_traits`](../../api/lib_jsl_geometry/transformation_traits/index.html) |

</div>

## Overview

Before any geometric algorithm can run, shapes must be written down. The simplest way gives each triangle its own three corner points. But a mesh then stores every shared corner several times, and moving one copy tears the surface. Solid modellers and finite-element codes instead keep each point once and describe shapes by referring to points. The library's primitive types work that way: **points live in a keyed table**, and lines, triangles, polygons, tetrahedra and meshes hold the **ids** of their points together with a shared handle to the table. Transforming a shape rewrites its points in the table, so every other shape using those points follows.

A second distinction runs through the types, that between **points** (positions) and **vectors** (displacements). A point plus a vector is a point, and two points differ by a vector, but two points cannot be added. The types make that distinction, and also allow coordinates to be held in polar, spherical or cylindrical form.

This entry describes those types. The algorithms in the other entries of this chapter take plain arrays such as `[f64; 3]` instead, so they can be used on data from anywhere.

## History

René Descartes' *La Géométrie* (1637) turned geometry into algebra with coordinates. Vectors arrived two centuries later, in Hermann Grassmann's *Ausdehnungslehre* (1844) and in Josiah Willard Gibbs's and Oliver Heaviside's vector analysis of the 1880s, which separated positions from directed magnitudes. Computer representations of solids began in the 1960s and 1970s. Bruce Baumgart's winged-edge structure (1972) and the boundary representations of CAD systems store each vertex once and build edges and faces on top, the idea used here. The point/vector distinction became standard through affine geometry in graphics, as in Tony DeRose's "coordinate-free geometric programming" (1989).

## The idea

### Points and vectors

For each dimension there are three coordinate types.
- A **coordinate vector** (`CoordinateVector1D`, `2D`, `3D`) is a position.
- A **free vector** (`FreeVector2D`, `3D`) is a displacement, with no position.
- A **unit vector** (`UnitVector1D`, `2D`, `3D`) is a direction, normalised on construction.

Adding a free vector to a coordinate vector moves it; dot and cross products are defined where they make sense. Components are `f32` ([`GeometryMeasure`](../../api/lib_jsl_geometry/common/type.GeometryMeasure.html)). The zero-dimensional layer is the [`IsPoint`](../../api/lib_jsl_geometry/dim_0/trait.IsPoint.html) trait that every coordinate type implements.

### Lines, planes and polygons

A **line** (`Line1D`, `2D`, `3D`) is a pair of point ids, with a length and a direction from its head to its tail. A **plane** (`Plane2D`, `3D`) is a point and a unit normal; in 2D it is a line used as a mirror, and `Plane1D` is a mirror point. A **triangle** (`Triangle2D`, `3D`) holds three point ids, a **polygon face** (`PolygonFace2D`, `3D`) any number, and a **tetrahedron** (`Tetrahedron3D`) four. They report area, perimeter, centroid, normal or orientation, and the tetrahedron its signed volume. A polygon face splits into triangles as a fan from its first vertex, so it should be convex or at least star-shaped from that vertex.

### Meshes

A **mesh** is a list of face ids into a face table. `Mesh2D` and `SurfaceMesh3D` are made of polygon faces and report their faces, edges, normals and total area. `VolumeMesh3D` is a list of tetrahedra. Its volume is the sum of theirs, and its boundary is the set of tetrahedron faces used exactly once. `boundary_triangles` returns those faces, and `surface_mesh` turns them into a `SurfaceMesh3D` in the caller's tables.

### Keyed tables and the registry

A [`HashGeometryTable`](../../api/lib_jsl_geometry/tables/struct.HashGeometryTable.html) maps ids ([`PointId`](../../api/lib_jsl_geometry/common/struct.PointId.html), `LineId`, `FaceId`, `TriangleId`, `TetrahedronId`) to primitives. It is shared between shapes through a reference-counted handle, `SharedGeometryTable` (`Rc<RefCell<…>>`), so the tables are single-threaded. The [`GeometryTableRegistry`](../../api/lib_jsl_geometry/registry/struct.GeometryTableRegistry.html) owns one table of each kind and is the usual starting point. Insert points, then build shapes from their ids and the registry's handles.

### Coordinate systems

Each coordinate type records the system its components are in: Cartesian or polar in 2D; Cartesian, spherical $(r, \theta, \phi)$ with $\theta$ the inclination from $+z$, or cylindrical $(\rho, \phi, z)$ in 3D. The conversions, applied by `set_coordinate_system` and `converted_to`, are the usual ones,

$$
x = r \sin\theta \cos\phi, \quad y = r \sin\theta \sin\phi, \quad z = r \cos\theta ,
$$

with $\theta = \arccos(z/r)$ and $\phi = \operatorname{atan2}(y, x)$ back. Changing the system converts the stored components. Arithmetic and lengths always use the Cartesian values, so the result does not depend on the system a point happens to be stored in.

### Traits and transformations

Capabilities are traits shared across the types: `IsLine`, `IsPolygon`, `IsTriangle`, `IsMesh`, `HasCentroid`, `HasMeasure` and many more, collected in [`geometric_traits`](../../api/lib_jsl_geometry/geometric_traits/index.html). The [transformation traits](../../api/lib_jsl_geometry/transformation_traits/index.html) change a shape in place:
- `CanTranslate` moves it by the vector of a line;
- `CanRotate` turns it about a line's axis by Rodrigues' formula;
- `CanMirror` reflects it in a plane, by $p \mapsto p - 2\,((p - p_0) \cdot n)\,n$;
- `CanShear` applies $x \leftarrow x + k y$, with $k$ the length of a line.

Writing this chapter corrected the documentation of `CanShear`, which claimed to shear along the line's direction.

## Algorithm

1. **Build**: insert points into the registry's point table under chosen ids.
2. **Compose**: create lines, faces, triangles and tetrahedra from point ids and the table handle; insert them into their own tables if meshes will refer to them.
3. **Measure and transform**: methods resolve ids through the table on each call, so they always see the current positions.

## Implementation

```rust,ignore
{{#include ../../../crates/geometry/src/dim_3/transform_support.rs:rodrigues}}
```

- Every shape resolves its points on demand; a missing id gives `None` or zero measures rather than a panic.
- With the `serde` feature, ids and primitive types can be serialised.

## In practice

```rust,ignore
{{#include ../../../crates/geometry/examples/geometry_types.rs}}
```

```text
triangle area 1, perimeter 5.2361, normal UnitVector3D(Cartesian, 0, 0, 1)
edge length 2
after translating by (0, 0, 3): edge from CoordinateVector3D(Cartesian, 0, 0, 3) to CoordinateVector3D(Cartesian, 2, 0, 3)
after mirroring in z = 1: edge from CoordinateVector3D(Cartesian, 0, 0, -1) to CoordinateVector3D(Cartesian, 2, 0, -1)
(1, 1, 1) in spherical coordinates: [1.7320508, 0.9553166, 0.7853982]
plus (0, 0, 1): Cartesian [1.0000001, 1.0000001, 2.0]
```

The edge was never transformed itself, but it shares points 1 and 2 with the triangle, so it moved with it. The spherical round trip shows single precision: $1.0000001$.

**Strengths**

- Shared vertices: shapes stay connected when transformed, and meshes store each point once.
- Explicit points, vectors and unit vectors, and coordinates in polar, spherical or cylindrical form.

**Weaknesses**

- `f32` components: about seven digits, too few for large coordinates or accumulated transforms.
- `Rc<RefCell>` tables are not thread-safe, and the trait-heavy interface is verbose for simple tasks, for which the array-based algorithms are easier.
- Polygon triangulation is a fan, correct only for polygons star-shaped from the first vertex.

## Exercises

1. Explain why adding two points is meaningless but an affine combination $\sum w_i p_i$ with $\sum w_i = 1$ is not.
2. Derive the spherical-to-Cartesian conversion and its inverse, and find where the inverse is singular.
3. Show that reflection in a plane, $p \mapsto p - 2((p - p_0) \cdot n)\, n$, is an isometry with determinant $-1$.
4. Give a non-convex polygon whose fan triangulation from its first vertex covers area outside the polygon.
5. Show that the boundary of a tetrahedral mesh is the set of faces belonging to exactly one tetrahedron.

## Further reading

- T. DeRose, "A coordinate-free approach to geometric programming", in *Math for SIGGRAPH*, SIGGRAPH Course Notes, 1989.
- B. G. Baumgart, "Winged edge polyhedron representation", Stanford Artificial Intelligence Laboratory report AIM-179, 1972.
- P. J. Schneider and D. H. Eberly, *Geometric Tools for Computer Graphics*, Morgan Kaufmann, 2003, chapters 2–3.
- M. Mäntylä, *An Introduction to Solid Modeling*, Computer Science Press, 1988.
