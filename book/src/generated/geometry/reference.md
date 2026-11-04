| computation (reference) | result |
|---|---:|
| Delaunay triangulation, 500 uniform points (`Delaunay`) | 982 triangles, identical to Qhull's |
| 2D convex hull, same points (`ConvexHull`) | 16 vertices, same set: true; area difference 0.0e0 |
| 3D convex hull, 1000 points in a ball (`ConvexHull`) | 140 vertices, same set: true; volume difference 2.7e-15, area difference 2.3e-14 |
| 3D convex hull, 400 points on a sphere (`ConvexHull`) | 400 vertices, same set: true; volume difference 3.6e-15, area difference 4.8e-14 |
| Voronoi cells of 300 sites clipped to the unit square (`Voronoi` of the mirrored sites) | largest cell-area difference 1.6e-16 |
| 8 nearest neighbours of 200 queries among 4000 points in 3D (`cKDTree.query`) | identical for 200 of 200; largest distance difference 5.0e-16 |
| points within 0.08 of 100 centres, quadtree (`cKDTree.query_ball_point`) | same count for 100 of 100 |
| intersection, union and difference of 200 pairs of star polygons (Shapely) | largest area difference 5.3e-15 |
| intrinsic XYZ Euler angles, 100 rotations (`Rotation.from_euler`, `as_euler`) | matrix difference 8.9e-16, angle difference 3.6e-15 |
| intrinsic ZYX Euler angles, 100 rotations (`Rotation.from_euler`, `as_euler`) | matrix difference 1.0e-15, angle difference 1.0e-14 |
| intrinsic ZXZ Euler angles, 100 rotations (`Rotation.from_euler`, `as_euler`) | matrix difference 8.9e-16, angle difference 3.6e-15 |
| intrinsic YZY Euler angles, 100 rotations (`Rotation.from_euler`, `as_euler`) | matrix difference 8.9e-16, angle difference 3.6e-15 |
| spherical linear interpolation at 11 times (`Slerp`) | largest quaternion component difference 5.6e-16 |
