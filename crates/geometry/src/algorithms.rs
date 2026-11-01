//! Geometry algorithms.

pub mod delaunay_triangulation;
pub mod convex_hull;
pub mod slicing;
pub mod rotations;
pub mod quaternions;
pub mod nearest_point_finding;
pub mod intersection;
pub mod kd_trees;
pub mod region_tree;
pub mod oct_trees;
pub mod quad_trees;
pub mod voronoi;

pub use delaunay_triangulation::{Triangulation, delaunay_triangulation};
pub use convex_hull::{Hull3D, convex_hull_2d, convex_hull_3d, convex_polygon_contains_2d, polygon_area_2d};
pub use slicing::{Contour, slice_mesh, slice_mesh_levels};
pub use rotations::{Axis, EulerSequence, Matrix3, Rotation2, Rotation3};
pub use quaternions::Quaternion;
pub use nearest_point_finding::{Closest, closest_point_on_aabb, closest_point_on_line, closest_point_on_mesh, closest_point_on_plane, closest_point_on_polygon_boundary, closest_point_on_polyline, closest_point_on_ray, closest_point_on_segment, closest_point_on_sphere, closest_point_on_triangle, closest_points_between_segments, distance_point_to_line_3d};
pub use intersection::{Intersection2D, TriangleHit, aabb_aabb, circle_circle, line_line_2d, line_plane_parameter, plane_plane, ray_aabb, ray_plane, ray_sphere, ray_triangle, segment_plane, segment_segment_2d, segment_triangle, sphere_sphere, triangle_plane};
pub use kd_trees::KdTree;
pub use oct_trees::OctTree;
pub use region_tree::RegionTree;
pub use quad_trees::QuadTree;
pub use voronoi::{VoronoiDiagram, polygon_signed_area, voronoi_diagram};
