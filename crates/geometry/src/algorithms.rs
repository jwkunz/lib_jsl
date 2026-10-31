//! Geometry algorithms.

pub mod delaunay_triangulation;
pub mod convex_hull;
pub mod slicing;
pub mod rotations;
pub mod quaternions;
pub mod nearest_point_finding;
pub mod intersection;
pub mod kd_trees;
pub mod oct_trees;
pub mod quad_trees;

pub use delaunay_triangulation::delaunay_triangulation;
pub use convex_hull::convex_hull;
pub use slicing::slicing;
pub use rotations::{Axis, EulerSequence, Matrix3, Rotation2, Rotation3};
pub use quaternions::Quaternion;
pub use nearest_point_finding::nearest_point_finding;
pub use intersection::intersection;
pub use kd_trees::KdTree;
pub use oct_trees::OctTree;
pub use quad_trees::QuadTree;
