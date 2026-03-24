//! Public entry point for the concrete geometry implementation layer.
//!
//! This module re-exports the most commonly used concrete types so downstream code can start with
//! a compact import surface while still benefiting from the more focused on-disk organization.
//!
//! The underlying implementations live in dimension-specific modules such as
//! [`crate::geometry::zero_d::point3d`], [`crate::geometry::one_d::line3d`],
//! [`crate::geometry::two_d::triangle3d`], and [`crate::geometry::three_d::mesh3d`].
//!
//! Typical usage starts with a [`GeometryTableRegistry`], which provides the shared keyed tables
//! used by the graph of concrete primitives.
//!
//! ```rust
//! use lib_jsl::geometry::common::{IsGeometryTable, IsGeometryTableBase, PointId};
//! use lib_jsl::geometry::concrete::{GeometryTableRegistry, Line3D, Point3D};
//!
//! let mut registry = GeometryTableRegistry::new();
//! registry
//!     .point_table_mut()
//!     .insert(PointId(1), Point3D::new(0.0, 0.0, 0.0))
//!     .unwrap();
//! registry
//!     .point_table_mut()
//!     .insert(PointId(2), Point3D::new(1.0, 0.0, 0.0))
//!     .unwrap();
//!
//! let line = Line3D::new(PointId(1), PointId(2), registry.point_table().clone());
//! assert!((line.length() - 1.0).abs() < 1e-5);
//! ```

/// Re-exported concrete line implementation for ergonomic public imports.
pub use crate::geometry::one_d::Line3D;
/// Re-exported concrete geometry registry that owns the core shared tables.
pub use crate::geometry::registry::GeometryTableRegistry;
/// Re-exported keyed table implementations used by the concrete geometry graph.
pub use crate::geometry::tables::{HashGeometryTable, SharedGeometryTable};
/// Re-exported three-dimensional concrete geometry types.
pub use crate::geometry::three_d::{Mesh3D, Plane3D, UnitVector3D};
/// Re-exported polygonal concrete geometry types.
pub use crate::geometry::two_d::{PolygonFace3D, Triangle3D};
/// Re-exported concrete point implementation.
pub use crate::geometry::zero_d::Point3D;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::common::{FaceId, IsGeometryTable, IsGeometryTableBase, PointId, TriangleId};
    use crate::geometry::three_d::IsMesh;
    use crate::geometry::two_d::IsPolygon;

    #[test]
    fn concrete_geometry_smoke_test() {
        let mut registry = GeometryTableRegistry::new();

        registry
            .point_table_mut()
            .insert(PointId(1), Point3D::new(0.0, 0.0, 0.0))
            .unwrap();
        registry
            .point_table_mut()
            .insert(PointId(2), Point3D::new(1.0, 0.0, 0.0))
            .unwrap();
        registry
            .point_table_mut()
            .insert(PointId(3), Point3D::new(0.0, 1.0, 0.0))
            .unwrap();

        let triangle = Triangle3D::new(PointId(1), PointId(2), PointId(3), registry.point_table().clone());
        registry
            .triangle_table_mut()
            .insert(TriangleId(1), triangle.clone())
            .unwrap();

        let face = PolygonFace3D::new(vec![PointId(1), PointId(2), PointId(3)], registry.point_table().clone());
        registry.face_table_mut().insert(FaceId(1), face).unwrap();

        let mesh = Mesh3D::new(vec![FaceId(1)], registry.point_table().clone(), registry.face_table().clone());

        assert!((triangle.area() - 0.5).abs() < 1e-5);
        assert_eq!(mesh.face_count(), 1);
        assert!((mesh.surface_area() - 0.5).abs() < 1e-5);
    }
}
