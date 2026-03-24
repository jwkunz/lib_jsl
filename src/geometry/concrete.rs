//! Compatibility re-exports for the concrete geometry implementation surface.
//!
//! The actual implementations now live in focused modules such as
//! [`crate::geometry::zero_d::point3d`], [`crate::geometry::one_d::line3d`],
//! [`crate::geometry::two_d::triangle3d`], and [`crate::geometry::three_d::mesh3d`].

pub use crate::geometry::one_d::Line3D;
pub use crate::geometry::registry::GeometryTableRegistry;
pub use crate::geometry::tables::{HashGeometryTable, SharedGeometryTable};
pub use crate::geometry::three_d::{Mesh3D, Plane3D, UnitVector3D};
pub use crate::geometry::two_d::{PolygonFace3D, Triangle3D};
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
