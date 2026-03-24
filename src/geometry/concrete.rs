//! Public entry point for the concrete geometry implementation layer.
//!
//! This module re-exports the most commonly used concrete types so downstream code can start with
//! a compact import surface while still benefiting from the more focused on-disk organization.
//!
//! The underlying implementations live in dimension-specific modules such as
//! [`crate::geometry::two_d::point2d`], [`crate::geometry::three_d::point3d`],
//! [`crate::geometry::two_d::triangle2d`], and [`crate::geometry::three_d::surface_mesh3d`].
//!
//! Typical usage starts with a [`GeometryTableRegistry`], which provides the shared keyed tables
//! used by the graph of concrete primitives.
//!
//! ```rust
//! use lib_jsl::geometry::common::{IsGeometryTable, IsGeometryTableBase, PointId};
//! use lib_jsl::geometry::concrete::{GeometryTableRegistry, Line3D, Point3D};
//! use lib_jsl::geometry::one_d::IsLine;
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

/// Re-exported one-dimensional concrete geometry types.
pub use crate::geometry::one_d::{Line1D, Plane1D, Point1D, UnitVector1D};
/// Re-exported two-dimensional concrete geometry types.
pub use crate::geometry::two_d::{
    Line2D, Mesh2D, Plane2D, Point2D, PolygonFace2D, Triangle2D, UnitVector2D,
};
/// Re-exported concrete geometry registry that owns the core shared tables.
pub use crate::geometry::registry::GeometryTableRegistry;
/// Re-exported keyed table implementations used by the concrete geometry graph.
pub use crate::geometry::tables::{HashGeometryTable, SharedGeometryTable};
/// Re-exported three-dimensional concrete geometry types.
pub use crate::geometry::three_d::{
    Line3D, Plane3D, Point3D, PolygonFace3D, SurfaceMesh3D, Tetrahedron3D, Triangle3D,
    UnitVector3D,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::common::{
        FaceId, HasEdges, IsGeometryTable, IsGeometryTableBase, IsValid, PointId, TriangleId,
    };
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

        let mesh = SurfaceMesh3D::new(vec![FaceId(1)], registry.point_table().clone(), registry.face_table().clone());

        assert!((triangle.area() - 0.5).abs() < 1e-5);
        assert_eq!(mesh.face_count(), 1);
        assert!((mesh.surface_area() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn concrete_geometry_2d_mesh_smoke_test() {
        let point_table = std::rc::Rc::new(std::cell::RefCell::new(HashGeometryTable::new()));
        point_table
            .borrow_mut()
            .insert(PointId(1), Point2D::new(0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(2), Point2D::new(1.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(3), Point2D::new(0.0, 1.0))
            .unwrap();

        let face_table = std::rc::Rc::new(std::cell::RefCell::new(HashGeometryTable::new()));
        face_table
            .borrow_mut()
            .insert(
                FaceId(1),
                PolygonFace2D::new(vec![PointId(1), PointId(2), PointId(3)], point_table.clone()),
            )
            .unwrap();

        let mesh = Mesh2D::new(vec![FaceId(1)], point_table, face_table);
        assert_eq!(mesh.face_count(), 1);
        assert!((mesh.surface_area() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn concrete_tetrahedron3d_smoke_test() {
        let point_table = std::rc::Rc::new(std::cell::RefCell::new(HashGeometryTable::new()));
        point_table
            .borrow_mut()
            .insert(PointId(1), Point3D::new(0.0, 0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(2), Point3D::new(1.0, 0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(3), Point3D::new(0.0, 1.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(4), Point3D::new(0.0, 0.0, 1.0))
            .unwrap();

        let tetrahedron =
            Tetrahedron3D::new(PointId(1), PointId(2), PointId(3), PointId(4), point_table);

        assert_eq!(tetrahedron.face_count(), 4);
        assert_eq!(tetrahedron.edge_count(), 6);
        assert!((tetrahedron.volume() - (1.0 / 6.0)).abs() < 1e-5);
        assert!(tetrahedron.surface_area() > 2.0);
        assert!(tetrahedron.is_valid());
    }

    #[test]
    fn polygon_face2d_triangle_round_trip_smoke_test() {
        let point_table = std::rc::Rc::new(std::cell::RefCell::new(HashGeometryTable::new()));
        point_table
            .borrow_mut()
            .insert(PointId(1), Point2D::new(0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(2), Point2D::new(2.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(3), Point2D::new(2.0, 1.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(4), Point2D::new(0.0, 1.0))
            .unwrap();

        let polygon = PolygonFace2D::new(
            vec![PointId(1), PointId(2), PointId(3), PointId(4)],
            point_table.clone(),
        );
        let triangles = polygon.triangulate();
        let recomposed = PolygonFace2D::from_triangles(&triangles).unwrap();

        assert_eq!(triangles.len(), 2);
        assert!((triangles.iter().map(|triangle| triangle.area()).sum::<f32>() - polygon.area()).abs() < 1e-5);
        assert_eq!(recomposed.vertex_ids().collect::<Vec<_>>(), polygon.vertex_ids().collect::<Vec<_>>());
    }

    #[test]
    fn polygon_face3d_triangle_round_trip_smoke_test() {
        let point_table = std::rc::Rc::new(std::cell::RefCell::new(HashGeometryTable::new()));
        point_table
            .borrow_mut()
            .insert(PointId(1), Point3D::new(0.0, 0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(2), Point3D::new(2.0, 0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(3), Point3D::new(2.0, 1.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(4), Point3D::new(0.0, 1.0, 0.0))
            .unwrap();

        let polygon = PolygonFace3D::new(
            vec![PointId(1), PointId(2), PointId(3), PointId(4)],
            point_table.clone(),
        );
        let triangles = polygon.triangulate();
        let recomposed = PolygonFace3D::from_triangles(&triangles).unwrap();

        assert_eq!(triangles.len(), 2);
        assert!((triangles.iter().map(|triangle| triangle.area()).sum::<f32>() - polygon.area()).abs() < 1e-5);
        assert_eq!(recomposed.vertex_ids().collect::<Vec<_>>(), polygon.vertex_ids().collect::<Vec<_>>());
    }
}
