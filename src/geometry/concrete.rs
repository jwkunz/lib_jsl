//! Public entry point for the concrete geometry implementation layer.
//!
//! This module re-exports the most commonly used concrete types so downstream code can start with
//! a compact import surface while still benefiting from the more focused on-disk organization.
//!
//! The underlying implementations live in dimension-specific modules such as
//! [`crate::geometry::two_d::coordinate_vector2d`], [`crate::geometry::three_d::coordinate_vector3d`],
//! [`crate::geometry::two_d::triangle2d`], and [`crate::geometry::three_d::surface_mesh3d`].
//!
//! Typical usage starts with a [`GeometryTableRegistry`], which provides the shared keyed tables
//! used by the graph of concrete primitives.
//!
//! ```rust
//! use lib_jsl::geometry::common::{IsGeometryTable, IsGeometryTableBase, PointId};
//! use lib_jsl::geometry::concrete::{GeometryTableRegistry, FreeVector3D, Line3D, CoordinateVector3D};
//! use lib_jsl::geometry::one_d::IsLine;
//!
//! let mut registry = GeometryTableRegistry::new();
//! registry
//!     .point_table_mut()
//!     .insert(PointId(1), CoordinateVector3D::new(0.0, 0.0, 0.0))
//!     .unwrap();
//! registry
//!     .point_table_mut()
//!     .insert(PointId(2), CoordinateVector3D::new(1.0, 0.0, 0.0))
//!     .unwrap();
//!
//! let line = Line3D::new(PointId(1), PointId(2), registry.point_table().clone());
//! assert!((line.length() - 1.0).abs() < 1e-5);
//! let shifted = CoordinateVector3D::new(0.0, 0.0, 0.0) + FreeVector3D::new(1.0, 2.0, 3.0);
//! assert_eq!(shifted, CoordinateVector3D::new(1.0, 2.0, 3.0));
//! ```

/// Re-exported one-dimensional concrete geometry types.
pub use crate::geometry::one_d::{Line1D, Plane1D, CoordinateVector1D, UnitVector1D};
/// Re-exported two-dimensional concrete geometry types.
pub use crate::geometry::two_d::{
    FreeVector2D, Line2D, Mesh2D, Plane2D, CoordinateVector2D, PolygonFace2D, Triangle2D, UnitVector2D,
};
/// Re-exported concrete geometry registry that owns the core shared tables.
pub use crate::geometry::registry::GeometryTableRegistry;
/// Re-exported keyed table implementations used by the concrete geometry graph.
pub use crate::geometry::tables::{HashGeometryTable, SharedGeometryTable};
/// Re-exported three-dimensional concrete geometry types.
pub use crate::geometry::three_d::{
    FreeVector3D, Line3D, Plane3D, CoordinateVector3D, PolygonFace3D, SurfaceMesh3D, Tetrahedron3D,
    Triangle3D, UnitVector3D, VolumeMesh3D,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::common::{
        FaceId, HasEdges, IsGeometryTable, IsGeometryTableBase, IsValid, PointId, TetrahedronId,
        TriangleId,
    };
    use crate::geometry::coordinate_systems::{CoordinateSystem2D, CoordinateSystem3D};
    use crate::geometry::one_d::IsLine;
    use crate::geometry::three_d::{IsMesh, IsTetrahedron};
    use crate::geometry::two_d::IsPolygon;

    #[test]
    fn concrete_geometry_smoke_test() {
        let mut registry = GeometryTableRegistry::new();

        registry
            .point_table_mut()
            .insert(PointId(1), CoordinateVector3D::new(0.0, 0.0, 0.0))
            .unwrap();
        registry
            .point_table_mut()
            .insert(PointId(2), CoordinateVector3D::new(1.0, 0.0, 0.0))
            .unwrap();
        registry
            .point_table_mut()
            .insert(PointId(3), CoordinateVector3D::new(0.0, 1.0, 0.0))
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
            .insert(PointId(1), CoordinateVector2D::new(0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(2), CoordinateVector2D::new(1.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(3), CoordinateVector2D::new(0.0, 1.0))
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
            .insert(PointId(1), CoordinateVector3D::new(0.0, 0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(2), CoordinateVector3D::new(1.0, 0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(3), CoordinateVector3D::new(0.0, 1.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(4), CoordinateVector3D::new(0.0, 0.0, 1.0))
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
    fn concrete_volume_mesh3d_smoke_test() {
        let mut registry = GeometryTableRegistry::new();
        registry
            .point_table_mut()
            .insert(PointId(1), CoordinateVector3D::new(0.0, 0.0, 0.0))
            .unwrap();
        registry
            .point_table_mut()
            .insert(PointId(2), CoordinateVector3D::new(1.0, 0.0, 0.0))
            .unwrap();
        registry
            .point_table_mut()
            .insert(PointId(3), CoordinateVector3D::new(0.0, 1.0, 0.0))
            .unwrap();
        registry
            .point_table_mut()
            .insert(PointId(4), CoordinateVector3D::new(0.0, 0.0, 1.0))
            .unwrap();

        let tetrahedron = Tetrahedron3D::new(
            PointId(1),
            PointId(2),
            PointId(3),
            PointId(4),
            registry.point_table().clone(),
        );
        let volume_mesh = VolumeMesh3D::from_tetrahedra(vec![tetrahedron]).unwrap();
        let surface_mesh = volume_mesh.surface_mesh(registry.face_table().clone()).unwrap();

        assert_eq!(volume_mesh.tetrahedron_count(), 1);
        assert_eq!(volume_mesh.boundary_triangles().len(), 4);
        assert_eq!(volume_mesh.boundary_faces().len(), 4);
        assert_eq!(surface_mesh.face_count(), 4);
        assert_eq!(registry.face_table().size(), 4);
        assert!((volume_mesh.volume() - (1.0 / 6.0)).abs() < 1e-5);
        assert!(volume_mesh.is_valid());
    }

    #[test]
    fn tetrahedron_trait_contract_smoke_test() {
        fn assert_is_tetrahedron<'a, T>(_value: &T)
        where
            T: IsTetrahedron<'a, CoordinateVector3D, UnitVector3D>,
            <T as HasEdges>::Edge: crate::geometry::one_d::IsLine<'a, CoordinateVector3D>,
        {
        }

        let point_table = std::rc::Rc::new(std::cell::RefCell::new(HashGeometryTable::new()));
        point_table
            .borrow_mut()
            .insert(PointId(1), CoordinateVector3D::new(0.0, 0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(2), CoordinateVector3D::new(1.0, 0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(3), CoordinateVector3D::new(0.0, 1.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(4), CoordinateVector3D::new(0.0, 0.0, 1.0))
            .unwrap();

        let tetrahedron =
            Tetrahedron3D::new(PointId(1), PointId(2), PointId(3), PointId(4), point_table);
        assert_is_tetrahedron(&tetrahedron);
    }

    #[test]
    fn registry_exposes_tetrahedron_table() {
        let mut registry = GeometryTableRegistry::new();
        registry
            .point_table_mut()
            .insert(PointId(1), CoordinateVector3D::new(0.0, 0.0, 0.0))
            .unwrap();
        registry
            .point_table_mut()
            .insert(PointId(2), CoordinateVector3D::new(1.0, 0.0, 0.0))
            .unwrap();
        registry
            .point_table_mut()
            .insert(PointId(3), CoordinateVector3D::new(0.0, 1.0, 0.0))
            .unwrap();
        registry
            .point_table_mut()
            .insert(PointId(4), CoordinateVector3D::new(0.0, 0.0, 1.0))
            .unwrap();

        let tetrahedron = Tetrahedron3D::new(
            PointId(1),
            PointId(2),
            PointId(3),
            PointId(4),
            registry.point_table().clone(),
        );
        registry
            .tetrahedron_table_mut()
            .insert(TetrahedronId(1), tetrahedron.clone())
            .unwrap();

        assert_eq!(registry.tetrahedron_table().size(), 1);
        assert_eq!(registry.tetrahedron_table().get(&TetrahedronId(1)), Some(tetrahedron));
    }

    #[test]
    fn coordinate_vector2d_coordinate_system_setter_converts_and_preserves_basis_on_math() {
        let mut coordinate_vector = CoordinateVector2D::new(1.0, 1.0);
        coordinate_vector.set_coordinate_system(CoordinateSystem2D::Polar);

        let translated = coordinate_vector + FreeVector2D::new(1.0, 0.0);

        assert_eq!(coordinate_vector.coordinate_system(), CoordinateSystem2D::Polar);
        assert_eq!(translated.coordinate_system(), CoordinateSystem2D::Polar);
        assert!((coordinate_vector.x() - 1.0).abs() < 1e-5);
        assert!((coordinate_vector.y() - 1.0).abs() < 1e-5);
        assert!((translated.x() - 2.0).abs() < 1e-5);
        assert!((translated.y() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn coordinate_vector3d_coordinate_system_setter_converts_and_preserves_basis_on_math() {
        let mut coordinate_vector = CoordinateVector3D::new(1.0, 0.0, 1.0);
        coordinate_vector.set_coordinate_system(CoordinateSystem3D::Spherical);

        let translated = coordinate_vector + FreeVector3D::new(0.0, 1.0, 0.0);

        assert_eq!(coordinate_vector.coordinate_system(), CoordinateSystem3D::Spherical);
        assert_eq!(translated.coordinate_system(), CoordinateSystem3D::Spherical);
        assert!((coordinate_vector.x() - 1.0).abs() < 1e-5);
        assert!((coordinate_vector.y() - 0.0).abs() < 1e-5);
        assert!((coordinate_vector.z() - 1.0).abs() < 1e-5);
        assert!((translated.x() - 1.0).abs() < 1e-5);
        assert!((translated.y() - 1.0).abs() < 1e-5);
        assert!((translated.z() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn line_length_uses_cartesian_coordinates_after_basis_change() {
        let point_table = std::rc::Rc::new(std::cell::RefCell::new(HashGeometryTable::new()));
        let mut a = CoordinateVector2D::new(0.0, 0.0);
        let mut b = CoordinateVector2D::new(3.0, 4.0);
        a.set_coordinate_system(CoordinateSystem2D::Polar);
        b.set_coordinate_system(CoordinateSystem2D::Polar);

        point_table.borrow_mut().insert(PointId(1), a).unwrap();
        point_table.borrow_mut().insert(PointId(2), b).unwrap();

        let line = Line2D::new(PointId(1), PointId(2), point_table);
        assert!((line.length() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn polygon_face2d_triangle_round_trip_smoke_test() {
        let point_table = std::rc::Rc::new(std::cell::RefCell::new(HashGeometryTable::new()));
        point_table
            .borrow_mut()
            .insert(PointId(1), CoordinateVector2D::new(0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(2), CoordinateVector2D::new(2.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(3), CoordinateVector2D::new(2.0, 1.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(4), CoordinateVector2D::new(0.0, 1.0))
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
            .insert(PointId(1), CoordinateVector3D::new(0.0, 0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(2), CoordinateVector3D::new(2.0, 0.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(3), CoordinateVector3D::new(2.0, 1.0, 0.0))
            .unwrap();
        point_table
            .borrow_mut()
            .insert(PointId(4), CoordinateVector3D::new(0.0, 1.0, 0.0))
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
