//! Concrete root registry for the public geometry table graph.
//!
//! [`GeometryTableRegistry`] is the main starting point for users of the concrete API. It owns the
//! current core keyed tables for points, lines, polygon faces, triangles, and tetrahedra, and
//! exposes them through the [`IsGeometryTableBase`](crate::geometry::common::IsGeometryTableBase)
//! contract.
//!
//! Each concrete primitive borrows one or more of these tables through shared handles. This lets
//! higher-level objects such as lines, polygon faces, triangles, and meshes resolve their child
//! geometry by stable IDs while still sharing a single point table.

use crate::geometry::common::{
    FaceId, IsGeometryTableBase, LineId, PointId, TetrahedronId, TriangleId,
};
use crate::geometry::tables::{HashGeometryTable, SharedGeometryTable};
use crate::geometry::three_d::{
    Line3D, CoordinateVector3D, PolygonFace3D, Tetrahedron3D, Triangle3D, UnitVector3D,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Root registry holding the current concrete point, line, face, triangle, and tetrahedron tables.
///
/// This type is intentionally small and explicit so it can serve as the top-level integration
/// object for applications building a geometry graph on top of the trait system.
#[derive(Debug, Clone)]
pub struct GeometryTableRegistry {
    point_table: SharedGeometryTable<PointId, CoordinateVector3D>,
    line_table: SharedGeometryTable<LineId, Line3D>,
    face_table: SharedGeometryTable<FaceId, PolygonFace3D>,
    triangle_table: SharedGeometryTable<TriangleId, Triangle3D>,
    tetrahedron_table: SharedGeometryTable<TetrahedronId, Tetrahedron3D>,
}

impl GeometryTableRegistry {
    /// Creates an empty registry with all currently supported core tables initialized.
    ///
    /// The returned registry is ready to hand out shared table handles to concrete primitives.
    pub fn new() -> Self {
        Self {
            point_table: Rc::new(RefCell::new(HashGeometryTable::new())),
            line_table: Rc::new(RefCell::new(HashGeometryTable::new())),
            face_table: Rc::new(RefCell::new(HashGeometryTable::new())),
            triangle_table: Rc::new(RefCell::new(HashGeometryTable::new())),
            tetrahedron_table: Rc::new(RefCell::new(HashGeometryTable::new())),
        }
    }
}

impl Default for GeometryTableRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> IsGeometryTableBase<'a> for GeometryTableRegistry {
    type Point = CoordinateVector3D;
    type Normal = UnitVector3D;
    type Line = Line3D;
    type Face = PolygonFace3D;
    type Triangle = Triangle3D;
    type Tetrahedron = Tetrahedron3D;
    type PointTable = SharedGeometryTable<PointId, CoordinateVector3D>;
    type LineTable = SharedGeometryTable<LineId, Line3D>;
    type FaceTable = SharedGeometryTable<FaceId, PolygonFace3D>;
    type TriangleTable = SharedGeometryTable<TriangleId, Triangle3D>;
    type TetrahedronTable = SharedGeometryTable<TetrahedronId, Tetrahedron3D>;

    fn point_table(&self) -> &Self::PointTable {
        &self.point_table
    }

    fn point_table_mut(&mut self) -> &mut Self::PointTable {
        &mut self.point_table
    }

    fn line_table(&self) -> &Self::LineTable {
        &self.line_table
    }

    fn line_table_mut(&mut self) -> &mut Self::LineTable {
        &mut self.line_table
    }

    fn face_table(&self) -> &Self::FaceTable {
        &self.face_table
    }

    fn face_table_mut(&mut self) -> &mut Self::FaceTable {
        &mut self.face_table
    }

    fn triangle_table(&self) -> &Self::TriangleTable {
        &self.triangle_table
    }

    fn triangle_table_mut(&mut self) -> &mut Self::TriangleTable {
        &mut self.triangle_table
    }

    fn tetrahedron_table(&self) -> &Self::TetrahedronTable {
        &self.tetrahedron_table
    }

    fn tetrahedron_table_mut(&mut self) -> &mut Self::TetrahedronTable {
        &mut self.tetrahedron_table
    }
}
