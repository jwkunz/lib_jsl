//! Concrete root registry for the current core geometry tables.

use crate::geometry::common::{
    FaceId, IsGeometryTableBase, LineId, PointId, TriangleId,
};
use crate::geometry::tables::{HashGeometryTable, SharedGeometryTable};
use crate::geometry::three_d::UnitVector3D;
use crate::geometry::two_d::{PolygonFace3D, Triangle3D};
use crate::geometry::{one_d::Line3D, zero_d::Point3D};
use std::cell::RefCell;
use std::rc::Rc;

/// Root registry holding the current concrete point, line, face, and triangle tables.
#[derive(Debug, Clone)]
pub struct GeometryTableRegistry {
    point_table: SharedGeometryTable<PointId, Point3D>,
    line_table: SharedGeometryTable<LineId, Line3D>,
    face_table: SharedGeometryTable<FaceId, PolygonFace3D>,
    triangle_table: SharedGeometryTable<TriangleId, Triangle3D>,
}

impl GeometryTableRegistry {
    /// Creates an empty registry with all core tables initialized.
    pub fn new() -> Self {
        Self {
            point_table: Rc::new(RefCell::new(HashGeometryTable::new())),
            line_table: Rc::new(RefCell::new(HashGeometryTable::new())),
            face_table: Rc::new(RefCell::new(HashGeometryTable::new())),
            triangle_table: Rc::new(RefCell::new(HashGeometryTable::new())),
        }
    }
}

impl Default for GeometryTableRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> IsGeometryTableBase<'a> for GeometryTableRegistry {
    type Point = Point3D;
    type Normal = UnitVector3D;
    type Line = Line3D;
    type Face = PolygonFace3D;
    type Triangle = Triangle3D;
    type PointTable = SharedGeometryTable<PointId, Point3D>;
    type LineTable = SharedGeometryTable<LineId, Line3D>;
    type FaceTable = SharedGeometryTable<FaceId, PolygonFace3D>;
    type TriangleTable = SharedGeometryTable<TriangleId, Triangle3D>;

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
}
