//! Concrete three-dimensional line type for the public geometry API.
//!
//! [`Line3D`] stores stable point identifiers plus a shared point table handle, allowing it to
//! resolve endpoints lazily from the shared geometry graph instead of embedding point values.

use crate::common::{
    GeometricPrimitive, GeometricPrimitive3D, GeometryMeasure, HasVertices, PointId,
};
use crate::one_d::IsLine;
use crate::tables::SharedGeometryTable;
use crate::three_d::{CoordinateVector3D, UnitVector3D};
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

/// Concrete 3D line implementation backed by point ids and a shared point table.
#[derive(Debug, Clone, Serialize)]
pub struct Line3D {
    head_id: PointId,
    tail_id: PointId,
    #[serde(skip_serializing)]
    vertex_table: SharedGeometryTable<PointId, CoordinateVector3D>,
}

impl PartialEq for Line3D {
    fn eq(&self, other: &Self) -> bool {
        self.head_id == other.head_id && self.tail_id == other.tail_id
    }
}

impl Eq for Line3D {}

impl Hash for Line3D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.head_id.hash(state);
        self.tail_id.hash(state);
    }
}

impl Line3D {
    /// Creates a line from two point ids and a shared point table.
    ///
    /// The ids are interpreted against `vertex_table` whenever the line resolves its endpoints,
    /// length, midpoint, or direction.
    pub fn new(
        head_id: PointId,
        tail_id: PointId,
        vertex_table: SharedGeometryTable<PointId, CoordinateVector3D>,
    ) -> Self {
        Self {
            head_id,
            tail_id,
            vertex_table,
        }
    }
}

impl Display for Line3D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Line3D({:?} -> {:?})", self.head_id, self.tail_id)
    }
}

impl GeometricPrimitive for Line3D {}
impl GeometricPrimitive3D for Line3D {}

impl<'a> HasVertices<'a> for Line3D {
    type Vertex = CoordinateVector3D;
    type VertexTable = SharedGeometryTable<PointId, CoordinateVector3D>;

    fn vertex_table(&self) -> &Self::VertexTable {
        &self.vertex_table
    }

    fn vertex_table_mut(&mut self) -> &mut Self::VertexTable {
        &mut self.vertex_table
    }

    fn set_vertex_table(&mut self, table: &'a mut Self::VertexTable) {
        self.vertex_table = table.clone();
    }
}

impl<'a> IsLine<'a, CoordinateVector3D> for Line3D {
    fn head_id(&self) -> PointId {
        self.head_id
    }

    fn set_head_id(&mut self, point_id: PointId) -> Result<(), String> {
        self.head_id = point_id;
        Ok(())
    }

    fn tail_id(&self) -> PointId {
        self.tail_id
    }

    fn set_tail_id(&mut self, point_id: PointId) -> Result<(), String> {
        self.tail_id = point_id;
        Ok(())
    }

    fn length(&self) -> GeometryMeasure {
        match (self.head(), self.tail()) {
            (Some(head), Some(tail)) => {
                ((tail.x() - head.x()).powi(2)
                    + (tail.y() - head.y()).powi(2)
                    + (tail.z() - head.z()).powi(2))
                    .sqrt()
            }
            _ => 0.0,
        }
    }

    fn midpoint(&self) -> Option<CoordinateVector3D> {
        match (self.head(), self.tail()) {
            (Some(head), Some(tail)) => Some(CoordinateVector3D::new(
                (head.x() + tail.x()) / 2.0,
                (head.y() + tail.y()) / 2.0,
                (head.z() + tail.z()) / 2.0,
            )),
            _ => None,
        }
    }

    fn direction(&self) -> impl crate::common::IsUnitVector {
        match (self.head(), self.tail()) {
            (Some(head), Some(tail)) => {
                UnitVector3D::new(tail.x() - head.x(), tail.y() - head.y(), tail.z() - head.z())
            }
            _ => UnitVector3D::new(1.0, 0.0, 0.0),
        }
    }
}
