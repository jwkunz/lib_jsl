//! Concrete one-dimensional line type.

use crate::geometry::common::{GeometricPrimitive, GeometryMeasure, HasVertices, PointId};
use crate::geometry::one_d::{IsLine, Point1D, UnitVector1D};
use crate::geometry::tables::SharedGeometryTable;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

/// Concrete 1D line implementation backed by point ids and a shared point table.
#[derive(Debug, Clone, Serialize)]
pub struct Line1D {
    head_id: PointId,
    tail_id: PointId,
    #[serde(skip_serializing)]
    vertex_table: SharedGeometryTable<PointId, Point1D>,
}

impl PartialEq for Line1D {
    fn eq(&self, other: &Self) -> bool {
        self.head_id == other.head_id && self.tail_id == other.tail_id
    }
}

impl Eq for Line1D {}

impl Hash for Line1D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.head_id.hash(state);
        self.tail_id.hash(state);
    }
}

impl Line1D {
    /// Creates a line from two point ids and a shared point table.
    pub fn new(
        head_id: PointId,
        tail_id: PointId,
        vertex_table: SharedGeometryTable<PointId, Point1D>,
    ) -> Self {
        Self {
            head_id,
            tail_id,
            vertex_table,
        }
    }
}

impl Display for Line1D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Line1D({:?} -> {:?})", self.head_id, self.tail_id)
    }
}

impl GeometricPrimitive for Line1D {}

impl<'a> HasVertices<'a> for Line1D {
    type Vertex = Point1D;
    type VertexTable = SharedGeometryTable<PointId, Point1D>;

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

impl<'a> IsLine<'a, Point1D> for Line1D {
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
            (Some(head), Some(tail)) => (tail[0] - head[0]).abs(),
            _ => 0.0,
        }
    }

    fn midpoint(&self) -> Option<Point1D> {
        match (self.head(), self.tail()) {
            (Some(head), Some(tail)) => Some(Point1D::new((head[0] + tail[0]) / 2.0)),
            _ => None,
        }
    }

    fn direction(&self) -> impl crate::geometry::common::IsUnitVector {
        match (self.head(), self.tail()) {
            (Some(head), Some(tail)) => UnitVector1D::new(tail[0] - head[0]),
            _ => UnitVector1D::new(1.0),
        }
    }
}
