//! Concrete triangle type built from two-dimensional points.

use crate::geometry::common::{
    GeometricPrimitive, GeometricPrimitive2D, GeometryMeasure, HasCentroid, HasEdges, HasMeasure,
    HasVertices, IsPlane, PointId,
};
use crate::geometry::tables::SharedGeometryTable;
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanShear, CanTranslate};
use crate::geometry::two_d::{
    HasOrientation, IsPolygon, IsTriangle, Line2D, Plane2D, Point2D, PolygonFace2D, UnitVector2D,
    Orientation2D,
};
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

/// Concrete 2D triangle implementation backed by three point ids.
#[derive(Debug, Clone, Serialize)]
pub struct Triangle2D {
    a_id: PointId,
    b_id: PointId,
    c_id: PointId,
    #[serde(skip_serializing)]
    vertex_table: SharedGeometryTable<PointId, Point2D>,
}

impl PartialEq for Triangle2D {
    fn eq(&self, other: &Self) -> bool {
        self.a_id == other.a_id && self.b_id == other.b_id && self.c_id == other.c_id
    }
}

impl Eq for Triangle2D {}

impl Hash for Triangle2D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.a_id.hash(state);
        self.b_id.hash(state);
        self.c_id.hash(state);
    }
}

impl Triangle2D {
    /// Creates a triangle from three point ids and a shared point table.
    pub fn new(
        a_id: PointId,
        b_id: PointId,
        c_id: PointId,
        vertex_table: SharedGeometryTable<PointId, Point2D>,
    ) -> Self {
        Self {
            a_id,
            b_id,
            c_id,
            vertex_table,
        }
    }
}

impl Display for Triangle2D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Triangle2D({:?}, {:?}, {:?})", self.a_id, self.b_id, self.c_id)
    }
}

impl GeometricPrimitive for Triangle2D {}
impl GeometricPrimitive2D for Triangle2D {}

impl<'a> HasVertices<'a> for Triangle2D {
    type Vertex = Point2D;
    type VertexTable = SharedGeometryTable<PointId, Point2D>;

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

impl HasEdges for Triangle2D {
    type Edge = Line2D;

    fn edge_count(&self) -> usize {
        3
    }

    fn edge(&self, index: usize) -> Option<Self::Edge> {
        let (head, tail) = match index {
            0 => (self.a_id, self.b_id),
            1 => (self.b_id, self.c_id),
            2 => (self.c_id, self.a_id),
            _ => return None,
        };
        Some(Line2D::new(head, tail, self.vertex_table.clone()))
    }
}

impl HasCentroid for Triangle2D {
    type Point = Point2D;

    fn centroid(&self) -> Self::Point {
        let a = self.a().unwrap_or(Point2D::new(0.0, 0.0));
        let b = self.b().unwrap_or(Point2D::new(0.0, 0.0));
        let c = self.c().unwrap_or(Point2D::new(0.0, 0.0));
        Point2D::new((a.x() + b.x() + c.x()) / 3.0, (a.y() + b.y() + c.y()) / 3.0)
    }
}

impl HasMeasure for Triangle2D {
    fn measure(&self) -> GeometryMeasure {
        self.area()
    }
}

impl HasOrientation for Triangle2D {
    fn orientation(&self) -> Orientation2D {
        match (self.a(), self.b(), self.c()) {
            (Some(a), Some(b), Some(c)) => {
                let cross = (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
                if cross > 0.0 {
                    Orientation2D::CounterClockwise
                } else if cross < 0.0 {
                    Orientation2D::Clockwise
                } else {
                    Orientation2D::Collinear
                }
            }
            _ => Orientation2D::Collinear,
        }
    }
}

impl CanTranslate for Triangle2D {
    type Point = Point2D;

    fn translate<'a, L>(&mut self, translation_vector: &L)
    where
        L: crate::geometry::one_d::IsLine<'a, Self::Point>,
    {
        let mut polygon = PolygonFace2D::new(vec![self.a_id, self.b_id, self.c_id], self.vertex_table.clone());
        polygon.translate(translation_vector);
    }
}

impl CanRotate for Triangle2D {
    type Point = Point2D;

    fn rotate<'a, L>(&mut self, axis: &L, angle_radians: GeometryMeasure)
    where
        L: crate::geometry::one_d::IsLine<'a, Self::Point>,
    {
        let mut polygon = PolygonFace2D::new(vec![self.a_id, self.b_id, self.c_id], self.vertex_table.clone());
        polygon.rotate(axis, angle_radians);
    }
}

impl CanShear for Triangle2D {
    type Point = Point2D;

    fn shear<'a, L>(&mut self, shear_line: &L)
    where
        L: crate::geometry::one_d::IsLine<'a, Self::Point>,
    {
        let mut polygon = PolygonFace2D::new(vec![self.a_id, self.b_id, self.c_id], self.vertex_table.clone());
        polygon.shear(shear_line);
    }
}

impl CanMirror for Triangle2D {
    type Point = Point2D;
    type Normal = UnitVector2D;

    fn mirror<P>(&mut self, mirror_plane: &P)
    where
        P: IsPlane<Point = Self::Point, Normal = Self::Normal>,
    {
        let mut polygon = PolygonFace2D::new(vec![self.a_id, self.b_id, self.c_id], self.vertex_table.clone());
        polygon.mirror(mirror_plane);
    }
}

impl<'a> IsPolygon<'a, Point2D, UnitVector2D> for Triangle2D {
    fn vertex_ids(&self) -> Box<dyn Iterator<Item = PointId> + '_> {
        Box::new([self.a_id, self.b_id, self.c_id].into_iter())
    }

    fn set_vertex_id(&mut self, index: usize, point_id: PointId) -> Result<(), String> {
        match index {
            0 => self.a_id = point_id,
            1 => self.b_id = point_id,
            2 => self.c_id = point_id,
            _ => return Err(format!("vertex index {} is out of bounds", index)),
        }
        Ok(())
    }

    fn normal(&self) -> UnitVector2D {
        PolygonFace2D::new(vec![self.a_id, self.b_id, self.c_id], self.vertex_table.clone()).normal()
    }

    fn perimeter(&self) -> GeometryMeasure {
        PolygonFace2D::new(vec![self.a_id, self.b_id, self.c_id], self.vertex_table.clone()).perimeter()
    }

    fn area(&self) -> GeometryMeasure {
        PolygonFace2D::new(vec![self.a_id, self.b_id, self.c_id], self.vertex_table.clone()).area()
    }

    fn plane(&self) -> impl IsPlane<Point = Point2D, Normal = UnitVector2D> {
        Plane2D::new(self.centroid(), self.normal())
    }
}

impl<'a> IsTriangle<'a, Point2D, UnitVector2D> for Triangle2D {
    fn a_id(&self) -> PointId {
        self.a_id
    }

    fn set_a_id(&mut self, point_id: PointId) -> Result<(), String> {
        self.a_id = point_id;
        Ok(())
    }

    fn b_id(&self) -> PointId {
        self.b_id
    }

    fn set_b_id(&mut self, point_id: PointId) -> Result<(), String> {
        self.b_id = point_id;
        Ok(())
    }

    fn c_id(&self) -> PointId {
        self.c_id
    }

    fn set_c_id(&mut self, point_id: PointId) -> Result<(), String> {
        self.c_id = point_id;
        Ok(())
    }
}
