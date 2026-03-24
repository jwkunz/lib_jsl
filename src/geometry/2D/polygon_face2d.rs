//! Concrete polygon face type built from two-dimensional points.

use crate::geometry::common::{
    GeometricPrimitive, GeometricPrimitive2D, GeometryMeasure, HasCentroid, HasEdges,
    HasMeasure, HasVertices, PointId,
};
use crate::geometry::one_d::IsLine;
use crate::geometry::tables::SharedGeometryTable;
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanShear, CanTranslate};
use crate::geometry::two_d::transform_support::{reflect_point_across_plane_2d, rotate_point_around_anchor_2d};
use crate::geometry::two_d::{HasOrientation, IsPolygon, Line2D, Plane2D, Point2D, UnitVector2D, Orientation2D};
use crate::geometry::three_d::IsPlane;
use serde::Serialize;
use std::collections::HashSet;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

/// Concrete 2D polygon face backed by ordered point ids.
#[derive(Debug, Clone, Serialize)]
pub struct PolygonFace2D {
    vertex_ids: Vec<PointId>,
    #[serde(skip_serializing)]
    vertex_table: SharedGeometryTable<PointId, Point2D>,
}

impl PartialEq for PolygonFace2D {
    fn eq(&self, other: &Self) -> bool {
        self.vertex_ids == other.vertex_ids
    }
}

impl Eq for PolygonFace2D {}

impl Hash for PolygonFace2D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertex_ids.hash(state);
    }
}

impl PolygonFace2D {
    /// Creates a polygon face from ordered point ids and a shared point table.
    pub fn new(vertex_ids: Vec<PointId>, vertex_table: SharedGeometryTable<PointId, Point2D>) -> Self {
        Self {
            vertex_ids,
            vertex_table,
        }
    }

    fn resolved_points(&self) -> Option<Vec<Point2D>> {
        self.vertex_ids
            .iter()
            .map(|id| self.get_vertex(id))
            .collect::<Option<Vec<_>>>()
    }

    fn unique_vertex_ids(&self) -> Vec<PointId> {
        let mut seen = HashSet::new();
        self.vertex_ids.iter().copied().filter(|id| seen.insert(*id)).collect()
    }
}

impl Display for PolygonFace2D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "PolygonFace2D(vertices={:?})", self.vertex_ids)
    }
}

impl GeometricPrimitive for PolygonFace2D {}
impl GeometricPrimitive2D for PolygonFace2D {}

impl<'a> HasVertices<'a> for PolygonFace2D {
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

impl HasEdges for PolygonFace2D {
    type Edge = Line2D;

    fn edge_count(&self) -> usize {
        self.vertex_ids.len().saturating_sub(1).max((self.vertex_ids.len() >= 3) as usize)
    }

    fn edge(&self, index: usize) -> Option<Self::Edge> {
        if self.vertex_ids.len() < 2 {
            return None;
        }
        let count = self.vertex_ids.len();
        if index >= count {
            return None;
        }
        let head = self.vertex_ids[index];
        let tail = self.vertex_ids[(index + 1) % count];
        Some(Line2D::new(head, tail, self.vertex_table.clone()))
    }
}

impl HasCentroid for PolygonFace2D {
    type Point = Point2D;

    fn centroid(&self) -> Self::Point {
        if let Some(points) = self.resolved_points() {
            let count = points.len() as GeometryMeasure;
            let sum = points
                .into_iter()
                .fold(Point2D::new(0.0, 0.0), |acc, point| acc + point);
            sum / count
        } else {
            Point2D::new(0.0, 0.0)
        }
    }
}

impl HasMeasure for PolygonFace2D {
    fn measure(&self) -> GeometryMeasure {
        self.area()
    }
}

impl HasOrientation for PolygonFace2D {
    fn orientation(&self) -> Orientation2D {
        let Some(points) = self.resolved_points() else {
            return Orientation2D::Collinear;
        };
        if points.len() < 3 {
            return Orientation2D::Collinear;
        }
        let signed_area = points
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let b = points[(i + 1) % points.len()];
                a[0] * b[1] - b[0] * a[1]
            })
            .sum::<GeometryMeasure>();
        if signed_area > 0.0 {
            Orientation2D::CounterClockwise
        } else if signed_area < 0.0 {
            Orientation2D::Clockwise
        } else {
            Orientation2D::Collinear
        }
    }
}

impl CanTranslate for PolygonFace2D {
    type Point = Point2D;

    fn translate<'a, L>(&mut self, translation_vector: &L)
    where
        L: IsLine<'a, Self::Point>,
    {
        let (Some(head), Some(tail)) = (translation_vector.head(), translation_vector.tail()) else {
            return;
        };
        let delta = tail - head;
        for point_id in self.unique_vertex_ids() {
            if let Some(mut point) = self.get_vertex(&point_id) {
                point = point + delta;
                let _ = self.insert_vertex(point_id, point);
            }
        }
    }
}

impl CanRotate for PolygonFace2D {
    type Point = Point2D;

    fn rotate<'a, L>(&mut self, axis: &L, angle_radians: GeometryMeasure)
    where
        L: IsLine<'a, Self::Point>,
    {
        let Some(origin) = axis.head() else {
            return;
        };
        for point_id in self.unique_vertex_ids() {
            if let Some(point) = self.get_vertex(&point_id) {
                let rotated = rotate_point_around_anchor_2d(point, origin, angle_radians);
                let _ = self.insert_vertex(point_id, rotated);
            }
        }
    }
}

impl CanShear for PolygonFace2D {
    type Point = Point2D;

    fn shear<'a, L>(&mut self, shear_line: &L)
    where
        L: IsLine<'a, Self::Point>,
    {
        let factor = shear_line.length();
        for point_id in self.unique_vertex_ids() {
            if let Some(mut point) = self.get_vertex(&point_id) {
                point[0] += factor * point[1];
                let _ = self.insert_vertex(point_id, point);
            }
        }
    }
}

impl CanMirror for PolygonFace2D {
    type Point = Point2D;
    type Normal = UnitVector2D;

    fn mirror<P>(&mut self, mirror_plane: &P)
    where
        P: IsPlane<Point = Self::Point, Normal = Self::Normal>,
    {
        for point_id in self.unique_vertex_ids() {
            if let Some(point) = self.get_vertex(&point_id) {
                let reflected = reflect_point_across_plane_2d(point, mirror_plane.point(), mirror_plane.normal());
                let _ = self.insert_vertex(point_id, reflected);
            }
        }
    }
}

impl<'a> IsPolygon<'a, Point2D, UnitVector2D> for PolygonFace2D {
    fn vertex_ids(&self) -> Box<dyn Iterator<Item = PointId> + '_> {
        Box::new(self.vertex_ids.iter().copied())
    }

    fn set_vertex_id(&mut self, index: usize, point_id: PointId) -> Result<(), String> {
        if let Some(slot) = self.vertex_ids.get_mut(index) {
            *slot = point_id;
            Ok(())
        } else {
            Err(format!("vertex index {} is out of bounds", index))
        }
    }

    fn normal(&self) -> UnitVector2D {
        match self.orientation() {
            Orientation2D::Clockwise => UnitVector2D::new(0.0, -1.0),
            Orientation2D::CounterClockwise => UnitVector2D::new(0.0, 1.0),
            Orientation2D::Collinear => UnitVector2D::new(0.0, 1.0),
        }
    }

    fn perimeter(&self) -> GeometryMeasure {
        (0..self.vertex_ids.len())
            .filter_map(|index| self.edge(index))
            .map(|edge| edge.length())
            .sum()
    }

    fn area(&self) -> GeometryMeasure {
        let Some(points) = self.resolved_points() else {
            return 0.0;
        };
        if points.len() < 3 {
            return 0.0;
        }
        points
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let b = points[(i + 1) % points.len()];
                a[0] * b[1] - b[0] * a[1]
            })
            .sum::<GeometryMeasure>()
            .abs()
            / 2.0
    }

    fn plane(&self) -> impl IsPlane<Point = Point2D, Normal = UnitVector2D> {
        Plane2D::new(self.centroid(), self.normal())
    }
}
