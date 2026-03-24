//! Concrete polygon face type built from three-dimensional points.

use crate::geometry::common::{
    GeometricPrimitive, GeometricPrimitive3D, GeometryMeasure, HasCentroid, HasEdges,
    HasMeasure, HasVertices, PointId,
};
use crate::geometry::one_d::{IsLine, Line3D};
use crate::geometry::tables::SharedGeometryTable;
use crate::geometry::three_d::transform_support::{reflect_point_across_plane, rotate_point_around_axis};
use crate::geometry::three_d::{IsPlane, Plane3D, UnitVector3D};
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanShear, CanTranslate};
use crate::geometry::two_d::{HasOrientation, IsPolygon, Orientation2D};
use crate::geometry::zero_d::Point3D;
use serde::Serialize;
use std::collections::HashSet;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

/// Concrete 3D polygon face backed by ordered point ids.
#[derive(Debug, Clone, Serialize)]
pub struct PolygonFace3D {
    vertex_ids: Vec<PointId>,
    #[serde(skip_serializing)]
    vertex_table: SharedGeometryTable<PointId, Point3D>,
}

impl PartialEq for PolygonFace3D {
    fn eq(&self, other: &Self) -> bool {
        self.vertex_ids == other.vertex_ids
    }
}

impl Eq for PolygonFace3D {}

impl Hash for PolygonFace3D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertex_ids.hash(state);
    }
}

impl PolygonFace3D {
    /// Creates a polygon face from ordered point ids and a shared point table.
    pub fn new(vertex_ids: Vec<PointId>, vertex_table: SharedGeometryTable<PointId, Point3D>) -> Self {
        Self {
            vertex_ids,
            vertex_table,
        }
    }

    fn resolved_points(&self) -> Option<Vec<Point3D>> {
        self.vertex_ids
            .iter()
            .map(|id| self.get_vertex(id))
            .collect::<Option<Vec<_>>>()
    }

    fn unique_vertex_ids(&self) -> Vec<PointId> {
        let mut seen = HashSet::new();
        self.vertex_ids
            .iter()
            .copied()
            .filter(|id| seen.insert(*id))
            .collect()
    }
}

impl Display for PolygonFace3D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "PolygonFace3D(vertices={:?})", self.vertex_ids)
    }
}

impl GeometricPrimitive for PolygonFace3D {}
impl GeometricPrimitive3D for PolygonFace3D {}

impl<'a> HasVertices<'a> for PolygonFace3D {
    type Vertex = Point3D;
    type VertexTable = SharedGeometryTable<PointId, Point3D>;

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

impl HasEdges for PolygonFace3D {
    type Edge = Line3D;

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
        Some(Line3D::new(head, tail, self.vertex_table.clone()))
    }
}

impl HasCentroid for PolygonFace3D {
    type Point = Point3D;

    fn centroid(&self) -> Self::Point {
        if let Some(points) = self.resolved_points() {
            let count = points.len() as GeometryMeasure;
            let sum = points
                .into_iter()
                .fold(Point3D::new(0.0, 0.0, 0.0), |acc, point| acc + point);
            sum / count
        } else {
            Point3D::new(0.0, 0.0, 0.0)
        }
    }
}

impl HasMeasure for PolygonFace3D {
    fn measure(&self) -> GeometryMeasure {
        self.area()
    }
}

impl HasOrientation for PolygonFace3D {
    fn orientation(&self) -> Orientation2D {
        let Some(points) = self.resolved_points() else {
            return Orientation2D::Collinear;
        };
        if points.len() < 3 {
            return Orientation2D::Collinear;
        }
        let mut signed_area = 0.0;
        for i in 0..points.len() {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            signed_area += a[0] * b[1] - b[0] * a[1];
        }
        if signed_area > 0.0 {
            Orientation2D::CounterClockwise
        } else if signed_area < 0.0 {
            Orientation2D::Clockwise
        } else {
            Orientation2D::Collinear
        }
    }
}

impl CanTranslate for PolygonFace3D {
    type Point = Point3D;

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

impl CanRotate for PolygonFace3D {
    type Point = Point3D;

    fn rotate<'a, L>(&mut self, axis: &L, angle_radians: GeometryMeasure)
    where
        L: IsLine<'a, Self::Point>,
    {
        let Some(origin) = axis.head() else {
            return;
        };
        let direction = axis.direction();
        for point_id in self.unique_vertex_ids() {
            if let Some(point) = self.get_vertex(&point_id) {
                let rotated = rotate_point_around_axis(point, origin, &direction, angle_radians);
                let _ = self.insert_vertex(point_id, rotated);
            }
        }
    }
}

impl CanShear for PolygonFace3D {
    type Point = Point3D;

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

impl CanMirror for PolygonFace3D {
    type Point = Point3D;
    type Normal = UnitVector3D;

    fn mirror<P>(&mut self, mirror_plane: &P)
    where
        P: IsPlane<Point = Self::Point, Normal = Self::Normal>,
    {
        for point_id in self.unique_vertex_ids() {
            if let Some(point) = self.get_vertex(&point_id) {
                let reflected = reflect_point_across_plane(point, mirror_plane.point(), mirror_plane.normal());
                let _ = self.insert_vertex(point_id, reflected);
            }
        }
    }
}

impl<'a> IsPolygon<'a, Point3D, UnitVector3D> for PolygonFace3D {
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

    fn normal(&self) -> UnitVector3D {
        let Some(points) = self.resolved_points() else {
            return UnitVector3D::new(0.0, 0.0, 1.0);
        };
        if points.len() < 3 {
            return UnitVector3D::new(0.0, 0.0, 1.0);
        }
        let a = points[0];
        let b = points[1];
        let c = points[2];
        let ab = b - a;
        let ac = c - a;
        UnitVector3D::from_point(Point3D::new(
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ))
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
        let origin = points[0];
        let mut area = 0.0;
        for i in 1..points.len() - 1 {
            let a = points[i] - origin;
            let b = points[i + 1] - origin;
            let cross = Point3D::new(
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            );
            area += ((cross[0] * cross[0]) + (cross[1] * cross[1]) + (cross[2] * cross[2])).sqrt() / 2.0;
        }
        area
    }

    fn plane(&self) -> impl IsPlane<Point = Point3D, Normal = UnitVector3D> {
        Plane3D::new(self.centroid(), self.normal())
    }
}
