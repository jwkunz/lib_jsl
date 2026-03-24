//! Concrete polygon face type built from two-dimensional points.

use crate::geometry::common::{
    GeometricPrimitive, GeometricPrimitive2D, GeometryMeasure, HasCentroid, HasEdges,
    HasMeasure, HasVertices, IsPlane, PointId,
};
use crate::geometry::one_d::IsLine;
use crate::geometry::tables::SharedGeometryTable;
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanShear, CanTranslate};
use crate::geometry::two_d::transform_support::{reflect_point_across_plane_2d, rotate_point_around_anchor_2d};
use crate::geometry::two_d::{
    FreeVector2D, HasOrientation, IsPolygon, Line2D, Plane2D, CoordinateVector2D, Triangle2D,
    UnitVector2D, Orientation2D,
};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

/// Concrete 2D polygon face backed by ordered point ids.
#[derive(Debug, Clone, Serialize)]
pub struct PolygonFace2D {
    vertex_ids: Vec<PointId>,
    #[serde(skip_serializing)]
    vertex_table: SharedGeometryTable<PointId, CoordinateVector2D>,
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
    pub fn new(vertex_ids: Vec<PointId>, vertex_table: SharedGeometryTable<PointId, CoordinateVector2D>) -> Self {
        Self {
            vertex_ids,
            vertex_table,
        }
    }

    /// Decomposes the polygon into a fan of triangles rooted at the first vertex.
    pub fn triangulate(&self) -> Vec<Triangle2D> {
        if self.vertex_ids.len() < 3 {
            return Vec::new();
        }
        let root = self.vertex_ids[0];
        (1..self.vertex_ids.len() - 1)
            .map(|index| {
                Triangle2D::new(
                    root,
                    self.vertex_ids[index],
                    self.vertex_ids[index + 1],
                    self.vertex_table.clone(),
                )
            })
            .collect()
    }

    /// Composes a polygon from a set of triangles that share a common boundary.
    pub fn from_triangles(triangles: &[Triangle2D]) -> Result<Self, String> {
        if triangles.is_empty() {
            return Err("cannot compose polygon from an empty triangle set".to_string());
        }

        let vertex_table = triangles[0].vertex_table().clone();
        let mut boundary_counts: HashMap<(u64, u64), usize> = HashMap::new();
        let mut boundary_direction: HashMap<(u64, u64), (PointId, PointId)> = HashMap::new();

        for triangle in triangles {
            let triangle_table = triangle.vertex_table().clone();
            if !std::rc::Rc::ptr_eq(&vertex_table, &triangle_table) {
                return Err("all triangles must share the same vertex table".to_string());
            }

            let ids: Vec<_> = triangle.vertex_ids().collect();
            for (head, tail) in [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
                let key = (head.0.min(tail.0), head.0.max(tail.0));
                *boundary_counts.entry(key).or_insert(0) += 1;
                boundary_direction.entry(key).or_insert((head, tail));
            }
        }

        let boundary_edges: Vec<_> = boundary_counts
            .iter()
            .filter(|(_, count)| **count == 1)
            .filter_map(|(key, _)| boundary_direction.get(key).copied())
            .collect();

        if boundary_edges.len() < 3 {
            return Err("triangle set does not expose a valid polygon boundary".to_string());
        }

        let mut adjacency: HashMap<PointId, Vec<PointId>> = HashMap::new();
        for (head, tail) in &boundary_edges {
            adjacency.entry(*head).or_default().push(*tail);
            adjacency.entry(*tail).or_default().push(*head);
        }

        if adjacency.values().any(|neighbors| neighbors.len() != 2) {
            return Err("triangle set does not form a single closed polygon boundary".to_string());
        }

        let start = adjacency
            .keys()
            .min_by_key(|id| id.0)
            .copied()
            .ok_or_else(|| "failed to determine polygon boundary start".to_string())?;

        let mut ordered = vec![start];
        let mut previous = None;
        let mut current = start;

        loop {
            let neighbors = adjacency
                .get(&current)
                .ok_or_else(|| "missing boundary adjacency entry".to_string())?;
            let next = neighbors
                .iter()
                .copied()
                .find(|candidate| Some(*candidate) != previous)
                .ok_or_else(|| "failed to walk polygon boundary".to_string())?;

            if next == start {
                break;
            }
            if ordered.contains(&next) {
                return Err("triangle set boundary is self-intersecting or duplicated".to_string());
            }

            ordered.push(next);
            previous = Some(current);
            current = next;
        }

        let desired_orientation = triangles[0].orientation();
        let candidate = Self::new(ordered.clone(), vertex_table.clone());
        if desired_orientation != Orientation2D::Collinear
            && candidate.orientation() != desired_orientation
        {
            let mut reversed = vec![ordered[0]];
            reversed.extend(ordered[1..].iter().rev().copied());
            ordered = reversed;
        }

        Ok(Self::new(ordered, vertex_table))
    }

    fn resolved_points(&self) -> Option<Vec<CoordinateVector2D>> {
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
    type Vertex = CoordinateVector2D;
    type VertexTable = SharedGeometryTable<PointId, CoordinateVector2D>;

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
    type Point = CoordinateVector2D;

    fn centroid(&self) -> Self::Point {
        if let Some(points) = self.resolved_points() {
            let count = points.len() as GeometryMeasure;
            let sum = points
                .into_iter()
                .fold(CoordinateVector2D::new(0.0, 0.0), |acc, point| acc + point);
            sum / count
        } else {
            CoordinateVector2D::new(0.0, 0.0)
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
                a.x() * b.y() - b.x() * a.y()
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
    type Point = CoordinateVector2D;

    fn translate<'a, L>(&mut self, translation_vector: &L)
    where
        L: IsLine<'a, Self::Point>,
    {
        let (Some(head), Some(tail)) = (translation_vector.head(), translation_vector.tail()) else {
            return;
        };
        let delta = FreeVector2D::new(tail.x() - head.x(), tail.y() - head.y());
        for point_id in self.unique_vertex_ids() {
            if let Some(mut point) = self.get_vertex(&point_id) {
                point = point + delta;
                let _ = self.insert_vertex(point_id, point);
            }
        }
    }
}

impl CanRotate for PolygonFace2D {
    type Point = CoordinateVector2D;

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
    type Point = CoordinateVector2D;

    fn shear<'a, L>(&mut self, shear_line: &L)
    where
        L: IsLine<'a, Self::Point>,
    {
        let factor = shear_line.length();
        for point_id in self.unique_vertex_ids() {
            if let Some(mut point) = self.get_vertex(&point_id) {
                let coords = point.cartesian_components();
                point = CoordinateVector2D::from_cartesian_components(
                    [coords[0] + factor * coords[1], coords[1]],
                    point.coordinate_system(),
                );
                let _ = self.insert_vertex(point_id, point);
            }
        }
    }
}

impl CanMirror for PolygonFace2D {
    type Point = CoordinateVector2D;
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

impl<'a> IsPolygon<'a, CoordinateVector2D, UnitVector2D> for PolygonFace2D {
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
                a.x() * b.y() - b.x() * a.y()
            })
            .sum::<GeometryMeasure>()
            .abs()
            / 2.0
    }

    fn plane(&self) -> impl IsPlane<Point = CoordinateVector2D, Normal = UnitVector2D> {
        Plane2D::new(self.centroid(), self.normal())
    }
}
