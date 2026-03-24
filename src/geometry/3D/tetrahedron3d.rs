//! Concrete tetrahedron type built from three-dimensional points.
//!
//! [`Tetrahedron3D`] represents a simple volumetric primitive defined by four point ids into a
//! shared point table. Faces and edges are derived on demand from those vertices.

use crate::geometry::common::{
    Canonicalize, GeometricPrimitive, GeometricPrimitive3D, GeometryMeasure, HasCentroid,
    HasEdges, HasMeasure, HasVertices, IsPlane, IsValid, PointId, Repair,
};
use crate::geometry::one_d::IsLine;
use crate::geometry::tables::SharedGeometryTable;
use crate::geometry::three_d::transform_support::{reflect_point_across_plane, rotate_point_around_axis};
use crate::geometry::three_d::{
    GeometryVector3D, IsTetrahedron, Line3D, Point3D, Triangle3D, UnitVector3D,
};
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanShear, CanTranslate};
use crate::geometry::two_d::IsPolygon;
use serde::Serialize;
use std::collections::HashSet;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

/// Concrete 3D tetrahedron implementation backed by four point ids.
#[derive(Debug, Clone, Serialize)]
pub struct Tetrahedron3D {
    a_id: PointId,
    b_id: PointId,
    c_id: PointId,
    d_id: PointId,
    #[serde(skip_serializing)]
    vertex_table: SharedGeometryTable<PointId, Point3D>,
}

impl PartialEq for Tetrahedron3D {
    fn eq(&self, other: &Self) -> bool {
        self.a_id == other.a_id
            && self.b_id == other.b_id
            && self.c_id == other.c_id
            && self.d_id == other.d_id
    }
}

impl Eq for Tetrahedron3D {}

impl Hash for Tetrahedron3D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.a_id.hash(state);
        self.b_id.hash(state);
        self.c_id.hash(state);
        self.d_id.hash(state);
    }
}

impl Tetrahedron3D {
    /// Creates a tetrahedron from four point ids and a shared point table.
    pub fn new(
        a_id: PointId,
        b_id: PointId,
        c_id: PointId,
        d_id: PointId,
        vertex_table: SharedGeometryTable<PointId, Point3D>,
    ) -> Self {
        Self {
            a_id,
            b_id,
            c_id,
            d_id,
            vertex_table,
        }
    }

    /// Returns the first vertex id.
    pub fn a_id(&self) -> PointId {
        self.a_id
    }

    /// Returns the second vertex id.
    pub fn b_id(&self) -> PointId {
        self.b_id
    }

    /// Returns the third vertex id.
    pub fn c_id(&self) -> PointId {
        self.c_id
    }

    /// Returns the fourth vertex id.
    pub fn d_id(&self) -> PointId {
        self.d_id
    }

    /// Resolves the first vertex from the shared point table.
    pub fn a(&self) -> Option<Point3D> {
        self.get_vertex(&self.a_id)
    }

    /// Resolves the second vertex from the shared point table.
    pub fn b(&self) -> Option<Point3D> {
        self.get_vertex(&self.b_id)
    }

    /// Resolves the third vertex from the shared point table.
    pub fn c(&self) -> Option<Point3D> {
        self.get_vertex(&self.c_id)
    }

    /// Resolves the fourth vertex from the shared point table.
    pub fn d(&self) -> Option<Point3D> {
        self.get_vertex(&self.d_id)
    }

    /// Returns the four triangular boundary faces of the tetrahedron.
    pub fn faces(&self) -> [Triangle3D; 4] {
        [
            Triangle3D::new(self.a_id, self.b_id, self.c_id, self.vertex_table.clone()),
            Triangle3D::new(self.a_id, self.b_id, self.d_id, self.vertex_table.clone()),
            Triangle3D::new(self.a_id, self.c_id, self.d_id, self.vertex_table.clone()),
            Triangle3D::new(self.b_id, self.c_id, self.d_id, self.vertex_table.clone()),
        ]
    }

    /// Returns the number of triangular faces.
    pub fn face_count(&self) -> usize {
        4
    }

    /// Returns the total surface area of the tetrahedron.
    pub fn surface_area(&self) -> GeometryMeasure {
        self.faces().iter().map(|face| face.area()).sum()
    }

    /// Returns the signed volume of the tetrahedron.
    pub fn signed_volume(&self) -> GeometryMeasure {
        let (Some(a), Some(b), Some(c), Some(d)) = (self.a(), self.b(), self.c(), self.d()) else {
            return 0.0;
        };
        let ab = [b.x() - a.x(), b.y() - a.y(), b.z() - a.z()];
        let ac = [c.x() - a.x(), c.y() - a.y(), c.z() - a.z()];
        let ad = [d.x() - a.x(), d.y() - a.y(), d.z() - a.z()];
        let cross = GeometryVector3D::new(
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        );
        (cross.x() * ad[0] + cross.y() * ad[1] + cross.z() * ad[2]) / 6.0
    }

    /// Returns the absolute tetrahedron volume.
    pub fn volume(&self) -> GeometryMeasure {
        self.signed_volume().abs()
    }

    fn unique_vertex_ids(&self) -> Vec<PointId> {
        let mut seen = HashSet::new();
        [self.a_id, self.b_id, self.c_id, self.d_id]
            .into_iter()
            .filter(|id| seen.insert(*id))
            .collect()
    }
}

impl Display for Tetrahedron3D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tetrahedron3D({:?}, {:?}, {:?}, {:?})",
            self.a_id, self.b_id, self.c_id, self.d_id
        )
    }
}

impl GeometricPrimitive for Tetrahedron3D {}
impl GeometricPrimitive3D for Tetrahedron3D {}

impl<'a> HasVertices<'a> for Tetrahedron3D {
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

impl HasEdges for Tetrahedron3D {
    type Edge = Line3D;

    fn edge_count(&self) -> usize {
        6
    }

    fn edge(&self, index: usize) -> Option<Self::Edge> {
        let (head, tail) = match index {
            0 => (self.a_id, self.b_id),
            1 => (self.a_id, self.c_id),
            2 => (self.a_id, self.d_id),
            3 => (self.b_id, self.c_id),
            4 => (self.b_id, self.d_id),
            5 => (self.c_id, self.d_id),
            _ => return None,
        };
        Some(Line3D::new(head, tail, self.vertex_table.clone()))
    }
}

impl HasCentroid for Tetrahedron3D {
    type Point = Point3D;

    fn centroid(&self) -> Self::Point {
        let a = self.a().unwrap_or(Point3D::new(0.0, 0.0, 0.0));
        let b = self.b().unwrap_or(Point3D::new(0.0, 0.0, 0.0));
        let c = self.c().unwrap_or(Point3D::new(0.0, 0.0, 0.0));
        let d = self.d().unwrap_or(Point3D::new(0.0, 0.0, 0.0));
        Point3D::new(
            (a.x() + b.x() + c.x() + d.x()) / 4.0,
            (a.y() + b.y() + c.y() + d.y()) / 4.0,
            (a.z() + b.z() + c.z() + d.z()) / 4.0,
        )
    }
}

impl HasMeasure for Tetrahedron3D {
    fn measure(&self) -> GeometryMeasure {
        self.volume()
    }
}

impl<'a> IsTetrahedron<'a, Point3D, UnitVector3D> for Tetrahedron3D {
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

    fn d_id(&self) -> PointId {
        self.d_id
    }

    fn set_d_id(&mut self, point_id: PointId) -> Result<(), String> {
        self.d_id = point_id;
        Ok(())
    }

    fn faces(&self) -> [Triangle3D; 4] {
        Tetrahedron3D::faces(self)
    }

    fn face_count(&self) -> usize {
        Tetrahedron3D::face_count(self)
    }

    fn surface_area(&self) -> GeometryMeasure {
        Tetrahedron3D::surface_area(self)
    }

    fn signed_volume(&self) -> GeometryMeasure {
        Tetrahedron3D::signed_volume(self)
    }

    fn volume(&self) -> GeometryMeasure {
        Tetrahedron3D::volume(self)
    }
}

impl IsValid for Tetrahedron3D {
    fn is_valid(&self) -> bool {
        self.unique_vertex_ids().len() == 4 && self.volume() > 1e-6
    }
}

impl Repair for Tetrahedron3D {
    fn repair(&mut self) -> Result<(), String> {
        if self.unique_vertex_ids().len() != 4 {
            return Err("tetrahedron vertices must be distinct".to_string());
        }
        if self.volume() <= 1e-6 {
            return Err("degenerate tetrahedron volume cannot be repaired automatically".to_string());
        }
        Ok(())
    }
}

impl Canonicalize for Tetrahedron3D {
    fn canonicalize(&mut self) {
        let mut ids = [self.a_id, self.b_id, self.c_id, self.d_id];
        ids.sort_by_key(|id| id.0);
        self.a_id = ids[0];
        self.b_id = ids[1];
        self.c_id = ids[2];
        self.d_id = ids[3];
    }
}

impl CanTranslate for Tetrahedron3D {
    type Point = Point3D;

    fn translate<'a, L>(&mut self, translation_vector: &L)
    where
        L: IsLine<'a, Self::Point>,
    {
        let (Some(head), Some(tail)) = (translation_vector.head(), translation_vector.tail()) else {
            return;
        };
        let delta = GeometryVector3D::new(tail.x() - head.x(), tail.y() - head.y(), tail.z() - head.z());
        for point_id in self.unique_vertex_ids() {
            if let Some(mut point) = self.get_vertex(&point_id) {
                point = point + delta;
                let _ = self.insert_vertex(point_id, point);
            }
        }
    }
}

impl CanRotate for Tetrahedron3D {
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

impl CanShear for Tetrahedron3D {
    type Point = Point3D;

    fn shear<'a, L>(&mut self, shear_line: &L)
    where
        L: IsLine<'a, Self::Point>,
    {
        let factor = shear_line.length();
        for point_id in self.unique_vertex_ids() {
            if let Some(mut point) = self.get_vertex(&point_id) {
                let coords = point.cartesian_components();
                point = Point3D::from_cartesian_components(
                    [coords[0] + factor * coords[1], coords[1], coords[2]],
                    point.coordinate_system(),
                );
                let _ = self.insert_vertex(point_id, point);
            }
        }
    }
}

impl CanMirror for Tetrahedron3D {
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
