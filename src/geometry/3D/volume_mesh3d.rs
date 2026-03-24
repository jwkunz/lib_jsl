//! Concrete volume mesh type built from three-dimensional tetrahedra.
//!
//! [`VolumeMesh3D`] models a tetrahedral volume mesh in the same keyed-table style as the rest of
//! the concrete geometry graph. The mesh stores ordered tetrahedron ids into a shared tetrahedron
//! table, and each tetrahedron ultimately resolves back to the shared point table. Boundary
//! surfaces are materialized into caller-provided face tables so the resulting surface meshes stay
//! connected to the broader geometry graph.

use crate::geometry::common::{
    Canonicalize, FaceId, GeometricPrimitive, GeometricPrimitive3D, GeometryMeasure, HasCentroid,
    HasEdges, HasTetrahedra, HasVertices, IsGeometryTable, IsPlane, IsValid, PointId, Repair,
    TetrahedronId,
};
use crate::geometry::one_d::IsLine;
use crate::geometry::tables::{HashGeometryTable, SharedGeometryTable};
use crate::geometry::three_d::{
    GeometryVector3D, IsVolumeMesh, Line3D, Point3D, PolygonFace3D, SurfaceMesh3D,
    Tetrahedron3D, Triangle3D, UnitVector3D,
};
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanShear, CanTranslate};
use crate::geometry::two_d::{IsPolygon, IsTriangle};
use serde::Serialize;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

/// Concrete 3D tetrahedral volume mesh implementation backed by tetrahedron ids.
#[derive(Debug, Clone, Serialize)]
pub struct VolumeMesh3D {
    tetrahedron_ids: Vec<TetrahedronId>,
    #[serde(skip_serializing)]
    vertex_table: SharedGeometryTable<PointId, Point3D>,
    #[serde(skip_serializing)]
    tetrahedron_table: SharedGeometryTable<TetrahedronId, Tetrahedron3D>,
}

impl PartialEq for VolumeMesh3D {
    fn eq(&self, other: &Self) -> bool {
        self.tetrahedron_ids == other.tetrahedron_ids
    }
}

impl Eq for VolumeMesh3D {}

impl Hash for VolumeMesh3D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tetrahedron_ids.hash(state);
    }
}

impl VolumeMesh3D {
    /// Creates a volume mesh from ordered tetrahedron ids and shared point/tetrahedron tables.
    ///
    /// Each tetrahedron id is resolved through `tetrahedron_table`, and the tetrahedra in that
    /// table in turn resolve their vertices through `vertex_table`.
    pub fn new(
        tetrahedron_ids: Vec<TetrahedronId>,
        vertex_table: SharedGeometryTable<PointId, Point3D>,
        tetrahedron_table: SharedGeometryTable<TetrahedronId, Tetrahedron3D>,
    ) -> Self {
        Self {
            tetrahedron_ids,
            vertex_table,
            tetrahedron_table,
        }
    }

    /// Creates a volume mesh from tetrahedra, inferring the shared point table and building a
    /// dedicated tetrahedron table with sequential ids.
    pub fn from_tetrahedra(tetrahedra: Vec<Tetrahedron3D>) -> Result<Self, String> {
        let vertex_table = tetrahedra
            .first()
            .map(|tetrahedron| tetrahedron.vertex_table().clone())
            .ok_or_else(|| "cannot compose volume mesh from an empty tetrahedron set".to_string())?;

        for tetrahedron in &tetrahedra {
            if !Rc::ptr_eq(&vertex_table, tetrahedron.vertex_table()) {
                return Err("all tetrahedra must share the same vertex table".to_string());
            }
        }

        let tetrahedron_table = Rc::new(RefCell::new(HashGeometryTable::new()));
        let mut tetrahedron_ids = Vec::with_capacity(tetrahedra.len());

        for (index, tetrahedron) in tetrahedra.into_iter().enumerate() {
            let tetrahedron_id = TetrahedronId(index as u64 + 1);
            tetrahedron_table
                .borrow_mut()
                .insert(tetrahedron_id, tetrahedron)?;
            tetrahedron_ids.push(tetrahedron_id);
        }

        Ok(Self::new(tetrahedron_ids, vertex_table, tetrahedron_table))
    }

    /// Resolves and returns all tetrahedra in mesh order.
    pub fn tetrahedra(&self) -> Vec<Tetrahedron3D> {
        self.tetrahedron_ids
            .iter()
            .filter_map(|tetrahedron_id| self.get_tetrahedron(tetrahedron_id))
            .collect()
    }

    /// Returns the number of tetrahedra in the mesh.
    pub fn tetrahedron_count(&self) -> usize {
        self.tetrahedron_ids.len()
    }

    /// Resolves and returns a tetrahedron by mesh-local index.
    pub fn tetrahedron(&self, index: usize) -> Option<Tetrahedron3D> {
        self.tetrahedron_ids
            .get(index)
            .and_then(|tetrahedron_id| self.get_tetrahedron(tetrahedron_id))
    }

    /// Returns the total enclosed volume of the mesh.
    pub fn volume(&self) -> GeometryMeasure {
        self.tetrahedra().iter().map(Tetrahedron3D::volume).sum()
    }

    /// Returns the total boundary surface area of the mesh.
    pub fn surface_area(&self) -> GeometryMeasure {
        self.boundary_triangles()
            .iter()
            .map(|triangle| triangle.area())
            .sum()
    }

    /// Returns the boundary triangles exposed by the tetrahedral cells.
    pub fn boundary_triangles(&self) -> Vec<Triangle3D> {
        let mut counts: HashMap<(u64, u64, u64), (usize, Triangle3D)> = HashMap::new();

        for tetrahedron in self.tetrahedra() {
            for face in tetrahedron.faces() {
                let mut key = [face.a_id().0, face.b_id().0, face.c_id().0];
                key.sort_unstable();
                let key = (key[0], key[1], key[2]);
                counts
                    .entry(key)
                    .and_modify(|entry| entry.0 += 1)
                    .or_insert((1, face));
            }
        }

        counts
            .into_values()
            .filter_map(|(count, triangle)| (count == 1).then_some(triangle))
            .collect()
    }

    /// Returns the boundary faces as polygon-face wrappers over the boundary triangles.
    pub fn boundary_faces(&self) -> Vec<PolygonFace3D> {
        self.boundary_triangles()
            .into_iter()
            .map(|triangle| {
                PolygonFace3D::new(
                    vec![triangle.a_id(), triangle.b_id(), triangle.c_id()],
                    self.vertex_table.clone(),
                )
            })
            .collect()
    }

    /// Materializes the boundary surface into a shared face table and returns a connected surface mesh.
    pub fn surface_mesh(
        &self,
        face_table: SharedGeometryTable<FaceId, PolygonFace3D>,
    ) -> Result<SurfaceMesh3D, String> {
        let mut face_ids = Vec::new();
        let next_face_id = face_table
            .borrow()
            .iter()
            .map(|(face_id, _)| face_id.0)
            .max()
            .unwrap_or(0)
            + 1;

        for (index, face) in self.boundary_faces().into_iter().enumerate() {
            let face_id = FaceId(next_face_id + index as u64);
            face_table.borrow_mut().insert(face_id, face)?;
            face_ids.push(face_id);
        }

        Ok(SurfaceMesh3D::new(face_ids, self.vertex_table.clone(), face_table))
    }

    fn unique_vertex_ids(&self) -> Vec<PointId> {
        let mut seen = HashSet::new();
        self.tetrahedra()
            .into_iter()
            .flat_map(|tetrahedron| {
                [
                    tetrahedron.a_id(),
                    tetrahedron.b_id(),
                    tetrahedron.c_id(),
                    tetrahedron.d_id(),
                ]
            })
            .filter(|id| seen.insert(*id))
            .collect()
    }

    fn resolved_edges(&self) -> Vec<Line3D> {
        let mut seen = HashSet::new();
        let mut edges = Vec::new();

        for tetrahedron in self.tetrahedra() {
            for edge_index in 0..tetrahedron.edge_count() {
                if let Some(edge) = tetrahedron.edge(edge_index) {
                    let head = edge.head_id().0.min(edge.tail_id().0);
                    let tail = edge.head_id().0.max(edge.tail_id().0);
                    if seen.insert((head, tail)) {
                        edges.push(edge);
                    }
                }
            }
        }

        edges
    }
}

impl Display for VolumeMesh3D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "VolumeMesh3D(tetrahedra={})", self.tetrahedron_ids.len())
    }
}

impl GeometricPrimitive for VolumeMesh3D {}
impl GeometricPrimitive3D for VolumeMesh3D {}

impl<'a> HasVertices<'a> for VolumeMesh3D {
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

        for tetrahedron_id in self.tetrahedron_ids.clone() {
            if let Some(mut tetrahedron) = self.get_tetrahedron(&tetrahedron_id) {
                tetrahedron.set_vertex_table(table);
                let _ = self.insert_tetrahedron(tetrahedron_id, tetrahedron);
            }
        }
    }
}

impl<'a> HasTetrahedra<'a> for VolumeMesh3D {
    type Point = Point3D;
    type Normal = UnitVector3D;
    type Tetrahedron = Tetrahedron3D;
    type TetrahedronTable = SharedGeometryTable<TetrahedronId, Tetrahedron3D>;

    fn tetrahedron_table(&self) -> &Self::TetrahedronTable {
        &self.tetrahedron_table
    }

    fn tetrahedron_table_mut(&mut self) -> &mut Self::TetrahedronTable {
        &mut self.tetrahedron_table
    }

    fn set_tetrahedron_table(&mut self, table: &'a mut Self::TetrahedronTable) {
        self.tetrahedron_table = table.clone();
    }
}

impl HasEdges for VolumeMesh3D {
    type Edge = Line3D;

    fn edge_count(&self) -> usize {
        self.resolved_edges().len()
    }

    fn edge(&self, index: usize) -> Option<Self::Edge> {
        self.resolved_edges().into_iter().nth(index)
    }
}

impl HasCentroid for VolumeMesh3D {
    type Point = Point3D;

    fn centroid(&self) -> Self::Point {
        let total_volume = self.volume();
        if total_volume <= 1e-6 {
            return Point3D::new(0.0, 0.0, 0.0);
        }

        let weighted_sum = self.tetrahedra().into_iter().fold(
            Point3D::new(0.0, 0.0, 0.0),
            |acc, tetrahedron| acc + tetrahedron.centroid() * tetrahedron.volume(),
        );

        weighted_sum / total_volume
    }
}

impl IsValid for VolumeMesh3D {
    fn is_valid(&self) -> bool {
        !self.tetrahedron_ids.is_empty()
            && self
                .tetrahedron_ids
                .iter()
                .all(|tetrahedron_id| self.contains_tetrahedron(tetrahedron_id))
            && self
                .tetrahedra()
                .iter()
                .all(|tetrahedron| tetrahedron.is_valid())
    }
}

impl Repair for VolumeMesh3D {
    fn repair(&mut self) -> Result<(), String> {
        for tetrahedron_id in self.tetrahedron_ids.clone() {
            let Some(mut tetrahedron) = self.get_tetrahedron(&tetrahedron_id) else {
                return Err(format!(
                    "tetrahedron id {:?} is missing from the tetrahedron table",
                    tetrahedron_id
                ));
            };
            tetrahedron.repair()?;
            self.insert_tetrahedron(tetrahedron_id, tetrahedron)?;
        }
        Ok(())
    }
}

impl Canonicalize for VolumeMesh3D {
    fn canonicalize(&mut self) {
        for tetrahedron_id in self.tetrahedron_ids.clone() {
            if let Some(mut tetrahedron) = self.get_tetrahedron(&tetrahedron_id) {
                tetrahedron.canonicalize();
                let _ = self.insert_tetrahedron(tetrahedron_id, tetrahedron);
            }
        }

        let mut ordered_ids = self
            .tetrahedron_ids
            .iter()
            .copied()
            .map(|tetrahedron_id| {
                let sort_key = self
                    .get_tetrahedron(&tetrahedron_id)
                    .map(|tetrahedron| {
                        (
                            tetrahedron.a_id().0,
                            tetrahedron.b_id().0,
                            tetrahedron.c_id().0,
                            tetrahedron.d_id().0,
                        )
                    })
                    .unwrap_or((u64::MAX, u64::MAX, u64::MAX, u64::MAX));
                (sort_key, tetrahedron_id)
            })
            .collect::<Vec<_>>();
        ordered_ids.sort_by_key(|(sort_key, _)| *sort_key);
        self.tetrahedron_ids = ordered_ids
            .into_iter()
            .map(|(_, tetrahedron_id)| tetrahedron_id)
            .collect();
    }
}

impl CanTranslate for VolumeMesh3D {
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

impl CanRotate for VolumeMesh3D {
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
                let rotated = crate::geometry::three_d::transform_support::rotate_point_around_axis(
                    point,
                    origin,
                    &direction,
                    angle_radians,
                );
                let _ = self.insert_vertex(point_id, rotated);
            }
        }
    }
}

impl CanShear for VolumeMesh3D {
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

impl CanMirror for VolumeMesh3D {
    type Point = Point3D;
    type Normal = UnitVector3D;

    fn mirror<P>(&mut self, mirror_plane: &P)
    where
        P: IsPlane<Point = Self::Point, Normal = Self::Normal>,
    {
        for point_id in self.unique_vertex_ids() {
            if let Some(point) = self.get_vertex(&point_id) {
                let reflected = crate::geometry::three_d::transform_support::reflect_point_across_plane(
                    point,
                    mirror_plane.point(),
                    mirror_plane.normal(),
                );
                let _ = self.insert_vertex(point_id, reflected);
            }
        }
    }
}

impl<'a> IsVolumeMesh<'a, Point3D, UnitVector3D> for VolumeMesh3D {
    fn tetrahedron_ids(&self) -> Box<dyn Iterator<Item = TetrahedronId> + '_> {
        Box::new(self.tetrahedron_ids.iter().copied())
    }

    fn set_tetrahedron_id(&mut self, index: usize, tetrahedron_id: TetrahedronId) -> Result<(), String> {
        if let Some(slot) = self.tetrahedron_ids.get_mut(index) {
            *slot = tetrahedron_id;
            Ok(())
        } else {
            Err(format!("tetrahedron index {} is out of bounds", index))
        }
    }

    fn volume(&self) -> GeometryMeasure {
        VolumeMesh3D::volume(self)
    }
}
