//! Concrete mesh type built from three-dimensional polygon faces.
//!
//! [`Mesh3D`] is the current top-level concrete aggregate primitive. It stores face ids into a
//! shared face table and resolves those faces against the same point table used by the rest of the
//! geometry graph.

use crate::geometry::common::{
    FaceId, GeometricPrimitive, GeometricPrimitive3D, GeometryMeasure, HasEdges, HasFaces,
    HasVertices, PointId,
};
use crate::geometry::one_d::{IsLine, Line3D};
use crate::geometry::tables::SharedGeometryTable;
use crate::geometry::three_d::{IsMesh, UnitVector3D};
use crate::geometry::two_d::{IsPolygon, PolygonFace3D};
use crate::geometry::zero_d::Point3D;
use serde::Serialize;
use std::collections::HashSet;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

/// Concrete 3D mesh implementation referencing polygon faces through ids.
#[derive(Debug, Clone, Serialize)]
pub struct Mesh3D {
    face_ids: Vec<FaceId>,
    #[serde(skip_serializing)]
    vertex_table: SharedGeometryTable<PointId, Point3D>,
    #[serde(skip_serializing)]
    face_table: SharedGeometryTable<FaceId, PolygonFace3D>,
}

impl PartialEq for Mesh3D {
    fn eq(&self, other: &Self) -> bool {
        self.face_ids == other.face_ids
    }
}

impl Eq for Mesh3D {}

impl Hash for Mesh3D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.face_ids.hash(state);
    }
}

impl Mesh3D {
    /// Creates a mesh from ordered face ids and shared point and face tables.
    ///
    /// Faces are resolved lazily through `face_table`, and each face in turn resolves its vertices
    /// through `vertex_table`.
    pub fn new(
        face_ids: Vec<FaceId>,
        vertex_table: SharedGeometryTable<PointId, Point3D>,
        face_table: SharedGeometryTable<FaceId, PolygonFace3D>,
    ) -> Self {
        Self {
            face_ids,
            vertex_table,
            face_table,
        }
    }

    fn resolved_edges(&self) -> Vec<Line3D> {
        let mut seen = HashSet::new();
        let mut edges = Vec::new();
        for face_id in &self.face_ids {
            if let Some(face) = self.get_face(face_id) {
                for edge_index in 0..face.edge_count() {
                    if let Some(edge) = face.edge(edge_index) {
                        let head = edge.head_id().0.min(edge.tail_id().0);
                        let tail = edge.head_id().0.max(edge.tail_id().0);
                        if seen.insert((head, tail)) {
                            edges.push(edge);
                        }
                    }
                }
            }
        }
        edges
    }
}

impl Display for Mesh3D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Mesh3D(faces={:?})", self.face_ids)
    }
}

impl GeometricPrimitive for Mesh3D {}
impl GeometricPrimitive3D for Mesh3D {}

impl<'a> HasVertices<'a> for Mesh3D {
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

impl<'a> HasFaces<'a> for Mesh3D {
    type Point = Point3D;
    type Normal = UnitVector3D;
    type Face = PolygonFace3D;
    type FaceTable = SharedGeometryTable<FaceId, PolygonFace3D>;

    fn face_table(&self) -> &Self::FaceTable {
        &self.face_table
    }

    fn face_table_mut(&mut self) -> &mut Self::FaceTable {
        &mut self.face_table
    }

    fn set_face_table(&mut self, table: &'a mut Self::FaceTable) {
        self.face_table = table.clone();
    }
}

impl HasEdges for Mesh3D {
    type Edge = Line3D;

    fn edge_count(&self) -> usize {
        self.resolved_edges().len()
    }

    fn edge(&self, index: usize) -> Option<Self::Edge> {
        self.resolved_edges().into_iter().nth(index)
    }
}

impl<'a> IsMesh<'a, Point3D, UnitVector3D> for Mesh3D {
    fn face_ids(&self) -> Box<dyn Iterator<Item = FaceId> + '_> {
        Box::new(self.face_ids.iter().copied())
    }

    fn set_face_id(&mut self, index: usize, face_id: FaceId) -> Result<(), String> {
        if let Some(slot) = self.face_ids.get_mut(index) {
            *slot = face_id;
            Ok(())
        } else {
            Err(format!("face index {} is out of bounds", index))
        }
    }

    fn surface_area(&self) -> GeometryMeasure {
        self.face_ids
            .iter()
            .filter_map(|face_id| self.get_face(face_id))
            .map(|face| face.area())
            .sum()
    }
}
