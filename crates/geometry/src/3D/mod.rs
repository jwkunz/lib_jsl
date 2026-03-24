//! Three-dimensional geometry traits.

/// Concrete three-dimensional line implementation.
pub mod line3d;
/// Concrete three-dimensional free-vector implementation.
pub mod free_vector3d;
/// Coordinate-system conversion helpers for 3D concrete coordinate vectors.
pub(crate) mod coordinate_conversions;
/// Concrete three-dimensional surface mesh implementation.
pub mod surface_mesh3d;
/// Concrete three-dimensional plane implementation.
pub mod plane3d;
/// Concrete three-dimensional coordinate-vector implementation.
pub mod coordinate_vector3d;
/// Concrete three-dimensional polygon face implementation.
pub mod polygon_face3d;
/// Concrete three-dimensional triangle implementation.
pub mod triangle3d;
/// Concrete three-dimensional tetrahedron implementation.
pub mod tetrahedron3d;
/// Concrete three-dimensional volume mesh implementation.
pub mod volume_mesh3d;
pub(crate) mod transform_support;
/// Concrete three-dimensional unit vector implementation.
pub mod unit_vector3d;

use crate::common::{
    FaceId, GeometricPrimitive, GeometryMeasure, HasCenter, HasCentroid, HasEdges, HasFaces,
    HasTetrahedra, HasVertices, IsUnitVector, PointId, TetrahedronId,
};
use crate::one_d::IsLine;
use crate::two_d::IsPolygon;
use crate::zero_d::IsPoint;
pub use line3d::Line3D;
pub use free_vector3d::FreeVector3D;
pub use surface_mesh3d::SurfaceMesh3D;
pub use plane3d::Plane3D;
pub use coordinate_vector3d::CoordinateVector3D;
pub use polygon_face3d::PolygonFace3D;
pub use tetrahedron3d::Tetrahedron3D;
pub use triangle3d::Triangle3D;
pub use unit_vector3d::UnitVector3D;
pub use volume_mesh3d::VolumeMesh3D;

/// A sphere primitive with radius-derived surface measures.
pub trait IsSphere<T: IsPoint>:
    GeometricPrimitive + HasCenter<Point = T> + crate::common::HasMeasure
{
    /// Returns the radius.
    fn radius(&self) -> GeometryMeasure;
    /// Returns the diameter.
    fn diameter(&self) -> GeometryMeasure;
    /// Returns the surface area.
    fn surface_area(&self) -> GeometryMeasure;
    /// Returns the enclosed volume.
    fn volume(&self) -> GeometryMeasure;
}

/// A tetrahedron primitive backed by four vertices and derived triangular faces.
pub trait IsTetrahedron<'a, T: IsPoint, N: IsUnitVector>:
    GeometricPrimitive + HasVertices<'a, Vertex = T> + HasEdges + HasCentroid<Point = T> + crate::common::HasMeasure
where
    <Self as HasEdges>::Edge: IsLine<'a, T>,
{
    /// Returns the point-table id of the first vertex.
    fn a_id(&self) -> PointId;
    /// Sets the point-table id of the first vertex.
    fn set_a_id(&mut self, point_id: PointId) -> Result<(), String>;
    /// Returns the point-table id of the second vertex.
    fn b_id(&self) -> PointId;
    /// Sets the point-table id of the second vertex.
    fn set_b_id(&mut self, point_id: PointId) -> Result<(), String>;
    /// Returns the point-table id of the third vertex.
    fn c_id(&self) -> PointId;
    /// Sets the point-table id of the third vertex.
    fn set_c_id(&mut self, point_id: PointId) -> Result<(), String>;
    /// Returns the point-table id of the fourth vertex.
    fn d_id(&self) -> PointId;
    /// Sets the point-table id of the fourth vertex.
    fn set_d_id(&mut self, point_id: PointId) -> Result<(), String>;

    /// Resolves and returns the first vertex from the point table.
    fn a(&self) -> Option<T> {
        self.get_vertex(&self.a_id())
    }

    /// Resolves and returns the second vertex from the point table.
    fn b(&self) -> Option<T> {
        self.get_vertex(&self.b_id())
    }

    /// Resolves and returns the third vertex from the point table.
    fn c(&self) -> Option<T> {
        self.get_vertex(&self.c_id())
    }

    /// Resolves and returns the fourth vertex from the point table.
    fn d(&self) -> Option<T> {
        self.get_vertex(&self.d_id())
    }

    /// Returns the four derived triangular faces of the tetrahedron.
    fn faces(&self) -> [Triangle3D; 4];
    /// Returns the number of triangular faces.
    fn face_count(&self) -> usize;
    /// Returns the total surface area of the tetrahedron.
    fn surface_area(&self) -> GeometryMeasure;
    /// Returns the signed volume of the tetrahedron.
    fn signed_volume(&self) -> GeometryMeasure;
    /// Returns the absolute tetrahedron volume.
    fn volume(&self) -> GeometryMeasure;
}

/// A surface mesh primitive with stored vertices and polygonal boundary faces.
///
/// This trait is explicitly for surface meshes rather than volumetric cell meshes. The current
/// surface-mesh model assumes vertices are the primary stored table. Faces and edges may be backed
/// by separate structures internally, but they are exposed through semantic mesh accessors rather
/// than through the older generic single-table pattern.
pub trait IsMesh<'a, T: IsPoint, N: IsUnitVector>:
    GeometricPrimitive
    + HasVertices<'a, Vertex = T>
    + HasFaces<'a, Point = T, Normal = N>
    + HasEdges
where
    <Self as HasEdges>::Edge: IsLine<'a, T>,
{
    /// Returns the ordered face-table ids that participate in this mesh.
    fn face_ids(&self) -> Box<dyn Iterator<Item = FaceId> + '_>;
    /// Replaces the face-table id at the given mesh face position.
    fn set_face_id(&mut self, index: usize, face_id: FaceId) -> Result<(), String>;
    /// Returns the number of faces in the mesh.
    fn face_count(&self) -> usize {
        self.face_ids().count()
    }
    /// Resolves and returns a face by mesh-local position.
    fn face(&self, index: usize) -> Option<<Self as HasFaces<'a>>::Face> {
        self.face_ids().nth(index).and_then(|face_id| self.get_face(&face_id))
    }
    /// Returns the total surface area of the mesh.
    fn surface_area(&self) -> GeometryMeasure;
    /// Returns the normal associated with a face, if present.
    fn face_normal(&self, index: usize) -> Option<N> {
        self.face(index).map(|face| face.normal())
    }
}

/// Named alias for the surface-mesh interpretation of [`IsMesh`].
///
/// This trait exists to make the intent of the current mesh abstraction obvious at the API level
/// while preserving compatibility for existing `IsMesh` implementors.
pub trait IsSurfaceMesh<'a, T: IsPoint, N: IsUnitVector>: IsMesh<'a, T, N>
where
    <Self as HasEdges>::Edge: IsLine<'a, T>,
{
}

impl<'a, T, N, M> IsSurfaceMesh<'a, T, N> for M
where
    T: IsPoint,
    N: IsUnitVector,
    M: IsMesh<'a, T, N>,
    <M as HasEdges>::Edge: IsLine<'a, T>,
{
}

/// A volumetric mesh primitive with stored vertices and tetrahedral cells.
pub trait IsVolumeMesh<'a, T: IsPoint, N: IsUnitVector>:
    GeometricPrimitive
    + HasVertices<'a, Vertex = T>
    + HasTetrahedra<'a, Point = T, Normal = N>
    + HasEdges
where
    <Self as HasEdges>::Edge: IsLine<'a, T>,
{
    /// Returns the ordered tetrahedron-table ids that participate in this mesh.
    fn tetrahedron_ids(&self) -> Box<dyn Iterator<Item = TetrahedronId> + '_>;
    /// Replaces the tetrahedron-table id at the given mesh cell position.
    fn set_tetrahedron_id(&mut self, index: usize, tetrahedron_id: TetrahedronId) -> Result<(), String>;
    /// Returns the number of tetrahedra in the mesh.
    fn tetrahedron_count(&self) -> usize {
        self.tetrahedron_ids().count()
    }
    /// Resolves and returns a tetrahedron by mesh-local position.
    fn tetrahedron(&self, index: usize) -> Option<<Self as HasTetrahedra<'a>>::Tetrahedron> {
        self.tetrahedron_ids()
            .nth(index)
            .and_then(|tetrahedron_id| self.get_tetrahedron(&tetrahedron_id))
    }
    /// Returns the total enclosed volume of the mesh.
    fn volume(&self) -> GeometryMeasure;
}
