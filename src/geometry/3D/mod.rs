//! Three-dimensional geometry traits.

/// Concrete three-dimensional line implementation.
pub mod line3d;
/// Concrete three-dimensional surface mesh implementation.
pub mod surface_mesh3d;
/// Concrete three-dimensional plane implementation.
pub mod plane3d;
/// Concrete three-dimensional point implementation.
pub mod point3d;
/// Concrete three-dimensional polygon face implementation.
pub mod polygon_face3d;
/// Concrete three-dimensional triangle implementation.
pub mod triangle3d;
/// Concrete three-dimensional tetrahedron implementation.
pub mod tetrahedron3d;
pub(crate) mod transform_support;
/// Concrete three-dimensional unit vector implementation.
pub mod unit_vector3d;

use crate::geometry::common::{
    FaceId, GeometricPrimitive, GeometryMeasure, HasCenter, HasEdges, HasFaces, HasVertices,
    IsUnitVector,
};
use crate::geometry::one_d::IsLine;
use crate::geometry::two_d::IsPolygon;
use crate::geometry::zero_d::IsPoint;
pub use line3d::Line3D;
pub use surface_mesh3d::SurfaceMesh3D;
pub use plane3d::Plane3D;
pub use point3d::Point3D;
pub use polygon_face3d::PolygonFace3D;
pub use tetrahedron3d::Tetrahedron3D;
pub use triangle3d::Triangle3D;
pub use unit_vector3d::UnitVector3D;

/// A plane primitive represented by a point and a unit normal.
pub trait IsPlane: GeometricPrimitive {
    /// Point type used to anchor the plane.
    type Point: IsPoint;
    /// Unit normal type used to orient the plane.
    type Normal: IsUnitVector;

    /// Returns a point on the plane.
    fn point(&self) -> Self::Point;
    /// Returns a mutable reference to a point on the plane.
    fn point_mut(&mut self) -> &mut Self::Point;
    /// Returns the plane normal.
    fn normal(&self) -> Self::Normal;
    /// Returns a mutable reference to the plane normal.
    fn normal_mut(&mut self) -> &mut Self::Normal;
}

/// A sphere primitive with radius-derived surface measures.
pub trait IsSphere<T: IsPoint>:
    GeometricPrimitive + HasCenter<Point = T> + crate::geometry::common::HasMeasure
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

/// A mesh primitive with stored vertices and derived edges plus polygonal faces.
///
/// The current mesh model assumes vertices are the primary stored table. Faces and edges may be
/// backed by separate structures internally, but they are exposed through semantic mesh accessors
/// rather than through [`UsesTable`](crate::geometry::common::UsesTable).
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
