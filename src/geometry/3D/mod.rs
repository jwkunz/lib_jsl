//! Three-dimensional geometry traits.

use crate::geometry::common::{
    GeometricPrimitive, GeometryMeasure, HasCenter, HasVertices, IsUnitVector,
};
use crate::geometry::zero_d::IsPoint;

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

/// A mesh primitive backed by a vertex table and face accessors.
pub trait IsMesh<'a, T: IsPoint>: GeometricPrimitive + HasVertices<'a, Item = T> {
    /// Face type returned by the mesh.
    type Face;

    /// Returns the number of faces in the mesh.
    fn face_count(&self) -> usize;
    /// Returns a face by index.
    fn face(&self, index: usize) -> Option<Self::Face>;
}
