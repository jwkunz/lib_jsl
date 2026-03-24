//! One-dimensional line- and path-oriented geometry traits.

use crate::geometry::common::{
    GeometricPrimitive, GeometryMeasure, HasVertices, IsUnitVector, ScalarOperable, SelfAddition,
    UsesTable,
};
use crate::geometry::zero_d::IsPoint;

/// A table-backed line primitive with two endpoints and derived line properties.
pub trait IsLine<'a, T: IsPoint>:
    GeometricPrimitive + ScalarOperable + SelfAddition + UsesTable<'a, Item = T>
{
    /// Returns the first endpoint.
    fn head(&self) -> T;
    /// Returns a mutable reference to the first endpoint.
    fn head_mut(&mut self) -> &mut T;
    /// Returns the second endpoint.
    fn tail(&self) -> T;
    /// Returns a mutable reference to the second endpoint.
    fn tail_mut(&mut self) -> &mut T;
    /// Returns the Euclidean length of the line.
    fn length(&self) -> GeometryMeasure;
    /// Returns the midpoint of the line.
    fn midpoint(&self) -> T;
    /// Returns the line direction as a unit vector.
    fn direction(&self) -> impl IsUnitVector;
}

/// A connected sequence of line segments backed by a point table.
pub trait IsPolyline<'a, T: IsPoint>:
    GeometricPrimitive + UsesTable<'a, Item = T> + HasVertices<'a, Item = T>
{
    /// Returns the number of segments in the polyline.
    fn segment_count(&self) -> usize;
    /// Returns the total path length.
    fn length(&self) -> GeometryMeasure;
}

/// A ray primitive with an origin and a unit direction.
pub trait IsRay<T: IsPoint>: GeometricPrimitive {
    /// Unit-vector direction type used by the ray.
    type Direction: IsUnitVector;

    /// Returns the ray origin.
    fn origin(&self) -> T;
    /// Returns a mutable reference to the ray origin.
    fn origin_mut(&mut self) -> &mut T;
    /// Returns the ray direction.
    fn direction(&self) -> Self::Direction;
    /// Returns a mutable reference to the ray direction.
    fn direction_mut(&mut self) -> &mut Self::Direction;
}

/// Marker trait for finite line segments.
pub trait IsSegment<'a, T: IsPoint>: IsLine<'a, T> {}
