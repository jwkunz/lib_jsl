//! One-dimensional line- and path-oriented geometry traits.

use crate::geometry::common::{
    GeometricPrimitive, GeometryMeasure, HasVertices, IsUnitVector, PointId,
};
use crate::geometry::zero_d::IsPoint;

/// A line primitive that resolves its endpoints through point identifiers.
pub trait IsLine<'a, T: IsPoint>:
    GeometricPrimitive + HasVertices<'a, Vertex = T>
{
    /// Returns the point-table id of the first endpoint.
    fn head_id(&self) -> PointId;
    /// Sets the point-table id of the first endpoint.
    fn set_head_id(&mut self, point_id: PointId) -> Result<(), String>;
    /// Returns the point-table id of the second endpoint.
    fn tail_id(&self) -> PointId;
    /// Sets the point-table id of the second endpoint.
    fn set_tail_id(&mut self, point_id: PointId) -> Result<(), String>;

    /// Resolves and returns the first endpoint from the point table.
    fn head(&self) -> Option<T> {
        self.get_vertex(&self.head_id())
    }

    /// Resolves and returns the second endpoint from the point table.
    fn tail(&self) -> Option<T> {
        self.get_vertex(&self.tail_id())
    }

    /// Returns the Euclidean length of the line.
    fn length(&self) -> GeometryMeasure;
    /// Returns the midpoint of the line, if it can be derived from the point table.
    fn midpoint(&self) -> Option<T>;
    /// Returns the line direction as a unit vector.
    fn direction(&self) -> impl IsUnitVector;
}

/// A connected sequence of line segments backed by a point table.
pub trait IsPolyline<'a, T: IsPoint>:
    GeometricPrimitive + HasVertices<'a, Vertex = T>
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
