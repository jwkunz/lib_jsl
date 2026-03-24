//! Two-dimensional shape traits and orientation helpers.

use crate::geometry::common::{
    GeometricPrimitive, GeometryMeasure, HasCentroid, HasCenter, HasEdges, HasMeasure,
    HasVertices,
};
use crate::geometry::one_d::IsLine;
use crate::geometry::three_d::IsPlane;
use crate::geometry::common::IsUnitVector;
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanShear, CanTranslate};
use crate::geometry::zero_d::IsPoint;

/// Relative winding/orientation classification for planar geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation2D {
    /// Points or edges are ordered clockwise.
    Clockwise,
    /// Points or edges are ordered counter-clockwise.
    CounterClockwise,
    /// No orientation can be determined because the data is collinear.
    Collinear,
}

/// Exposes the planar orientation of a primitive.
pub trait HasOrientation {
    /// Returns the primitive orientation.
    fn orientation(&self) -> Orientation2D;
}

/// A triangle primitive backed by a point table and treated as a polygon specialization.
pub trait IsTriangle<'a, T: IsPoint, N: IsUnitVector>: IsPolygon<'a, T, N> {
    /// Returns the first vertex.
    fn a(&self) -> T;
    /// Returns a mutable reference to the first vertex.
    fn a_mut(&mut self) -> &mut T;
    /// Returns the second vertex.
    fn b(&self) -> T;
    /// Returns a mutable reference to the second vertex.
    fn b_mut(&mut self) -> &mut T;
    /// Returns the third vertex.
    fn c(&self) -> T;
    /// Returns a mutable reference to the third vertex.
    fn c_mut(&mut self) -> &mut T;

    /// Returns the edge from `a` to `b`.
    fn edge_ab(&self) -> impl IsLine<'a, T>;
    /// Returns the edge from `b` to `c`.
    fn edge_bc(&self) -> impl IsLine<'a, T>;
    /// Returns the edge from `c` to `a`.
    fn edge_ca(&self) -> impl IsLine<'a, T>;
}

/// A polygon primitive defined by an ordered point table.
pub trait IsPolygon<'a, T: IsPoint, N: IsUnitVector>:
    GeometricPrimitive
    + HasVertices<'a, Item = T>
    + HasEdges
    + CanTranslate<Point = T>
    + CanRotate<Point = T>
    + CanShear<Point = T>
    + CanMirror<Point = T, Normal = N>
    + HasCentroid<Point = T>
    + HasMeasure
    + HasOrientation
{
    /// Returns the polygon normal.
    fn normal(&self) -> N;
    /// Returns the polygon perimeter.
    fn perimeter(&self) -> GeometryMeasure;
    /// Returns the polygon area.
    fn area(&self) -> GeometryMeasure;
    /// Returns the plane containing the polygon.
    fn plane(&self) -> impl IsPlane<Point = T, Normal = N>;
}

/// A rectangle specialization of `IsPolygon`.
pub trait IsRectangle<'a, T: IsPoint, N: IsUnitVector>: IsPolygon<'a, T, N> {
    /// Returns the rectangle width.
    fn width(&self) -> GeometryMeasure;
    /// Returns the rectangle height.
    fn height(&self) -> GeometryMeasure;
    /// Returns the rectangle diagonal length.
    fn diagonal(&self) -> GeometryMeasure;
}

/// A circle primitive with a center and radius-derived measures.
pub trait IsCircle<T: IsPoint>: GeometricPrimitive + HasCenter<Point = T> + HasMeasure {
    /// Returns the radius.
    fn radius(&self) -> GeometryMeasure;
    /// Returns the diameter.
    fn diameter(&self) -> GeometryMeasure;
    /// Returns the circumference.
    fn circumference(&self) -> GeometryMeasure;
    /// Returns the enclosed area.
    fn area(&self) -> GeometryMeasure;
}
