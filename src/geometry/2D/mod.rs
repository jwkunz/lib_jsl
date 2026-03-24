//! Two-dimensional shape traits and orientation helpers.

/// Concrete two-dimensional line implementation.
pub mod line2d;
/// Concrete two-dimensional mesh implementation.
pub mod mesh2d;
/// Concrete two-dimensional point implementation.
pub mod point2d;
/// Concrete two-dimensional plane implementation.
pub mod plane2d;
/// Concrete two-dimensional polygon face implementation.
pub mod polygon_face2d;
/// Internal helpers shared by 2D concrete implementations.
pub(crate) mod transform_support;
/// Concrete two-dimensional triangle implementation.
pub mod triangle2d;
/// Concrete two-dimensional unit vector implementation.
pub mod unit_vector2d;

use crate::geometry::common::{
    GeometricPrimitive, GeometryMeasure, HasCentroid, HasCenter, HasEdges, HasMeasure,
    HasVertices, PointId,
};
use crate::geometry::three_d::IsPlane;
use crate::geometry::common::IsUnitVector;
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanShear, CanTranslate};
use crate::geometry::zero_d::IsPoint;
pub use line2d::Line2D;
pub use mesh2d::Mesh2D;
pub use plane2d::Plane2D;
pub use point2d::Point2D;
pub use polygon_face2d::PolygonFace2D;
pub use triangle2d::Triangle2D;
pub use unit_vector2d::UnitVector2D;

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

    /// Returns the edge from `a` to `b`.
    fn edge_ab(&self) -> Option<<Self as HasEdges>::Edge> {
        self.edge(0)
    }
    /// Returns the edge from `b` to `c`.
    fn edge_bc(&self) -> Option<<Self as HasEdges>::Edge> {
        self.edge(1)
    }
    /// Returns the edge from `c` to `a`.
    fn edge_ca(&self) -> Option<<Self as HasEdges>::Edge> {
        self.edge(2)
    }
}

/// A polygon primitive defined by an ordered point table.
pub trait IsPolygon<'a, T: IsPoint, N: IsUnitVector>:
    GeometricPrimitive
    + HasVertices<'a, Vertex = T>
    + HasEdges
    + CanTranslate<Point = T>
    + CanRotate<Point = T>
    + CanShear<Point = T>
    + CanMirror<Point = T, Normal = N>
    + HasCentroid<Point = T>
    + HasMeasure
    + HasOrientation
{
    /// Returns the ordered point-table ids that define this polygon.
    fn vertex_ids(&self) -> Box<dyn Iterator<Item = PointId> + '_>;
    /// Replaces the point-table id at the given polygon vertex position.
    fn set_vertex_id(&mut self, index: usize, point_id: PointId) -> Result<(), String>;

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
