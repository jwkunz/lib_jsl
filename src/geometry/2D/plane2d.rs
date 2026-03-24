//! Concrete planar mirror/reference line abstraction for two-dimensional geometry.

use crate::geometry::common::{GeometricPrimitive, GeometricPrimitive2D, IsPlane};
use crate::geometry::two_d::{CoordinateVector2D, UnitVector2D};
use serde::Serialize;
use std::fmt::{self, Display, Formatter};

/// Concrete 2D plane-like primitive represented by a point and an in-plane unit normal.
///
/// In the current trait system, [`IsPlane`](crate::geometry::common::IsPlane) serves as the
/// shared mirror/reference abstraction. For 2D geometry this type corresponds to a line expressed
/// in point-normal form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
pub struct Plane2D {
    point: CoordinateVector2D,
    normal: UnitVector2D,
}

impl Plane2D {
    /// Creates a new point-normal mirror/reference line.
    pub fn new(point: CoordinateVector2D, normal: UnitVector2D) -> Self {
        Self { point, normal }
    }
}

impl Display for Plane2D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Plane2D(point={}, normal={})", self.point, self.normal)
    }
}

impl GeometricPrimitive for Plane2D {}
impl GeometricPrimitive2D for Plane2D {}

impl IsPlane for Plane2D {
    type Point = CoordinateVector2D;
    type Normal = UnitVector2D;

    fn point(&self) -> Self::Point {
        self.point
    }

    fn point_mut(&mut self) -> &mut Self::Point {
        &mut self.point
    }

    fn normal(&self) -> Self::Normal {
        self.normal
    }

    fn normal_mut(&mut self) -> &mut Self::Normal {
        &mut self.normal
    }
}
