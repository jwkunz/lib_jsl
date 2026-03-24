//! Concrete three-dimensional plane type for the public geometry API.

use crate::geometry::common::{GeometricPrimitive, GeometricPrimitive3D, IsPlane};
use crate::geometry::three_d::{CoordinateVector3D, UnitVector3D};
use serde::Serialize;
use std::fmt::{self, Display, Formatter};

/// Concrete 3D plane implementation defined by a point and unit normal.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
pub struct Plane3D {
    point: CoordinateVector3D,
    normal: UnitVector3D,
}

impl Plane3D {
    /// Creates a plane from a point and a unit normal.
    ///
    /// The point anchors the plane in space and the normal controls orientation.
    pub fn new(point: CoordinateVector3D, normal: UnitVector3D) -> Self {
        Self { point, normal }
    }
}

impl Display for Plane3D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Plane3D(point={}, normal={})", self.point, self.normal)
    }
}

impl GeometricPrimitive for Plane3D {}
impl GeometricPrimitive3D for Plane3D {}

impl IsPlane for Plane3D {
    type Point = CoordinateVector3D;
    type Normal = UnitVector3D;

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
