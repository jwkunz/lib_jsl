//! Concrete one-dimensional mirror plane abstraction.

use crate::geometry::common::{GeometricPrimitive, IsPlane};
use crate::geometry::one_d::{Point1D, UnitVector1D};
use serde::Serialize;
use std::fmt::{self, Display, Formatter};

/// Concrete 1D mirror reference represented by a point and unit normal.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
pub struct Plane1D {
    point: Point1D,
    normal: UnitVector1D,
}

impl Plane1D {
    /// Creates a new 1D mirror reference.
    pub fn new(point: Point1D, normal: UnitVector1D) -> Self {
        Self { point, normal }
    }
}

impl Display for Plane1D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Plane1D(point={}, normal={})", self.point, self.normal)
    }
}

impl GeometricPrimitive for Plane1D {}

impl IsPlane for Plane1D {
    type Point = Point1D;
    type Normal = UnitVector1D;

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
