//! Concrete two-dimensional unit vector type.

use crate::geometry::common::{
    CanNormalize, CoordinatePrimitive, DotProduct, GeometricPrimitive, GeometricPrimitive2D,
    GeometryMeasure, HasDimension, HasNorm, IsUnitVector, Normalize,
};
use crate::geometry::two_d::Point2D;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Index, IndexMut};

/// Concrete 2D unit-vector implementation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
pub struct UnitVector2D {
    coords: [GeometryMeasure; 2],
}

impl Eq for UnitVector2D {}

impl Hash for UnitVector2D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in self.coords {
            value.to_bits().hash(state);
        }
    }
}

impl UnitVector2D {
    /// Creates and normalizes a vector from the supplied coordinates.
    pub fn new(x: GeometryMeasure, y: GeometryMeasure) -> Self {
        Self::from_point(Point2D::new(x, y))
    }

    pub(crate) fn from_point(point: Point2D) -> Self {
        let norm = ((point[0] * point[0]) + (point[1] * point[1])).sqrt();
        if norm == 0.0 {
            Self { coords: [1.0, 0.0] }
        } else {
            Self {
                coords: [point[0] / norm, point[1] / norm],
            }
        }
    }
}

impl Display for UnitVector2D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "UnitVector2D({}, {})", self.coords[0], self.coords[1])
    }
}

impl GeometricPrimitive for UnitVector2D {}
impl GeometricPrimitive2D for UnitVector2D {}
impl CoordinatePrimitive for UnitVector2D {}
impl IsUnitVector for UnitVector2D {}
impl HasDimension for UnitVector2D {
    const DIM: usize = 2;
}

impl AsRef<GeometryMeasure> for UnitVector2D {
    fn as_ref(&self) -> &GeometryMeasure {
        &self.coords[0]
    }
}

impl AsMut<GeometryMeasure> for UnitVector2D {
    fn as_mut(&mut self) -> &mut GeometryMeasure {
        &mut self.coords[0]
    }
}

impl Index<usize> for UnitVector2D {
    type Output = GeometryMeasure;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl IndexMut<usize> for UnitVector2D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl DotProduct for UnitVector2D {
    type Output = GeometryMeasure;

    fn dot(&self, rhs: &Self) -> <Self as DotProduct>::Output {
        self.coords[0] * rhs.coords[0] + self.coords[1] * rhs.coords[1]
    }
}

impl HasNorm for UnitVector2D {
    fn norm(&self) -> GeometryMeasure {
        1.0
    }
}

impl Normalize for UnitVector2D {
    fn normalized(&self) -> Self {
        *self
    }
}

impl CanNormalize for UnitVector2D {
    fn normalize(&mut self) {
        *self = Self::from_point(Point2D::new(self.coords[0], self.coords[1]));
    }
}
