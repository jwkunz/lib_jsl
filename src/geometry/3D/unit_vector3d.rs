//! Concrete three-dimensional unit vector type for the public geometry API.
//!
//! [`UnitVector3D`] is used anywhere the concrete API needs a guaranteed direction, such as plane
//! normals or line directions returned from geometric queries.

use crate::geometry::common::{
    CanNormalize, CoordinatePrimitive, CrossProduct, DotProduct, GeometricPrimitive,
    GeometricPrimitive3D, GeometryMeasure, HasDimension, HasNorm, IsUnitVector, Normalize,
};
use crate::geometry::zero_d::Point3D;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Index, IndexMut};

/// Concrete 3D unit-vector implementation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
pub struct UnitVector3D {
    coords: [GeometryMeasure; 3],
}

impl Eq for UnitVector3D {}

impl Hash for UnitVector3D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in self.coords {
            value.to_bits().hash(state);
        }
    }
}

impl UnitVector3D {
    /// Creates and normalizes a vector from the supplied coordinates.
    ///
    /// A zero-length input falls back to the positive x-axis.
    pub fn new(x: GeometryMeasure, y: GeometryMeasure, z: GeometryMeasure) -> Self {
        Self::from_point(Point3D::new(x, y, z))
    }

    pub(crate) fn from_point(point: Point3D) -> Self {
        let norm = ((point[0] * point[0]) + (point[1] * point[1]) + (point[2] * point[2])).sqrt();
        if norm == 0.0 {
            Self {
                coords: [1.0, 0.0, 0.0],
            }
        } else {
            Self {
                coords: [point[0] / norm, point[1] / norm, point[2] / norm],
            }
        }
    }

    pub(crate) fn as_point(self) -> Point3D {
        Point3D::from_array(self.coords)
    }
}

impl Display for UnitVector3D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "UnitVector3D({}, {}, {})", self.coords[0], self.coords[1], self.coords[2])
    }
}

impl GeometricPrimitive for UnitVector3D {}
impl GeometricPrimitive3D for UnitVector3D {}
impl CoordinatePrimitive for UnitVector3D {}
impl IsUnitVector for UnitVector3D {}
impl HasDimension for UnitVector3D {
    const DIM: usize = 3;
}

impl AsRef<GeometryMeasure> for UnitVector3D {
    fn as_ref(&self) -> &GeometryMeasure {
        &self.coords[0]
    }
}

impl AsMut<GeometryMeasure> for UnitVector3D {
    fn as_mut(&mut self) -> &mut GeometryMeasure {
        &mut self.coords[0]
    }
}

impl Index<usize> for UnitVector3D {
    type Output = GeometryMeasure;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl IndexMut<usize> for UnitVector3D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl DotProduct for UnitVector3D {
    type Output = GeometryMeasure;

    fn dot(&self, rhs: &Self) -> <Self as DotProduct>::Output {
        self.coords[0] * rhs.coords[0] + self.coords[1] * rhs.coords[1] + self.coords[2] * rhs.coords[2]
    }
}

impl CrossProduct for UnitVector3D {
    type Output = Point3D;

    fn cross(&self, rhs: &Self) -> <Self as CrossProduct>::Output {
        Point3D::new(
            self.coords[1] * rhs.coords[2] - self.coords[2] * rhs.coords[1],
            self.coords[2] * rhs.coords[0] - self.coords[0] * rhs.coords[2],
            self.coords[0] * rhs.coords[1] - self.coords[1] * rhs.coords[0],
        )
    }
}

impl HasNorm for UnitVector3D {
    fn norm(&self) -> GeometryMeasure {
        1.0
    }
}

impl Normalize for UnitVector3D {
    fn normalized(&self) -> Self {
        *self
    }
}

impl CanNormalize for UnitVector3D {
    fn normalize(&mut self) {
        *self = Self::from_point(self.as_point());
    }
}
