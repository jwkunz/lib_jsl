//! Concrete one-dimensional unit vector type.

use crate::common::{
    CanNormalize, CoordinatePrimitive, DotProduct, GeometricPrimitive, GeometryMeasure,
    HasDimension, HasNorm, IsUnitVector, Normalize,
};
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Index, IndexMut};

/// Concrete 1D unit-vector implementation with values constrained to `-1` or `1`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
pub struct UnitVector1D {
    coord: GeometryMeasure,
}

impl Eq for UnitVector1D {}

impl Hash for UnitVector1D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coord.to_bits().hash(state);
    }
}

impl UnitVector1D {
    /// Creates a unit vector from a scalar direction.
    pub fn new(x: GeometryMeasure) -> Self {
        if x < 0.0 {
            Self { coord: -1.0 }
        } else {
            Self { coord: 1.0 }
        }
    }
}

impl Display for UnitVector1D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "UnitVector1D({})", self.coord)
    }
}

impl GeometricPrimitive for UnitVector1D {}
impl CoordinatePrimitive for UnitVector1D {}
impl IsUnitVector for UnitVector1D {}
impl HasDimension for UnitVector1D {
    const DIM: usize = 1;
}

impl AsRef<GeometryMeasure> for UnitVector1D {
    fn as_ref(&self) -> &GeometryMeasure {
        &self.coord
    }
}

impl AsMut<GeometryMeasure> for UnitVector1D {
    fn as_mut(&mut self) -> &mut GeometryMeasure {
        &mut self.coord
    }
}

impl Index<usize> for UnitVector1D {
    type Output = GeometryMeasure;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.coord,
            _ => panic!("UnitVector1D index out of bounds: {}", index),
        }
    }
}

impl IndexMut<usize> for UnitVector1D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.coord,
            _ => panic!("UnitVector1D index out of bounds: {}", index),
        }
    }
}

impl DotProduct for UnitVector1D {
    type Output = GeometryMeasure;

    fn dot(&self, rhs: &Self) -> <Self as DotProduct>::Output {
        self.coord * rhs.coord
    }
}

impl HasNorm for UnitVector1D {
    fn norm(&self) -> GeometryMeasure {
        1.0
    }
}

impl Normalize for UnitVector1D {
    fn normalized(&self) -> Self {
        *self
    }
}

impl CanNormalize for UnitVector1D {
    fn normalize(&mut self) {
        *self = Self::new(self.coord);
    }
}
