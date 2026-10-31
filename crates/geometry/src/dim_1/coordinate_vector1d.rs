//! Concrete one-dimensional coordinate-vector type.

use crate::common::{
    CanScale, CanScaleNonUniform, CoordinatePrimitive, GeometricPrimitive, GeometryMeasure,
    HasDimension, IsPlane, ScalarOperable, SelfAddition, SelfProductInner,
};
use crate::one_d::UnitVector1D;
use crate::transformation_traits::{CanMirror, CanTranslate};
use crate::zero_d::IsPoint;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// Concrete 1D coordinate-vector implementation backed by a single coordinate.
///
/// ```compile_fail
/// use lib_jsl::geometry::one_d::CoordinateVector1D;
/// use lib_jsl::geometry::zero_d::TransformablePoint;
///
/// fn requires_transformable<T: TransformablePoint>() {}
///
/// requires_transformable::<CoordinateVector1D>();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
pub struct CoordinateVector1D {
    coord: GeometryMeasure,
}

impl Eq for CoordinateVector1D {}

impl Hash for CoordinateVector1D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coord.to_bits().hash(state);
    }
}

impl CoordinateVector1D {
    /// Creates a coordinate vector from a single coordinate.
    pub fn new(x: GeometryMeasure) -> Self {
        Self { coord: x }
    }

    /// Returns the coordinate value.
    pub fn x(&self) -> GeometryMeasure {
        self.coord
    }
}

impl Display for CoordinateVector1D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "CoordinateVector1D({})", self.coord)
    }
}

impl GeometricPrimitive for CoordinateVector1D {}
impl CoordinatePrimitive for CoordinateVector1D {}
impl HasDimension for CoordinateVector1D {
    const DIM: usize = 1;
}

impl AsRef<GeometryMeasure> for CoordinateVector1D {
    fn as_ref(&self) -> &GeometryMeasure {
        &self.coord
    }
}

impl AsMut<GeometryMeasure> for CoordinateVector1D {
    fn as_mut(&mut self) -> &mut GeometryMeasure {
        &mut self.coord
    }
}

impl Index<usize> for CoordinateVector1D {
    type Output = GeometryMeasure;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.coord,
            _ => panic!("CoordinateVector1D index out of bounds: {}", index),
        }
    }
}

impl IndexMut<usize> for CoordinateVector1D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.coord,
            _ => panic!("CoordinateVector1D index out of bounds: {}", index),
        }
    }
}

impl Add<GeometryMeasure> for CoordinateVector1D {
    type Output = Self;

    fn add(self, rhs: GeometryMeasure) -> Self::Output {
        Self::new(self.coord + rhs)
    }
}

impl Sub<GeometryMeasure> for CoordinateVector1D {
    type Output = Self;

    fn sub(self, rhs: GeometryMeasure) -> Self::Output {
        Self::new(self.coord - rhs)
    }
}

impl Mul<GeometryMeasure> for CoordinateVector1D {
    type Output = Self;

    fn mul(self, rhs: GeometryMeasure) -> Self::Output {
        Self::new(self.coord * rhs)
    }
}

impl Div<GeometryMeasure> for CoordinateVector1D {
    type Output = Self;

    fn div(self, rhs: GeometryMeasure) -> Self::Output {
        Self::new(self.coord / rhs)
    }
}

impl Add for CoordinateVector1D {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.coord + rhs.coord)
    }
}

impl Sub for CoordinateVector1D {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.coord - rhs.coord)
    }
}

impl Mul<CoordinateVector1D> for CoordinateVector1D {
    type Output = GeometryMeasure;

    fn mul(self, rhs: CoordinateVector1D) -> Self::Output {
        self.coord * rhs.coord
    }
}

impl ScalarOperable for CoordinateVector1D {}
impl SelfAddition for CoordinateVector1D {}
impl SelfProductInner for CoordinateVector1D {}

impl CanScale for CoordinateVector1D {
    fn scale(&mut self, factor: GeometryMeasure) {
        self.coord *= factor;
    }
}

impl CanScaleNonUniform for CoordinateVector1D {
    type ScaleVector = CoordinateVector1D;

    fn scale_non_uniform(&mut self, factors: &Self::ScaleVector) {
        self.coord *= factors.coord;
    }
}

impl CanTranslate for CoordinateVector1D {
    type Point = CoordinateVector1D;

    fn translate<'a, L>(&mut self, translation_vector: &L)
    where
        L: crate::one_d::IsLine<'a, Self::Point>,
    {
        let (Some(head), Some(tail)) = (translation_vector.head(), translation_vector.tail()) else {
            return;
        };
        self.coord += tail.coord - head.coord;
    }
}

impl CanMirror for CoordinateVector1D {
    type Point = CoordinateVector1D;
    type Normal = UnitVector1D;

    fn mirror<P>(&mut self, mirror_plane: &P)
    where
        P: IsPlane<Point = Self::Point, Normal = Self::Normal>,
    {
        let origin = mirror_plane.point().coord;
        self.coord = 2.0 * origin - self.coord;
    }
}

impl IsPoint for CoordinateVector1D {}
