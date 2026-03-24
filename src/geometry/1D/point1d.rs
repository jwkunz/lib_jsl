//! Concrete one-dimensional point type.

use crate::geometry::common::{
    CanScale, CanScaleNonUniform, CoordinatePrimitive, GeometricPrimitive, GeometryMeasure,
    HasDimension, IsPlane, ScalarOperable, SelfAddition, SelfProductInner,
};
use crate::geometry::one_d::UnitVector1D;
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanTranslate};
use crate::geometry::zero_d::IsPoint;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// Concrete 1D point implementation backed by a single coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
pub struct Point1D {
    coord: GeometryMeasure,
}

impl Eq for Point1D {}

impl Hash for Point1D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.coord.to_bits().hash(state);
    }
}

impl Point1D {
    /// Creates a point from a single coordinate.
    pub fn new(x: GeometryMeasure) -> Self {
        Self { coord: x }
    }

    /// Returns the point coordinate.
    pub fn x(&self) -> GeometryMeasure {
        self.coord
    }
}

impl Display for Point1D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Point1D({})", self.coord)
    }
}

impl GeometricPrimitive for Point1D {}
impl CoordinatePrimitive for Point1D {}
impl HasDimension for Point1D {
    const DIM: usize = 1;
}

impl AsRef<GeometryMeasure> for Point1D {
    fn as_ref(&self) -> &GeometryMeasure {
        &self.coord
    }
}

impl AsMut<GeometryMeasure> for Point1D {
    fn as_mut(&mut self) -> &mut GeometryMeasure {
        &mut self.coord
    }
}

impl Index<usize> for Point1D {
    type Output = GeometryMeasure;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.coord,
            _ => panic!("Point1D index out of bounds: {}", index),
        }
    }
}

impl IndexMut<usize> for Point1D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.coord,
            _ => panic!("Point1D index out of bounds: {}", index),
        }
    }
}

impl Add<GeometryMeasure> for Point1D {
    type Output = Self;

    fn add(self, rhs: GeometryMeasure) -> Self::Output {
        Self::new(self.coord + rhs)
    }
}

impl Sub<GeometryMeasure> for Point1D {
    type Output = Self;

    fn sub(self, rhs: GeometryMeasure) -> Self::Output {
        Self::new(self.coord - rhs)
    }
}

impl Mul<GeometryMeasure> for Point1D {
    type Output = Self;

    fn mul(self, rhs: GeometryMeasure) -> Self::Output {
        Self::new(self.coord * rhs)
    }
}

impl Div<GeometryMeasure> for Point1D {
    type Output = Self;

    fn div(self, rhs: GeometryMeasure) -> Self::Output {
        Self::new(self.coord / rhs)
    }
}

impl Add for Point1D {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.coord + rhs.coord)
    }
}

impl Sub for Point1D {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.coord - rhs.coord)
    }
}

impl Mul<Point1D> for Point1D {
    type Output = GeometryMeasure;

    fn mul(self, rhs: Point1D) -> Self::Output {
        self.coord * rhs.coord
    }
}

impl ScalarOperable for Point1D {}
impl SelfAddition for Point1D {}
impl SelfProductInner for Point1D {}

impl CanScale for Point1D {
    fn scale(&mut self, factor: GeometryMeasure) {
        self.coord *= factor;
    }
}

impl CanScaleNonUniform for Point1D {
    type ScaleVector = Point1D;

    fn scale_non_uniform(&mut self, factors: &Self::ScaleVector) {
        self.coord *= factors.coord;
    }
}

impl CanTranslate for Point1D {
    type Point = Point1D;

    fn translate<'a, L>(&mut self, translation_vector: &L)
    where
        L: crate::geometry::one_d::IsLine<'a, Self::Point>,
    {
        let (Some(head), Some(tail)) = (translation_vector.head(), translation_vector.tail()) else {
            return;
        };
        self.coord += tail.coord - head.coord;
    }
}

impl CanRotate for Point1D {
    type Point = Point1D;

    fn rotate<'a, L>(&mut self, _axis: &L, _angle_radians: GeometryMeasure)
    where
        L: crate::geometry::one_d::IsLine<'a, Self::Point>,
    {
    }
}

impl CanMirror for Point1D {
    type Point = Point1D;
    type Normal = UnitVector1D;

    fn mirror<P>(&mut self, mirror_plane: &P)
    where
        P: IsPlane<Point = Self::Point, Normal = Self::Normal>,
    {
        let origin = mirror_plane.point().coord;
        self.coord = 2.0 * origin - self.coord;
    }
}

impl IsPoint for Point1D {}
