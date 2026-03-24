//! Concrete three-dimensional point type.

use crate::geometry::common::{
    CanScale, CanScaleNonUniform, CoordinatePrimitive, GeometricPrimitive, GeometricPrimitive3D,
    GeometryMeasure, HasDimension, ScalarOperable, SelfAddition, SelfProductInner,
};
use crate::geometry::one_d::IsLine;
use crate::geometry::three_d::transform_support::{reflect_point_across_plane, rotate_point_around_axis};
use crate::geometry::three_d::{IsPlane, UnitVector3D};
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanTranslate};
use crate::geometry::zero_d::IsPoint;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// Concrete 3D point implementation backed by `[x, y, z]` coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
pub struct Point3D {
    coords: [GeometryMeasure; 3],
}

impl Eq for Point3D {}

impl Hash for Point3D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in self.coords {
            value.to_bits().hash(state);
        }
    }
}

impl Point3D {
    /// Creates a point from `x`, `y`, and `z` coordinates.
    pub fn new(x: GeometryMeasure, y: GeometryMeasure, z: GeometryMeasure) -> Self {
        Self { coords: [x, y, z] }
    }

    pub(crate) fn from_array(coords: [GeometryMeasure; 3]) -> Self {
        Self { coords }
    }

    /// Returns the x-coordinate.
    pub fn x(&self) -> GeometryMeasure {
        self.coords[0]
    }

    /// Returns the y-coordinate.
    pub fn y(&self) -> GeometryMeasure {
        self.coords[1]
    }

    /// Returns the z-coordinate.
    pub fn z(&self) -> GeometryMeasure {
        self.coords[2]
    }
}

impl Display for Point3D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Point3D({}, {}, {})", self.x(), self.y(), self.z())
    }
}

impl GeometricPrimitive for Point3D {}
impl GeometricPrimitive3D for Point3D {}
impl CoordinatePrimitive for Point3D {}
impl HasDimension for Point3D {
    const DIM: usize = 3;
}

impl AsRef<GeometryMeasure> for Point3D {
    fn as_ref(&self) -> &GeometryMeasure {
        &self.coords[0]
    }
}

impl AsMut<GeometryMeasure> for Point3D {
    fn as_mut(&mut self) -> &mut GeometryMeasure {
        &mut self.coords[0]
    }
}

impl Index<usize> for Point3D {
    type Output = GeometryMeasure;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl IndexMut<usize> for Point3D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl Add<GeometryMeasure> for Point3D {
    type Output = Self;

    fn add(self, rhs: GeometryMeasure) -> Self::Output {
        Self::from_array([self[0] + rhs, self[1] + rhs, self[2] + rhs])
    }
}

impl Sub<GeometryMeasure> for Point3D {
    type Output = Self;

    fn sub(self, rhs: GeometryMeasure) -> Self::Output {
        Self::from_array([self[0] - rhs, self[1] - rhs, self[2] - rhs])
    }
}

impl Mul<GeometryMeasure> for Point3D {
    type Output = Self;

    fn mul(self, rhs: GeometryMeasure) -> Self::Output {
        Self::from_array([self[0] * rhs, self[1] * rhs, self[2] * rhs])
    }
}

impl Div<GeometryMeasure> for Point3D {
    type Output = Self;

    fn div(self, rhs: GeometryMeasure) -> Self::Output {
        Self::from_array([self[0] / rhs, self[1] / rhs, self[2] / rhs])
    }
}

impl Add for Point3D {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_array([self[0] + rhs[0], self[1] + rhs[1], self[2] + rhs[2]])
    }
}

impl Sub for Point3D {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_array([self[0] - rhs[0], self[1] - rhs[1], self[2] - rhs[2]])
    }
}

impl Mul<Point3D> for Point3D {
    type Output = GeometryMeasure;

    fn mul(self, rhs: Point3D) -> Self::Output {
        self[0] * rhs[0] + self[1] * rhs[1] + self[2] * rhs[2]
    }
}

impl ScalarOperable for Point3D {}
impl SelfAddition for Point3D {}
impl SelfProductInner for Point3D {}

impl CanScale for Point3D {
    fn scale(&mut self, factor: GeometryMeasure) {
        *self = *self * factor;
    }
}

impl CanScaleNonUniform for Point3D {
    type ScaleVector = Point3D;

    fn scale_non_uniform(&mut self, factors: &Self::ScaleVector) {
        self[0] *= factors[0];
        self[1] *= factors[1];
        self[2] *= factors[2];
    }
}

impl CanTranslate for Point3D {
    type Point = Point3D;

    fn translate<'a, L>(&mut self, translation_vector: &L)
    where
        L: IsLine<'a, Self::Point>,
    {
        let (Some(head), Some(tail)) = (translation_vector.head(), translation_vector.tail()) else {
            return;
        };
        *self = *self + (tail - head);
    }
}

impl CanRotate for Point3D {
    type Point = Point3D;

    fn rotate<'a, L>(&mut self, axis: &L, angle_radians: GeometryMeasure)
    where
        L: IsLine<'a, Self::Point>,
    {
        let Some(origin) = axis.head() else {
            return;
        };
        let direction = axis.direction();
        *self = rotate_point_around_axis(*self, origin, &direction, angle_radians);
    }
}

impl CanMirror for Point3D {
    type Point = Point3D;
    type Normal = UnitVector3D;

    fn mirror<P>(&mut self, mirror_plane: &P)
    where
        P: IsPlane<Point = Self::Point, Normal = Self::Normal>,
    {
        *self = reflect_point_across_plane(*self, mirror_plane.point(), mirror_plane.normal());
    }
}

impl IsPoint for Point3D {}
