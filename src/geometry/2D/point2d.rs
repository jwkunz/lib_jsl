//! Concrete two-dimensional point type for the public geometry API.

use crate::geometry::common::{
    CanScale, CanScaleNonUniform, CoordinatePrimitive, GeometricPrimitive, GeometricPrimitive2D,
    GeometryMeasure, HasDimension, ScalarOperable, SelfAddition, SelfProductInner,
};
use crate::geometry::one_d::IsLine;
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanTranslate};
use crate::geometry::two_d::transform_support::{reflect_point_across_plane_2d, rotate_point_around_anchor_2d};
use crate::geometry::two_d::{IsPlane, UnitVector2D};
use crate::geometry::zero_d::IsPoint;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// Concrete 2D point implementation backed by `[x, y]` coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
pub struct Point2D {
    coords: [GeometryMeasure; 2],
}

impl Eq for Point2D {}

impl Hash for Point2D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in self.coords {
            value.to_bits().hash(state);
        }
    }
}

impl Point2D {
    /// Creates a point from `x` and `y` coordinates.
    pub fn new(x: GeometryMeasure, y: GeometryMeasure) -> Self {
        Self { coords: [x, y] }
    }

    pub(crate) fn from_array(coords: [GeometryMeasure; 2]) -> Self {
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
}

impl Display for Point2D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Point2D({}, {})", self.x(), self.y())
    }
}

impl GeometricPrimitive for Point2D {}
impl GeometricPrimitive2D for Point2D {}
impl CoordinatePrimitive for Point2D {}
impl HasDimension for Point2D {
    const DIM: usize = 2;
}

impl AsRef<GeometryMeasure> for Point2D {
    fn as_ref(&self) -> &GeometryMeasure {
        &self.coords[0]
    }
}

impl AsMut<GeometryMeasure> for Point2D {
    fn as_mut(&mut self) -> &mut GeometryMeasure {
        &mut self.coords[0]
    }
}

impl Index<usize> for Point2D {
    type Output = GeometryMeasure;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl IndexMut<usize> for Point2D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl Add<GeometryMeasure> for Point2D {
    type Output = Self;

    fn add(self, rhs: GeometryMeasure) -> Self::Output {
        Self::from_array([self[0] + rhs, self[1] + rhs])
    }
}

impl Sub<GeometryMeasure> for Point2D {
    type Output = Self;

    fn sub(self, rhs: GeometryMeasure) -> Self::Output {
        Self::from_array([self[0] - rhs, self[1] - rhs])
    }
}

impl Mul<GeometryMeasure> for Point2D {
    type Output = Self;

    fn mul(self, rhs: GeometryMeasure) -> Self::Output {
        Self::from_array([self[0] * rhs, self[1] * rhs])
    }
}

impl Div<GeometryMeasure> for Point2D {
    type Output = Self;

    fn div(self, rhs: GeometryMeasure) -> Self::Output {
        Self::from_array([self[0] / rhs, self[1] / rhs])
    }
}

impl Add for Point2D {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_array([self[0] + rhs[0], self[1] + rhs[1]])
    }
}

impl Sub for Point2D {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_array([self[0] - rhs[0], self[1] - rhs[1]])
    }
}

impl Mul<Point2D> for Point2D {
    type Output = GeometryMeasure;

    fn mul(self, rhs: Point2D) -> Self::Output {
        self[0] * rhs[0] + self[1] * rhs[1]
    }
}

impl ScalarOperable for Point2D {}
impl SelfAddition for Point2D {}
impl SelfProductInner for Point2D {}

impl CanScale for Point2D {
    fn scale(&mut self, factor: GeometryMeasure) {
        *self = *self * factor;
    }
}

impl CanScaleNonUniform for Point2D {
    type ScaleVector = Point2D;

    fn scale_non_uniform(&mut self, factors: &Self::ScaleVector) {
        self[0] *= factors[0];
        self[1] *= factors[1];
    }
}

impl CanTranslate for Point2D {
    type Point = Point2D;

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

impl CanRotate for Point2D {
    type Point = Point2D;

    fn rotate<'a, L>(&mut self, axis: &L, angle_radians: GeometryMeasure)
    where
        L: IsLine<'a, Self::Point>,
    {
        let Some(origin) = axis.head() else {
            return;
        };
        *self = rotate_point_around_anchor_2d(*self, origin, angle_radians);
    }
}

impl CanMirror for Point2D {
    type Point = Point2D;
    type Normal = UnitVector2D;

    fn mirror<P>(&mut self, mirror_plane: &P)
    where
        P: IsPlane<Point = Self::Point, Normal = Self::Normal>,
    {
        *self = reflect_point_across_plane_2d(*self, mirror_plane.point(), mirror_plane.normal());
    }
}

impl IsPoint for Point2D {}
