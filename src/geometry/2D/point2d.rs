//! Concrete two-dimensional point type for the public geometry API.

use crate::geometry::common::{
    CanScale, CanScaleNonUniform, CoordinatePrimitive, GeometricPrimitive, GeometricPrimitive2D,
    GeometryMeasure, HasDimension, ScalarOperable, SelfAddition, SelfProductInner,
};
use crate::geometry::coordinate_systems::{CoordinateSystem2D, ToCartesian, ToPolar};
use crate::geometry::one_d::IsLine;
use crate::geometry::transformation_traits::{CanMirror, CanRotate, CanTranslate};
use crate::geometry::two_d::coordinate_conversions;
use crate::geometry::two_d::transform_support::{reflect_point_across_plane_2d, rotate_point_around_anchor_2d};
use crate::geometry::two_d::{IsPlane, UnitVector2D};
use crate::geometry::zero_d::IsPoint;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// Concrete 2D point implementation whose stored coordinates may be Cartesian or polar.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
pub struct Point2D {
    coords: [GeometryMeasure; 2],
    coordinate_system: CoordinateSystem2D,
}

impl Eq for Point2D {}

impl Hash for Point2D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in self.coords {
            value.to_bits().hash(state);
        }
        self.coordinate_system.hash(state);
    }
}

impl Point2D {
    /// Creates a point from Cartesian `x` and `y` coordinates.
    pub fn new(x: GeometryMeasure, y: GeometryMeasure) -> Self {
        Self {
            coords: [x, y],
            coordinate_system: CoordinateSystem2D::Cartesian,
        }
    }

    /// Creates a point from raw coordinates in the specified system.
    pub fn new_in_system(
        first: GeometryMeasure,
        second: GeometryMeasure,
        coordinate_system: CoordinateSystem2D,
    ) -> Self {
        Self {
            coords: [first, second],
            coordinate_system,
        }
    }

    pub(crate) fn from_array_in_system(
        coords: [GeometryMeasure; 2],
        coordinate_system: CoordinateSystem2D,
    ) -> Self {
        Self {
            coords,
            coordinate_system,
        }
    }

    pub(crate) fn from_cartesian_components(
        coords: [GeometryMeasure; 2],
        coordinate_system: CoordinateSystem2D,
    ) -> Self {
        Self::from_array_in_system(
            coordinate_conversions::from_cartesian(coords, coordinate_system),
            coordinate_system,
        )
    }

    /// Returns the raw stored coordinates in the currently declared coordinate system.
    pub fn raw_components(&self) -> [GeometryMeasure; 2] {
        self.coords
    }

    /// Returns the Cartesian `[x, y]` representation of this point.
    pub fn cartesian_components(&self) -> [GeometryMeasure; 2] {
        coordinate_conversions::to_cartesian(self.coords, self.coordinate_system)
    }

    /// Returns the current coordinate system.
    pub fn coordinate_system(&self) -> CoordinateSystem2D {
        self.coordinate_system
    }

    /// Converts the stored coordinates to `coordinate_system` if needed.
    pub fn set_coordinate_system(&mut self, coordinate_system: CoordinateSystem2D) {
        if self.coordinate_system != coordinate_system {
            let cartesian = self.cartesian_components();
            self.coords = coordinate_conversions::from_cartesian(cartesian, coordinate_system);
            self.coordinate_system = coordinate_system;
        }
    }

    /// Returns a copy represented in the requested coordinate system.
    pub fn converted_to(&self, coordinate_system: CoordinateSystem2D) -> Self {
        let mut converted = *self;
        converted.set_coordinate_system(coordinate_system);
        converted
    }

    /// Returns the Cartesian x-coordinate.
    pub fn x(&self) -> GeometryMeasure {
        self.cartesian_components()[0]
    }

    /// Returns the Cartesian y-coordinate.
    pub fn y(&self) -> GeometryMeasure {
        self.cartesian_components()[1]
    }
}

impl Display for Point2D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Point2D({:?}, {}, {})",
            self.coordinate_system, self.coords[0], self.coords[1]
        )
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
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components([cartesian[0] + rhs, cartesian[1] + rhs], self.coordinate_system)
    }
}

impl Sub<GeometryMeasure> for Point2D {
    type Output = Self;

    fn sub(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components([cartesian[0] - rhs, cartesian[1] - rhs], self.coordinate_system)
    }
}

impl Mul<GeometryMeasure> for Point2D {
    type Output = Self;

    fn mul(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components([cartesian[0] * rhs, cartesian[1] * rhs], self.coordinate_system)
    }
}

impl Div<GeometryMeasure> for Point2D {
    type Output = Self;

    fn div(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components([cartesian[0] / rhs, cartesian[1] / rhs], self.coordinate_system)
    }
}

impl Add for Point2D {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        Self::from_cartesian_components([lhs[0] + rhs[0], lhs[1] + rhs[1]], self.coordinate_system)
    }
}

impl Sub for Point2D {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        Self::from_cartesian_components([lhs[0] - rhs[0], lhs[1] - rhs[1]], self.coordinate_system)
    }
}

impl Mul<Point2D> for Point2D {
    type Output = GeometryMeasure;

    fn mul(self, rhs: Point2D) -> Self::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        lhs[0] * rhs[0] + lhs[1] * rhs[1]
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
        let coords = self.cartesian_components();
        let factor_coords = factors.cartesian_components();
        *self = Self::from_cartesian_components(
            [coords[0] * factor_coords[0], coords[1] * factor_coords[1]],
            self.coordinate_system,
        );
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
        let rotated = rotate_point_around_anchor_2d(*self, origin, angle_radians);
        *self = Self::from_cartesian_components(rotated.cartesian_components(), self.coordinate_system);
    }
}

impl CanMirror for Point2D {
    type Point = Point2D;
    type Normal = UnitVector2D;

    fn mirror<P>(&mut self, mirror_plane: &P)
    where
        P: IsPlane<Point = Self::Point, Normal = Self::Normal>,
    {
        let reflected = reflect_point_across_plane_2d(*self, mirror_plane.point(), mirror_plane.normal());
        *self = Self::from_cartesian_components(reflected.cartesian_components(), self.coordinate_system);
    }
}

impl ToCartesian for Point2D {
    type Cartesian = Point2D;

    fn to_cartesian(&self) -> Self::Cartesian {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem2D::Cartesian)
    }
}

impl ToPolar for Point2D {
    type Polar = Point2D;

    fn to_polar(&self) -> Self::Polar {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem2D::Polar)
    }
}

impl IsPoint for Point2D {}
