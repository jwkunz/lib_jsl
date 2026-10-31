//! Concrete three-dimensional coordinate-vector type for the public geometry API.
//!
//! [`CoordinateVector3D`] is the foundational coordinate-bearing concrete primitive used by the
//! current concrete graph model. Higher-dimensional objects in this crate ultimately resolve back
//! to coordinate-vector entries stored in a keyed point table.

use crate::common::{
    CanScale, CanScaleNonUniform, CoordinatePrimitive, GeometricPrimitive, GeometricPrimitive3D,
    GeometryMeasure, HasDimension, IsPlane, ScalarOperable, SelfAddition, SelfProductInner,
};
use crate::coordinate_systems::{
    CoordinateSystem3D, ToCartesian, ToCylindrical, ToSpherical,
};
use crate::one_d::IsLine;
use crate::three_d::coordinate_conversions;
use crate::three_d::transform_support::{reflect_point_across_plane, rotate_point_around_axis};
use crate::three_d::{FreeVector3D, UnitVector3D};
use crate::transformation_traits::{CanMirror, CanRotate, CanTranslate};
use crate::zero_d::IsPoint;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// Concrete 3D coordinate-vector implementation whose stored coordinates may be Cartesian,
/// spherical, or cylindrical.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
pub struct CoordinateVector3D {
    coords: [GeometryMeasure; 3],
    coordinate_system: CoordinateSystem3D,
}

impl Eq for CoordinateVector3D {}

impl Hash for CoordinateVector3D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in self.coords {
            value.to_bits().hash(state);
        }
        self.coordinate_system.hash(state);
    }
}

impl CoordinateVector3D {
    /// Creates a coordinate vector from Cartesian `x`, `y`, and `z` coordinates.
    pub fn new(x: GeometryMeasure, y: GeometryMeasure, z: GeometryMeasure) -> Self {
        Self {
            coords: [x, y, z],
            coordinate_system: CoordinateSystem3D::Cartesian,
        }
    }

    /// Creates a coordinate vector from raw coordinates in the specified system.
    pub fn new_in_system(
        first: GeometryMeasure,
        second: GeometryMeasure,
        third: GeometryMeasure,
        coordinate_system: CoordinateSystem3D,
    ) -> Self {
        Self {
            coords: [first, second, third],
            coordinate_system,
        }
    }

    pub(crate) fn from_array_in_system(
        coords: [GeometryMeasure; 3],
        coordinate_system: CoordinateSystem3D,
    ) -> Self {
        Self {
            coords,
            coordinate_system,
        }
    }

    pub(crate) fn from_cartesian_components(
        coords: [GeometryMeasure; 3],
        coordinate_system: CoordinateSystem3D,
    ) -> Self {
        Self::from_array_in_system(
            coordinate_conversions::from_cartesian(coords, coordinate_system),
            coordinate_system,
        )
    }

    /// Returns the raw stored coordinates in the currently declared coordinate system.
    pub fn raw_components(&self) -> [GeometryMeasure; 3] {
        self.coords
    }

    /// Returns the Cartesian `[x, y, z]` representation of this coordinate vector.
    pub fn cartesian_components(&self) -> [GeometryMeasure; 3] {
        coordinate_conversions::to_cartesian(self.coords, self.coordinate_system)
    }

    /// Returns the current coordinate system.
    pub fn coordinate_system(&self) -> CoordinateSystem3D {
        self.coordinate_system
    }

    /// Converts the stored coordinates to `coordinate_system` if needed.
    pub fn set_coordinate_system(&mut self, coordinate_system: CoordinateSystem3D) {
        if self.coordinate_system != coordinate_system {
            let cartesian = self.cartesian_components();
            self.coords = coordinate_conversions::from_cartesian(cartesian, coordinate_system);
            self.coordinate_system = coordinate_system;
        }
    }

    /// Returns a copy represented in the requested coordinate system.
    pub fn converted_to(&self, coordinate_system: CoordinateSystem3D) -> Self {
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

    /// Returns the Cartesian z-coordinate.
    pub fn z(&self) -> GeometryMeasure {
        self.cartesian_components()[2]
    }
}

impl Display for CoordinateVector3D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CoordinateVector3D({:?}, {}, {}, {})",
            self.coordinate_system, self.coords[0], self.coords[1], self.coords[2]
        )
    }
}

impl GeometricPrimitive for CoordinateVector3D {}
impl GeometricPrimitive3D for CoordinateVector3D {}
impl CoordinatePrimitive for CoordinateVector3D {}
impl HasDimension for CoordinateVector3D {
    const DIM: usize = 3;
}

impl AsRef<GeometryMeasure> for CoordinateVector3D {
    fn as_ref(&self) -> &GeometryMeasure {
        &self.coords[0]
    }
}

impl AsMut<GeometryMeasure> for CoordinateVector3D {
    fn as_mut(&mut self) -> &mut GeometryMeasure {
        &mut self.coords[0]
    }
}

impl Index<usize> for CoordinateVector3D {
    type Output = GeometryMeasure;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl IndexMut<usize> for CoordinateVector3D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl Add<GeometryMeasure> for CoordinateVector3D {
    type Output = Self;

    fn add(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components(
            [cartesian[0] + rhs, cartesian[1] + rhs, cartesian[2] + rhs],
            self.coordinate_system,
        )
    }
}

impl Sub<GeometryMeasure> for CoordinateVector3D {
    type Output = Self;

    fn sub(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components(
            [cartesian[0] - rhs, cartesian[1] - rhs, cartesian[2] - rhs],
            self.coordinate_system,
        )
    }
}

impl Mul<GeometryMeasure> for CoordinateVector3D {
    type Output = Self;

    fn mul(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components(
            [cartesian[0] * rhs, cartesian[1] * rhs, cartesian[2] * rhs],
            self.coordinate_system,
        )
    }
}

impl Div<GeometryMeasure> for CoordinateVector3D {
    type Output = Self;

    fn div(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components(
            [cartesian[0] / rhs, cartesian[1] / rhs, cartesian[2] / rhs],
            self.coordinate_system,
        )
    }
}

impl Add for CoordinateVector3D {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        Self::from_cartesian_components(
            [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]],
            self.coordinate_system,
        )
    }
}

impl Add<FreeVector3D> for CoordinateVector3D {
    type Output = Self;

    fn add(self, rhs: FreeVector3D) -> Self::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        Self::from_cartesian_components(
            [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]],
            self.coordinate_system,
        )
    }
}

impl Sub for CoordinateVector3D {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        Self::from_cartesian_components(
            [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]],
            self.coordinate_system,
        )
    }
}

impl Sub<FreeVector3D> for CoordinateVector3D {
    type Output = Self;

    fn sub(self, rhs: FreeVector3D) -> Self::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        Self::from_cartesian_components(
            [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]],
            self.coordinate_system,
        )
    }
}

impl Mul<CoordinateVector3D> for CoordinateVector3D {
    type Output = GeometryMeasure;

    fn mul(self, rhs: CoordinateVector3D) -> Self::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
    }
}

impl ScalarOperable for CoordinateVector3D {}
impl SelfAddition for CoordinateVector3D {}
impl SelfProductInner for CoordinateVector3D {}

impl CanScale for CoordinateVector3D {
    fn scale(&mut self, factor: GeometryMeasure) {
        *self = *self * factor;
    }
}

impl CanScaleNonUniform for CoordinateVector3D {
    type ScaleVector = FreeVector3D;

    fn scale_non_uniform(&mut self, factors: &Self::ScaleVector) {
        let coords = self.cartesian_components();
        let factor_coords = factors.cartesian_components();
        *self = Self::from_cartesian_components(
            [
                coords[0] * factor_coords[0],
                coords[1] * factor_coords[1],
                coords[2] * factor_coords[2],
            ],
            self.coordinate_system,
        );
    }
}

impl CanTranslate for CoordinateVector3D {
    type Point = CoordinateVector3D;

    fn translate<'a, L>(&mut self, translation_vector: &L)
    where
        L: IsLine<'a, Self::Point>,
    {
        let (Some(head), Some(tail)) = (translation_vector.head(), translation_vector.tail()) else {
            return;
        };
        let delta = FreeVector3D::from_cartesian_components(
            [tail.x() - head.x(), tail.y() - head.y(), tail.z() - head.z()],
            self.coordinate_system,
        );
        *self = *self + delta;
    }
}

impl CanRotate for CoordinateVector3D {
    type Point = CoordinateVector3D;

    fn rotate<'a, L>(&mut self, axis: &L, angle_radians: GeometryMeasure)
    where
        L: IsLine<'a, Self::Point>,
    {
        let Some(origin) = axis.head() else {
            return;
        };
        let direction = axis.direction();
        let rotated = rotate_point_around_axis(*self, origin, &direction, angle_radians);
        *self = Self::from_cartesian_components(rotated.cartesian_components(), self.coordinate_system);
    }
}

impl CanMirror for CoordinateVector3D {
    type Point = CoordinateVector3D;
    type Normal = UnitVector3D;

    fn mirror<P>(&mut self, mirror_plane: &P)
    where
        P: IsPlane<Point = Self::Point, Normal = Self::Normal>,
    {
        let reflected = reflect_point_across_plane(*self, mirror_plane.point(), mirror_plane.normal());
        *self = Self::from_cartesian_components(reflected.cartesian_components(), self.coordinate_system);
    }
}

impl ToCartesian for CoordinateVector3D {
    type Cartesian = CoordinateVector3D;

    fn to_cartesian(&self) -> Self::Cartesian {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem3D::Cartesian)
    }
}

impl ToSpherical for CoordinateVector3D {
    type Spherical = CoordinateVector3D;

    fn to_spherical(&self) -> Self::Spherical {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem3D::Spherical)
    }
}

impl ToCylindrical for CoordinateVector3D {
    type Cylindrical = CoordinateVector3D;

    fn to_cylindrical(&self) -> Self::Cylindrical {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem3D::Cylindrical)
    }
}

impl IsPoint for CoordinateVector3D {}
