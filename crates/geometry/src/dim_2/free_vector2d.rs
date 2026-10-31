//! Concrete two-dimensional free-vector type for the public geometry API.
//!
//! [`FreeVector2D`] represents displacement, scale factors, and other non-coordinate-vector
//! vectors. Its stored coordinates may be Cartesian or polar, and all math is performed in
//! Cartesian space before converting the result back to the vector's declared coordinate system.

use crate::common::{
    CanNormalize, CanScale, CanScaleNonUniform, CoordinatePrimitive, DotProduct,
    GeometricPrimitive, GeometricPrimitive2D, GeometryMeasure, HasDimension, HasNorm, Normalize,
    ScalarOperable, SelfAddition, SelfProductInner,
};
use crate::coordinate_systems::{CoordinateSystem2D, ToCartesian, ToPolar};
use crate::two_d::coordinate_conversions;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// Concrete 2D free-vector implementation whose stored coordinates may be Cartesian or polar.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
pub struct FreeVector2D {
    coords: [GeometryMeasure; 2],
    coordinate_system: CoordinateSystem2D,
}

impl Eq for FreeVector2D {}

impl Hash for FreeVector2D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in self.coords {
            value.to_bits().hash(state);
        }
        self.coordinate_system.hash(state);
    }
}

impl FreeVector2D {
    /// Creates a vector from Cartesian `x` and `y` components.
    pub fn new(x: GeometryMeasure, y: GeometryMeasure) -> Self {
        Self {
            coords: [x, y],
            coordinate_system: CoordinateSystem2D::Cartesian,
        }
    }

    /// Creates a vector from raw coordinates in the specified system.
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

    /// Returns the Cartesian `[x, y]` representation of this vector.
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

    /// Returns the Cartesian x-component.
    pub fn x(&self) -> GeometryMeasure {
        self.cartesian_components()[0]
    }

    /// Returns the Cartesian y-component.
    pub fn y(&self) -> GeometryMeasure {
        self.cartesian_components()[1]
    }
}

impl Display for FreeVector2D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FreeVector2D({:?}, {}, {})",
            self.coordinate_system, self.coords[0], self.coords[1]
        )
    }
}

impl GeometricPrimitive for FreeVector2D {}
impl GeometricPrimitive2D for FreeVector2D {}
impl CoordinatePrimitive for FreeVector2D {}
impl HasDimension for FreeVector2D {
    const DIM: usize = 2;
}

impl AsRef<GeometryMeasure> for FreeVector2D {
    fn as_ref(&self) -> &GeometryMeasure {
        &self.coords[0]
    }
}

impl AsMut<GeometryMeasure> for FreeVector2D {
    fn as_mut(&mut self) -> &mut GeometryMeasure {
        &mut self.coords[0]
    }
}

impl Index<usize> for FreeVector2D {
    type Output = GeometryMeasure;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl IndexMut<usize> for FreeVector2D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl Add<GeometryMeasure> for FreeVector2D {
    type Output = Self;

    fn add(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components([cartesian[0] + rhs, cartesian[1] + rhs], self.coordinate_system)
    }
}

impl Sub<GeometryMeasure> for FreeVector2D {
    type Output = Self;

    fn sub(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components([cartesian[0] - rhs, cartesian[1] - rhs], self.coordinate_system)
    }
}

impl Mul<GeometryMeasure> for FreeVector2D {
    type Output = Self;

    fn mul(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components([cartesian[0] * rhs, cartesian[1] * rhs], self.coordinate_system)
    }
}

impl Div<GeometryMeasure> for FreeVector2D {
    type Output = Self;

    fn div(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components([cartesian[0] / rhs, cartesian[1] / rhs], self.coordinate_system)
    }
}

impl Add for FreeVector2D {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        Self::from_cartesian_components([lhs[0] + rhs[0], lhs[1] + rhs[1]], self.coordinate_system)
    }
}

impl Sub for FreeVector2D {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        Self::from_cartesian_components([lhs[0] - rhs[0], lhs[1] - rhs[1]], self.coordinate_system)
    }
}

impl Mul<FreeVector2D> for FreeVector2D {
    type Output = GeometryMeasure;

    fn mul(self, rhs: FreeVector2D) -> Self::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        lhs[0] * rhs[0] + lhs[1] * rhs[1]
    }
}

impl ScalarOperable for FreeVector2D {}
impl SelfAddition for FreeVector2D {}
impl SelfProductInner for FreeVector2D {}

impl CanScale for FreeVector2D {
    fn scale(&mut self, factor: GeometryMeasure) {
        *self = *self * factor;
    }
}

impl CanScaleNonUniform for FreeVector2D {
    type ScaleVector = FreeVector2D;

    fn scale_non_uniform(&mut self, factors: &Self::ScaleVector) {
        let coords = self.cartesian_components();
        let factor_coords = factors.cartesian_components();
        *self = Self::from_cartesian_components(
            [coords[0] * factor_coords[0], coords[1] * factor_coords[1]],
            self.coordinate_system,
        );
    }
}

impl DotProduct for FreeVector2D {
    type Output = GeometryMeasure;

    fn dot(&self, rhs: &Self) -> <Self as DotProduct>::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        lhs[0] * rhs[0] + lhs[1] * rhs[1]
    }
}

impl HasNorm for FreeVector2D {
    fn norm(&self) -> GeometryMeasure {
        let coords = self.cartesian_components();
        (coords[0] * coords[0] + coords[1] * coords[1]).sqrt()
    }
}

impl Normalize for FreeVector2D {
    fn normalized(&self) -> Self {
        let norm = self.norm();
        if norm == 0.0 {
            Self::new(0.0, 0.0)
        } else {
            Self::from_cartesian_components(
                [self.x() / norm, self.y() / norm],
                self.coordinate_system,
            )
        }
    }
}

impl CanNormalize for FreeVector2D {
    fn normalize(&mut self) {
        *self = self.normalized();
    }
}

impl ToCartesian for FreeVector2D {
    type Cartesian = FreeVector2D;

    fn to_cartesian(&self) -> Self::Cartesian {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem2D::Cartesian)
    }
}

impl ToPolar for FreeVector2D {
    type Polar = FreeVector2D;

    fn to_polar(&self) -> Self::Polar {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem2D::Polar)
    }
}
