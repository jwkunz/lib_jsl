//! Concrete three-dimensional free-vector type for the public geometry API.
//!
//! [`FreeVector3D`] represents displacement, scale-factor, and direction-with-magnitude data.
//! Its stored coordinates may be Cartesian, spherical, or cylindrical, but arithmetic is carried
//! out in Cartesian space and then converted back to the vector's declared coordinate system.

use crate::common::{
    CanNormalize, CanScale, CanScaleNonUniform, CoordinatePrimitive, CrossProduct, DotProduct,
    GeometricPrimitive, GeometricPrimitive3D, GeometryMeasure, HasDimension, HasNorm, Normalize,
    ScalarOperable, SelfAddition, SelfProductInner,
};
use crate::coordinate_systems::{
    CoordinateSystem3D, ToCartesian, ToCylindrical, ToSpherical,
};
use crate::three_d::coordinate_conversions;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// Concrete 3D geometry-vector implementation whose stored coordinates may be Cartesian,
/// spherical, or cylindrical.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
pub struct FreeVector3D {
    coords: [GeometryMeasure; 3],
    coordinate_system: CoordinateSystem3D,
}

impl Eq for FreeVector3D {}

impl Hash for FreeVector3D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in self.coords {
            value.to_bits().hash(state);
        }
        self.coordinate_system.hash(state);
    }
}

impl FreeVector3D {
    /// Creates a vector from Cartesian `x`, `y`, and `z` components.
    pub fn new(x: GeometryMeasure, y: GeometryMeasure, z: GeometryMeasure) -> Self {
        Self {
            coords: [x, y, z],
            coordinate_system: CoordinateSystem3D::Cartesian,
        }
    }

    /// Creates a vector from raw coordinates in the specified system.
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

    /// Returns the Cartesian `[x, y, z]` representation of this vector.
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

    /// Returns the Cartesian x-component.
    pub fn x(&self) -> GeometryMeasure {
        self.cartesian_components()[0]
    }

    /// Returns the Cartesian y-component.
    pub fn y(&self) -> GeometryMeasure {
        self.cartesian_components()[1]
    }

    /// Returns the Cartesian z-component.
    pub fn z(&self) -> GeometryMeasure {
        self.cartesian_components()[2]
    }
}

impl Display for FreeVector3D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FreeVector3D({:?}, {}, {}, {})",
            self.coordinate_system, self.coords[0], self.coords[1], self.coords[2]
        )
    }
}

impl GeometricPrimitive for FreeVector3D {}
impl GeometricPrimitive3D for FreeVector3D {}
impl CoordinatePrimitive for FreeVector3D {}
impl HasDimension for FreeVector3D {
    const DIM: usize = 3;
}

impl AsRef<GeometryMeasure> for FreeVector3D {
    fn as_ref(&self) -> &GeometryMeasure {
        &self.coords[0]
    }
}

impl AsMut<GeometryMeasure> for FreeVector3D {
    fn as_mut(&mut self) -> &mut GeometryMeasure {
        &mut self.coords[0]
    }
}

impl Index<usize> for FreeVector3D {
    type Output = GeometryMeasure;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl IndexMut<usize> for FreeVector3D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl Add<GeometryMeasure> for FreeVector3D {
    type Output = Self;

    fn add(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components(
            [cartesian[0] + rhs, cartesian[1] + rhs, cartesian[2] + rhs],
            self.coordinate_system,
        )
    }
}

impl Sub<GeometryMeasure> for FreeVector3D {
    type Output = Self;

    fn sub(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components(
            [cartesian[0] - rhs, cartesian[1] - rhs, cartesian[2] - rhs],
            self.coordinate_system,
        )
    }
}

impl Mul<GeometryMeasure> for FreeVector3D {
    type Output = Self;

    fn mul(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components(
            [cartesian[0] * rhs, cartesian[1] * rhs, cartesian[2] * rhs],
            self.coordinate_system,
        )
    }
}

impl Div<GeometryMeasure> for FreeVector3D {
    type Output = Self;

    fn div(self, rhs: GeometryMeasure) -> Self::Output {
        let cartesian = self.cartesian_components();
        Self::from_cartesian_components(
            [cartesian[0] / rhs, cartesian[1] / rhs, cartesian[2] / rhs],
            self.coordinate_system,
        )
    }
}

impl Add for FreeVector3D {
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

impl Sub for FreeVector3D {
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

impl Mul<FreeVector3D> for FreeVector3D {
    type Output = GeometryMeasure;

    fn mul(self, rhs: FreeVector3D) -> Self::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
    }
}

impl ScalarOperable for FreeVector3D {}
impl SelfAddition for FreeVector3D {}
impl SelfProductInner for FreeVector3D {}

impl CanScale for FreeVector3D {
    fn scale(&mut self, factor: GeometryMeasure) {
        *self = *self * factor;
    }
}

impl CanScaleNonUniform for FreeVector3D {
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

impl DotProduct for FreeVector3D {
    type Output = GeometryMeasure;

    fn dot(&self, rhs: &Self) -> <Self as DotProduct>::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
    }
}

impl CrossProduct for FreeVector3D {
    type Output = FreeVector3D;

    fn cross(&self, rhs: &Self) -> <Self as CrossProduct>::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        Self::from_cartesian_components(
            [
                lhs[1] * rhs[2] - lhs[2] * rhs[1],
                lhs[2] * rhs[0] - lhs[0] * rhs[2],
                lhs[0] * rhs[1] - lhs[1] * rhs[0],
            ],
            self.coordinate_system,
        )
    }
}

impl HasNorm for FreeVector3D {
    fn norm(&self) -> GeometryMeasure {
        let coords = self.cartesian_components();
        (coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2]).sqrt()
    }
}

impl Normalize for FreeVector3D {
    fn normalized(&self) -> Self {
        let norm = self.norm();
        if norm == 0.0 {
            Self::new(0.0, 0.0, 0.0)
        } else {
            Self::from_cartesian_components(
                [self.x() / norm, self.y() / norm, self.z() / norm],
                self.coordinate_system,
            )
        }
    }
}

impl CanNormalize for FreeVector3D {
    fn normalize(&mut self) {
        *self = self.normalized();
    }
}

impl ToCartesian for FreeVector3D {
    type Cartesian = FreeVector3D;

    fn to_cartesian(&self) -> Self::Cartesian {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem3D::Cartesian)
    }
}

impl ToSpherical for FreeVector3D {
    type Spherical = FreeVector3D;

    fn to_spherical(&self) -> Self::Spherical {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem3D::Spherical)
    }
}

impl ToCylindrical for FreeVector3D {
    type Cylindrical = FreeVector3D;

    fn to_cylindrical(&self) -> Self::Cylindrical {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem3D::Cylindrical)
    }
}
