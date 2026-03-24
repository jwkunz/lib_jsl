//! Concrete three-dimensional unit vector type for the public geometry API.
//!
//! [`UnitVector3D`] is used anywhere the concrete API needs a guaranteed direction, such as plane
//! normals or line directions returned from geometric queries.

use crate::geometry::common::{
    CanNormalize, CoordinatePrimitive, CrossProduct, DotProduct, GeometricPrimitive,
    GeometricPrimitive3D, GeometryMeasure, HasDimension, HasNorm, IsUnitVector, Normalize,
};
use crate::geometry::coordinate_systems::{
    CoordinateSystem3D, ToCartesian, ToCylindrical, ToSpherical,
};
use crate::geometry::three_d::coordinate_conversions;
use crate::geometry::three_d::GeometryVector3D;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Index, IndexMut};

/// Concrete 3D unit-vector implementation whose stored coordinates may be Cartesian, spherical, or cylindrical.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
pub struct UnitVector3D {
    coords: [GeometryMeasure; 3],
    coordinate_system: CoordinateSystem3D,
}

impl Eq for UnitVector3D {}

impl Hash for UnitVector3D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in self.coords {
            value.to_bits().hash(state);
        }
        self.coordinate_system.hash(state);
    }
}

impl UnitVector3D {
    /// Creates and normalizes a vector from Cartesian coordinates.
    pub fn new(x: GeometryMeasure, y: GeometryMeasure, z: GeometryMeasure) -> Self {
        Self::new_in_system(x, y, z, CoordinateSystem3D::Cartesian)
    }

    /// Creates and normalizes a vector from raw coordinates in the supplied system.
    pub fn new_in_system(
        first: GeometryMeasure,
        second: GeometryMeasure,
        third: GeometryMeasure,
        coordinate_system: CoordinateSystem3D,
    ) -> Self {
        let cartesian = coordinate_conversions::to_cartesian([first, second, third], coordinate_system);
        Self::from_cartesian_components(cartesian, coordinate_system)
    }

    /// Creates a unit vector by normalizing a free geometry vector.
    pub fn from_vector(vector: GeometryVector3D) -> Self {
        Self::from_cartesian_components(vector.cartesian_components(), vector.coordinate_system())
    }

    pub(crate) fn from_cartesian_components(
        coords: [GeometryMeasure; 3],
        coordinate_system: CoordinateSystem3D,
    ) -> Self {
        let norm = (coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2]).sqrt();
        let normalized = if norm == 0.0 {
            [1.0, 0.0, 0.0]
        } else {
            [coords[0] / norm, coords[1] / norm, coords[2] / norm]
        };
        Self {
            coords: coordinate_conversions::from_cartesian(normalized, coordinate_system),
            coordinate_system,
        }
    }

    /// Returns this unit vector as a free geometry vector with the same components.
    pub fn as_vector(self) -> GeometryVector3D {
        GeometryVector3D::from_cartesian_components(self.cartesian_components(), self.coordinate_system)
    }

    /// Returns the raw stored coordinates in the current coordinate system.
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

impl Display for UnitVector3D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UnitVector3D({:?}, {}, {}, {})",
            self.coordinate_system, self.coords[0], self.coords[1], self.coords[2]
        )
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
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
    }
}

impl CrossProduct for UnitVector3D {
    type Output = GeometryVector3D;

    fn cross(&self, rhs: &Self) -> <Self as CrossProduct>::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        GeometryVector3D::from_cartesian_components(
            [
                lhs[1] * rhs[2] - lhs[2] * rhs[1],
                lhs[2] * rhs[0] - lhs[0] * rhs[2],
                lhs[0] * rhs[1] - lhs[1] * rhs[0],
            ],
            self.coordinate_system,
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
        *self = Self::from_cartesian_components(self.cartesian_components(), self.coordinate_system);
    }
}

impl ToCartesian for UnitVector3D {
    type Cartesian = UnitVector3D;

    fn to_cartesian(&self) -> Self::Cartesian {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem3D::Cartesian)
    }
}

impl ToSpherical for UnitVector3D {
    type Spherical = UnitVector3D;

    fn to_spherical(&self) -> Self::Spherical {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem3D::Spherical)
    }
}

impl ToCylindrical for UnitVector3D {
    type Cylindrical = UnitVector3D;

    fn to_cylindrical(&self) -> Self::Cylindrical {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem3D::Cylindrical)
    }
}
