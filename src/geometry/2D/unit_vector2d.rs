//! Concrete two-dimensional unit vector type.

use crate::geometry::common::{
    CanNormalize, CoordinatePrimitive, DotProduct, GeometricPrimitive, GeometricPrimitive2D,
    GeometryMeasure, HasDimension, HasNorm, IsUnitVector, Normalize,
};
use crate::geometry::coordinate_systems::{CoordinateSystem2D, ToCartesian, ToPolar};
use crate::geometry::two_d::coordinate_conversions;
use crate::geometry::two_d::FreeVector2D;
use serde::Serialize;
use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Index, IndexMut};

/// Concrete 2D unit-vector implementation whose stored coordinates may be Cartesian or polar.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
pub struct UnitVector2D {
    coords: [GeometryMeasure; 2],
    coordinate_system: CoordinateSystem2D,
}

impl Eq for UnitVector2D {}

impl Hash for UnitVector2D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in self.coords {
            value.to_bits().hash(state);
        }
        self.coordinate_system.hash(state);
    }
}

impl UnitVector2D {
    /// Creates and normalizes a vector from Cartesian coordinates.
    pub fn new(x: GeometryMeasure, y: GeometryMeasure) -> Self {
        Self::new_in_system(x, y, CoordinateSystem2D::Cartesian)
    }

    /// Creates and normalizes a vector from raw coordinates in the supplied system.
    pub fn new_in_system(
        first: GeometryMeasure,
        second: GeometryMeasure,
        coordinate_system: CoordinateSystem2D,
    ) -> Self {
        let cartesian = coordinate_conversions::to_cartesian([first, second], coordinate_system);
        Self::from_cartesian_components(cartesian, coordinate_system)
    }

    pub(crate) fn from_cartesian_components(
        coords: [GeometryMeasure; 2],
        coordinate_system: CoordinateSystem2D,
    ) -> Self {
        let norm = (coords[0] * coords[0] + coords[1] * coords[1]).sqrt();
        let normalized = if norm == 0.0 {
            [1.0, 0.0]
        } else {
            [coords[0] / norm, coords[1] / norm]
        };
        Self {
            coords: coordinate_conversions::from_cartesian(normalized, coordinate_system),
            coordinate_system,
        }
    }

    /// Creates a unit vector by normalizing a free geometry vector.
    pub fn from_vector(vector: FreeVector2D) -> Self {
        Self::from_cartesian_components(vector.cartesian_components(), vector.coordinate_system())
    }

    /// Returns this unit vector as a free geometry vector with the same components.
    pub fn as_vector(self) -> FreeVector2D {
        FreeVector2D::from_cartesian_components(self.cartesian_components(), self.coordinate_system)
    }

    /// Returns the raw stored coordinates in the current coordinate system.
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

impl Display for UnitVector2D {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UnitVector2D({:?}, {}, {})",
            self.coordinate_system, self.coords[0], self.coords[1]
        )
    }
}

impl GeometricPrimitive for UnitVector2D {}
impl GeometricPrimitive2D for UnitVector2D {}
impl CoordinatePrimitive for UnitVector2D {}
impl IsUnitVector for UnitVector2D {}
impl HasDimension for UnitVector2D {
    const DIM: usize = 2;
}

impl AsRef<GeometryMeasure> for UnitVector2D {
    fn as_ref(&self) -> &GeometryMeasure {
        &self.coords[0]
    }
}

impl AsMut<GeometryMeasure> for UnitVector2D {
    fn as_mut(&mut self) -> &mut GeometryMeasure {
        &mut self.coords[0]
    }
}

impl Index<usize> for UnitVector2D {
    type Output = GeometryMeasure;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl IndexMut<usize> for UnitVector2D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl DotProduct for UnitVector2D {
    type Output = GeometryMeasure;

    fn dot(&self, rhs: &Self) -> <Self as DotProduct>::Output {
        let lhs = self.cartesian_components();
        let rhs = rhs.cartesian_components();
        lhs[0] * rhs[0] + lhs[1] * rhs[1]
    }
}

impl HasNorm for UnitVector2D {
    fn norm(&self) -> GeometryMeasure {
        1.0
    }
}

impl Normalize for UnitVector2D {
    fn normalized(&self) -> Self {
        *self
    }
}

impl CanNormalize for UnitVector2D {
    fn normalize(&mut self) {
        *self = Self::from_cartesian_components(self.cartesian_components(), self.coordinate_system);
    }
}

impl ToCartesian for UnitVector2D {
    type Cartesian = UnitVector2D;

    fn to_cartesian(&self) -> Self::Cartesian {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem2D::Cartesian)
    }
}

impl ToPolar for UnitVector2D {
    type Polar = UnitVector2D;

    fn to_polar(&self) -> Self::Polar {
        Self::from_cartesian_components(self.cartesian_components(), CoordinateSystem2D::Polar)
    }
}
