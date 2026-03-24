//! Coordinate-system enums, markers, and conversion traits.

use crate::geometry::common::GeometricPrimitive;

/// Supported coordinate systems for 2D primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoordinateSystem2D {
    /// Standard x/y Cartesian coordinates.
    Cartesian,
    /// Radius/angle polar coordinates.
    Polar,
}

/// Supported coordinate systems for 3D primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoordinateSystem3D {
    /// Standard x/y/z Cartesian coordinates.
    Cartesian,
    /// Radius/inclination/azimuth spherical coordinates.
    Spherical,
    /// Radius/angle/height cylindrical coordinates.
    Cylindrical,
}

/// Marker trait for 2D coordinate-system tags.
pub trait IsCoordinateSystem2D {}

/// Marker trait for 2D Cartesian coordinate systems.
pub trait IsCartesian2D: IsCoordinateSystem2D {}

/// Marker trait for 2D polar coordinate systems.
pub trait IsPolar: IsCoordinateSystem2D {}

/// Marker trait for 3D coordinate-system tags.
pub trait IsCoordinateSystem3D {}

/// Marker trait for 3D Cartesian coordinate systems.
pub trait IsCartesian3D: IsCoordinateSystem3D {}

/// Marker trait for 3D spherical coordinate systems.
pub trait IsSpherical: IsCoordinateSystem3D {}

/// Marker trait for 3D cylindrical coordinate systems.
pub trait IsCylindrical: IsCoordinateSystem3D {}

/// Converts a value into a Cartesian representation.
pub trait ToCartesian {
    /// Cartesian result type.
    type Cartesian: GeometricPrimitive;

    /// Returns the Cartesian representation of `self`.
    fn to_cartesian(&self) -> Self::Cartesian;
}

/// Converts a value into a polar representation.
pub trait ToPolar {
    /// Polar result type.
    type Polar: GeometricPrimitive;

    /// Returns the polar representation of `self`.
    fn to_polar(&self) -> Self::Polar;
}

/// Converts a value into a spherical representation.
pub trait ToSpherical {
    /// Spherical result type.
    type Spherical: GeometricPrimitive;

    /// Returns the spherical representation of `self`.
    fn to_spherical(&self) -> Self::Spherical;
}

/// Converts a value into a cylindrical representation.
pub trait ToCylindrical {
    /// Cylindrical result type.
    type Cylindrical: GeometricPrimitive;

    /// Returns the cylindrical representation of `self`.
    fn to_cylindrical(&self) -> Self::Cylindrical;
}
