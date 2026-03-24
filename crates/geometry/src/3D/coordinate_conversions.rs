//! Coordinate-system conversion helpers for three-dimensional concrete geometry types.

use crate::common::GeometryMeasure;
use crate::coordinate_systems::CoordinateSystem3D;

/// Converts 3D coordinates from the supplied system into Cartesian `[x, y, z]`.
pub fn to_cartesian(coords: [GeometryMeasure; 3], system: CoordinateSystem3D) -> [GeometryMeasure; 3] {
    match system {
        CoordinateSystem3D::Cartesian => coords,
        CoordinateSystem3D::Spherical => {
            let radius = coords[0];
            let inclination = coords[1];
            let azimuth = coords[2];
            [
                radius * inclination.sin() * azimuth.cos(),
                radius * inclination.sin() * azimuth.sin(),
                radius * inclination.cos(),
            ]
        }
        CoordinateSystem3D::Cylindrical => {
            let radius = coords[0];
            let azimuth = coords[1];
            let height = coords[2];
            [radius * azimuth.cos(), radius * azimuth.sin(), height]
        }
    }
}

/// Converts Cartesian `[x, y, z]` coordinates into the requested 3D system.
pub fn from_cartesian(coords: [GeometryMeasure; 3], system: CoordinateSystem3D) -> [GeometryMeasure; 3] {
    match system {
        CoordinateSystem3D::Cartesian => coords,
        CoordinateSystem3D::Spherical => {
            let x = coords[0];
            let y = coords[1];
            let z = coords[2];
            let radius = (x * x + y * y + z * z).sqrt();
            let inclination = if radius == 0.0 { 0.0 } else { (z / radius).acos() };
            let azimuth = y.atan2(x);
            [radius, inclination, azimuth]
        }
        CoordinateSystem3D::Cylindrical => {
            let x = coords[0];
            let y = coords[1];
            let z = coords[2];
            let radius = (x * x + y * y).sqrt();
            let azimuth = y.atan2(x);
            [radius, azimuth, z]
        }
    }
}
