//! Coordinate-system conversion helpers for two-dimensional concrete geometry types.

use crate::geometry::common::GeometryMeasure;
use crate::geometry::coordinate_systems::CoordinateSystem2D;

/// Converts 2D coordinates from the supplied system into Cartesian `[x, y]`.
pub fn to_cartesian(coords: [GeometryMeasure; 2], system: CoordinateSystem2D) -> [GeometryMeasure; 2] {
    match system {
        CoordinateSystem2D::Cartesian => coords,
        CoordinateSystem2D::Polar => {
            let radius = coords[0];
            let angle = coords[1];
            [radius * angle.cos(), radius * angle.sin()]
        }
    }
}

/// Converts Cartesian `[x, y]` coordinates into the requested 2D system.
pub fn from_cartesian(coords: [GeometryMeasure; 2], system: CoordinateSystem2D) -> [GeometryMeasure; 2] {
    match system {
        CoordinateSystem2D::Cartesian => coords,
        CoordinateSystem2D::Polar => {
            let x = coords[0];
            let y = coords[1];
            let radius = (x * x + y * y).sqrt();
            let angle = y.atan2(x);
            [radius, angle]
        }
    }
}
