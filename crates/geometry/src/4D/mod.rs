//! Four-dimensional geometry extension traits.

use crate::common::GeometricPrimitive;

/// Marker trait for future 4D-specific primitive behavior.
pub trait GeometricPrimitive4DExt: GeometricPrimitive {}

pub struct SpaceTimeVector4D {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub struct ChristoffelSymbols4D;

