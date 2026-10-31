//! Four-dimensional geometry extension traits.

use crate::common::GeometricPrimitive;

/// Marker trait for future 4D-specific primitive behavior.
pub trait GeometricPrimitive4DExt: GeometricPrimitive {}

pub mod christoffel_symbols;
pub mod space_time_vector;

pub use christoffel_symbols::{ChristoffelSymbols4D, Metric4, invert_4x4};
pub use space_time_vector::{Causality, SpaceTimeVector4D};
