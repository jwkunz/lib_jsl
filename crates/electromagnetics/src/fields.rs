//! Electromagnetic fields: structured domains, field containers and the operations on them.

pub mod domain;
pub mod field_container;
pub mod interpolation;
pub mod physics;
pub mod surface_calculus;
pub mod coordinate_conversion;
pub mod vector_calculus;

pub use domain::{Axis, Domain, DomainKind, QuadratureRule};
pub use field_container::{FieldQuantity, HarmonicField, Rank, TimeAxis, TimeField};
pub use interpolation::{Extrapolation, InterpolationMethod};
