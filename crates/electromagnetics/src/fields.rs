//! Electromagnetic fields: structured domains, field containers and the operations on them.
//!
//! See the book entry [Fields on grids](https://jwkunz.github.io/lib_jsl/book/em-arrays/fields.html).

pub mod domain;
pub mod field_container;
pub mod interpolation;
pub mod physics;
pub mod spherical_waves;
pub mod surface_calculus;
pub mod coordinate_conversion;
pub mod vector_calculus;

pub use domain::{Axis, Domain, DomainKind, QuadratureRule};
pub use field_container::{FieldQuantity, HarmonicField, Rank, TimeAxis, TimeField};
pub use interpolation::{Extrapolation, InterpolationMethod};
pub use spherical_waves::{SphericalWaveExpansion, vector_spherical_harmonics};
