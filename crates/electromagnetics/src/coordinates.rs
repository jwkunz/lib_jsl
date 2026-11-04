//! Coordinate systems, polarization bases and rigid transforms.
//!
//! See the book entry [Fields on grids](https://jwkunz.github.io/lib_jsl/book/em-arrays/fields.html#bases-interpolation-and-rigid-transforms).

pub mod polarization;
pub mod systems;
pub mod transforms;

pub use polarization::PolarizationBasis;
pub use systems::{CoordinateSystem, Position};
pub use transforms::RigidTransform;
