//! Umbrella facade crate for the `lib_jsl` workspace.
//!
//! This crate re-exports the domain crates so downstream code can continue using a compact,
//! top-level API such as [`geometry`], [`dsp`], or [`ffts`] while the repository itself remains
//! organized as a multi-crate workspace.

pub use lib_jsl_core as prelude;
pub use lib_jsl_derivatives as derivatives;
pub use lib_jsl_dsp as dsp;
pub use lib_jsl_ffts as ffts;
pub use lib_jsl_geometry as geometry;
pub use lib_jsl_integration as integration;
pub use lib_jsl_interpolation as interpolation;
pub use lib_jsl_number_theory as number_theory;
pub use lib_jsl_optimization as optimization;
pub use lib_jsl_random as random;
