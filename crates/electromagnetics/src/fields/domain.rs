//! Structured surface and volume domains with quadrature weights.

pub mod axis;

pub use axis::{Axis, QuadratureRule, gauss_legendre_nodes};
