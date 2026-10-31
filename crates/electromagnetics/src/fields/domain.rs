//! Structured surface and volume domains with quadrature weights.

pub mod axis;
pub mod structured;

pub use axis::{Axis, QuadratureRule, gauss_legendre_nodes};
pub use structured::{Domain, DomainKind};
