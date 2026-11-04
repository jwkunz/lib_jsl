//! Structured surface and volume domains with quadrature weights.
//!
//! See the book entry [Fields on grids](https://jwkunz.github.io/lib_jsl/book/em-arrays/fields.html).

pub mod axis;
pub mod structured;

pub use axis::{Axis, QuadratureRule, gauss_legendre_nodes};
pub use structured::{Domain, DomainKind};
