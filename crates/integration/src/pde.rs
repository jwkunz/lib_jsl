//! Partial differential equations: finite difference, finite element and spectral methods.

// Guards such as `!(x > 0.0)` are deliberate: unlike `x <= 0.0` they also reject NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod finite_differences;
pub mod finite_elements;
