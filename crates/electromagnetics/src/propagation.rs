//! Propagation and system models: free-space and empirical path loss, multipath, atmospheric
//! attenuation and the radar equation.
//!
//! See the book entry [Propagation](https://jwkunz.github.io/lib_jsl/book/em-waves/propagation.html).

// Guards such as `!(x > 0.0)` are deliberate: unlike `x <= 0.0` they also reject NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod atmosphere;
pub mod multipath;
pub mod path_loss;
pub mod radar_equation;
