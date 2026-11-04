//! Transmission lines and microwave networks: line transformations, two-port parameters and the
//! Smith chart. Phasors use the engineering convention `e^{+jωt}`.
//!
//! See the book entry [Transmission lines, two-port networks and the Smith chart](https://jwkunz.github.io/lib_jsl/book/em-waves/transmission-lines.html).

// Guards such as `!(x > 0.0)` are deliberate: unlike `x <= 0.0` they also reject NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod s_parameters;
pub mod smith_chart;
pub mod transmission_lines;
