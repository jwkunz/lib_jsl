//! Statistics: descriptive statistics, correlation, hypothesis tests, kernel density estimation
//! and the bootstrap.

// Guards such as `!(x > 0.0)` are deliberate: unlike `x <= 0.0` they also reject NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod descriptive;
pub mod correlation;
