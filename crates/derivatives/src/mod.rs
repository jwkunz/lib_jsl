//! Numerical differentiation by finite differences: derivatives of functions of one variable, and
//! gradients, Hessians and Jacobians of functions of several.
//!
//! Every routine chooses its default step to balance the truncation error of the difference
//! formula against the rounding error of subtracting nearly equal function values. See the book
//! entry [Finite difference derivatives](https://jwkunz.github.io/lib_jsl/book/derivatives/finite-differences.html).

pub mod derivatives_1d_scalar;
pub mod derivatives_vectors;
pub mod adaptive;
