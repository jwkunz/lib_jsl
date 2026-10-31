pub mod interpolation_trait;
pub mod nd;
pub mod one_d;
pub mod three_d;
pub mod two_d;

pub use one_d::{linear_interpolator, natural_cubic_spline_interpolator, polynomial_interpolator};
