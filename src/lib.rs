pub mod prelude;
pub mod interpolation {
    pub mod interpolation_trait;
    pub mod linear_interpolator;
    pub mod natural_cubic_spline_interpolator;
    pub mod polynomial_interpolator;
}
pub mod number_theory {
    pub mod base_exponent_in_modulus;
    pub mod greatest_common_divisor;
    pub mod integer_square_root;
    pub mod multiplicative_inverse_modulo;
    pub mod text_coded_integer;
}
pub mod integration {
    pub mod adaptive_trapezoid_integration_engine;
    pub mod adaptive_gl7k15_integration_engine;
    pub mod calculate_midpoints;
    pub mod can_integrate;
    pub mod midpoint_integration_engine;
    pub mod simpsons_1_3_integration_engine;
    pub mod trapezoidal_integration_engine;
}
