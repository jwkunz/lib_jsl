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
    pub mod adaptive_gl7k15_integration_engine;
    pub mod adaptive_trapezoid_integration_engine;
    pub mod calculate_midpoints;
    pub mod can_integrate;
    pub mod dormand_prince;
    pub mod midpoint_integration_engine;
    pub mod monte_carlo_integration;
    pub mod runge_kutta_4;
    pub mod simpsons_1_3_integration_engine;
    pub mod trapezoidal_integration_engine;
}
pub mod random {
    pub mod distributions;
    pub mod split_mix_64;
    pub mod uniform_generator;
    pub mod xoshiro256plusplus;
    pub mod histogram;
}
pub mod filters {
    pub mod linear_filter;
}
pub mod optimization {
    pub mod optimization_traits;
    pub mod golden_section_search;
    pub mod gradient_descent;
    pub mod newton_raphson_method;
    pub mod simulated_annealing;
    pub mod nelder_mead_method;
}
pub mod derivatives{
    pub mod derivatives_1d_scalar;
}
