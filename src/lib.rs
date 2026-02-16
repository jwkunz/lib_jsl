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
    pub mod histogram;
    pub mod split_mix_64;
    pub mod uniform_generator;
    pub mod xoshiro256plusplus;
}
pub mod dsp{
    pub mod stream_operator;
    pub mod windows;
    pub mod sinc;
    pub mod convolve;
    pub mod cross_correlation;
    pub mod filters {
        pub mod discrete_linear_filter;
        pub mod firwin;
        pub mod firwin2;
        pub mod overlap_and_add_fir;
        pub mod derivative_filter;
        pub mod boxcar_integrator;
    }
    pub mod transformations{
        pub mod channel_impairment;
        pub mod frequency_mixer;
        pub mod non_linearity_transformers;
    }
    pub mod resampling{
        pub mod polyphase_arbitrary_resampling;
        pub mod polyphase_integer_resampling;
    }
}

pub mod optimization {
    pub mod golden_section_search;
    pub mod gradient_descent;
    pub mod nelder_mead_method;
    pub mod newton_raphson_method;
    pub mod optimization_traits;
    pub mod simulated_annealing;
}
pub mod derivatives {
    pub mod derivatives_1d_scalar;
    pub mod derivatives_vectors;
}
pub mod ffts {
    pub mod best_fft;
    pub mod bluestein_fft;
    pub mod fft_engine_trait;
    pub mod optimized_radix2;
    pub mod optimized_split_radix;
    pub mod simd_fft;
    pub mod simple_cooley_tukey;
    #[cfg(test)]
    pub mod test_bench_data;
}
