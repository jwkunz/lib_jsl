//! Special mathematical functions.

// Guards such as `!(x > 0.0)` are deliberate: unlike `x <= 0.0` they also reject NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod gamma;
pub mod beta;
pub mod factorial;
pub mod binomial_coefficients;
pub mod incomplete_gamma;
pub mod error_function;
pub mod exponential_integrals;
pub mod incomplete_beta;
pub mod bessel_functions_integer;
pub mod bessel_functions_fractional;
pub mod bessel_functions_spherical;
pub mod airy_functions;
pub mod fresnel_integrals;
pub mod cosine_integrals;
pub mod sine_integrals;
pub mod dawsons_integral;
pub mod fermi_dirac_integrals;
pub mod inverse_x_log_x;
pub mod lambert_w;
pub mod zeta;
pub mod polylogarithm;
pub mod elliptic_integrals;
pub mod jacobian_elliptic_functions;
pub mod hypergeometric_functions;
pub mod statistical_functions;
pub mod orthogonal_polynomials;
pub mod spherical_harmonics;

pub use gamma::{
    gamma,
    ln_gamma,
    digamma,
};
pub use beta::{
    beta,
    ln_beta,
};
pub use factorial::{
    MAX_U64_FACTORIAL,
    factorial_checked,
    factorial,
    factorial_f64,
    ln_factorial,
};
pub use binomial_coefficients::{
    binomial_coefficient_checked,
    binomial_coefficient,
    ln_binomial_coefficient,
};
pub use incomplete_gamma::{
    incomplete_gamma,
    incomplete_gamma_upper,
};
pub use error_function::{
    error_function,
    erfc,
    erf_inv,
};
pub use exponential_integrals::{
    exponential_integral,
    exponential_integral_en,
    exponential_integral_ei,
};
pub use incomplete_beta::incomplete_beta;
pub use bessel_functions_integer::{
    bessel_j_integer,
    bessel_y_integer,
    bessel_jy_integer,
    bessel_i_integer,
    bessel_k_integer,
};
pub use bessel_functions_fractional::{
    bessel_jy,
    bessel_ik,
    bessel_fractional,
    bessel_y_fractional,
    bessel_i_fractional,
    bessel_k_fractional,
};
pub use bessel_functions_spherical::{
    bessel_spherical,
    bessel_spherical_y,
};
pub use airy_functions::{
    airy_ai,
    airy_bi,
};
pub use fresnel_integrals::fresnel_integrals;
pub use cosine_integrals::cosine_integral;
pub use sine_integrals::{
    sine_integral,
    sine_integral_shifted,
};
pub use dawsons_integral::dawson_integral;
pub use fermi_dirac_integrals::fermi_dirac_integral;
pub use inverse_x_log_x::{
    lambert_w0,
    lambert_w_m1,
    inverse_x_log_x,
    inverse_x_log_x_lower,
};
pub use elliptic_integrals::{
    carlson_rc,
    carlson_rf,
    carlson_rd,
    carlson_rj,
    elliptic_integral,
    elliptic_integral_second,
    elliptic_integral_third,
    elliptic_f,
    elliptic_e,
    elliptic_pi,
};
pub use jacobian_elliptic_functions::jacobian_elliptic;
pub use hypergeometric_functions::{
    hypergeometric,
    hypergeometric_1f1,
    hypergeometric_0f1,
};
pub use statistical_functions::{
    normal_pdf,
    normal_cdf,
    normal_quantile,
    gamma_cdf,
    chi_square_cdf,
    chi_square_survival,
    chi_square_quantile,
    beta_cdf,
    student_t_cdf,
    student_t_quantile,
    f_cdf,
    poisson_pmf,
    poisson_cdf,
    binomial_pmf,
    binomial_cdf,
};
pub use orthogonal_polynomials::{
    legendre,
    legendre_with_derivative,
    chebyshev_t,
    chebyshev_u,
    hermite,
    hermite_he,
    laguerre,
    jacobi,
};
pub use spherical_harmonics::{
    associated_legendre,
    normalized_associated_legendre,
    spherical_harmonic,
    real_spherical_harmonic,
};
pub use lambert_w::lambert_w;
pub use zeta::{
    zeta,
    hurwitz_zeta,
};
pub use polylogarithm::polylog;
