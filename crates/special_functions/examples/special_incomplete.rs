//! Incomplete gamma and beta functions and the error function.
//!
//! Run with `cargo run -p lib_jsl_special_functions --example special_incomplete`.

use lib_jsl_special_functions::{dawson_integral, erf_inv, erfc, error_function, incomplete_beta, incomplete_gamma, incomplete_gamma_upper};

fn main() {
    // The probability that a normal variable lies within one, two and three standard deviations.
    for k in 1..=3 {
        println!("P(|Z| < {k}) = {:.10}", error_function(k as f64 / 2f64.sqrt()));
    }
    // In the far tail, erfc keeps full relative accuracy where 1 − erf has none left.
    println!("erfc(7) = {:e}, 1 − erf(7) = {:e}", erfc(7.0), 1.0 - error_function(7.0));
    println!("erf⁻¹(0.95) = {:.12}", erf_inv(0.95));
    // P(a, x) and Q(a, x): the chance that a Poisson process of rate 1 has its 3rd event by time 2.
    println!("P(3, 2) = {:.12}, Q(3, 2) = {:.12}", incomplete_gamma(3.0, 2.0), incomplete_gamma_upper(3.0, 2.0));
    // I_x(a, b): the chance that the median of 5 uniform samples is below 0.3.
    println!("I_0.3(3, 3) = {:.12}", incomplete_beta(3.0, 3.0, 0.3));
    println!("Dawson F(1) = {:.15}", dawson_integral(1.0));
}
