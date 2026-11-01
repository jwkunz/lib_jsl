//! Exponential, sine, cosine and Fresnel integrals.
//!
//! Run with `cargo run -p lib_jsl_special_functions --example special_exponential_integrals`.

use lib_jsl_special_functions::{cosine_integral, exponential_integral, exponential_integral_ei, exponential_integral_en, fresnel_integrals, sine_integral};

fn main() {
    println!("E₁(1) = {:.15}, E₃(2) = {:.15}", exponential_integral(1.0), exponential_integral_en(3, 2.0));
    // The logarithmic integral li(x) = Ei(ln x) approximates the number of primes below x.
    let x: f64 = 1e6;
    println!("li(10⁶) = Ei(ln 10⁶) = {:.1} (78498 primes below 10⁶)", exponential_integral_ei(x.ln()));
    // The Gibbs overshoot of a Fourier series: (2/π) Si(π) = 1.1790.
    println!("(2/π) Si(π) = {:.10}", 2.0 / std::f64::consts::PI * sine_integral(std::f64::consts::PI));
    println!("Ci(1) = {:.15}", cosine_integral(1.0));
    // Fresnel integrals: the diffraction pattern at a straight edge, at t = 1.
    let (s, c) = fresnel_integrals(1.0);
    println!("S(1) = {s:.15}, C(1) = {c:.15}");
}
