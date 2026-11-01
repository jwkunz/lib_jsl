//! The gamma function and its relatives.
//!
//! Run with `cargo run -p lib_jsl_special_functions --example special_gamma`.

use lib_jsl_special_functions::{beta, binomial_coefficient, digamma, factorial, gamma, ln_binomial_coefficient, ln_gamma};

fn main() {
    // Γ interpolates the factorial: Γ(n + 1) = n!, and Γ(½) = √π.
    println!("Γ(6) = {}, 5! = {}", gamma(6.0), factorial(5));
    println!("Γ(½)² = {:.15} (π = {:.15})", gamma(0.5).powi(2), std::f64::consts::PI);
    // Γ overflows past 171.6, but its logarithm does not.
    println!("Γ(200) = {}, ln Γ(200) = {:.6}", gamma(200.0), ln_gamma(200.0));
    // The volume of the unit ball in 10 dimensions, π^(n/2) / Γ(n/2 + 1).
    println!("volume of the unit 10-ball: {:.6}", std::f64::consts::PI.powi(5) / gamma(6.0));
    // B(a, b) = Γ(a) Γ(b) / Γ(a + b), and the digamma function ψ = Γ'/Γ (ψ(1) = −γ).
    println!("B(2.5, 1.5) = {:.12}, ψ(1) = {:.15}", beta(2.5, 1.5), digamma(1.0));
    // Binomial coefficients: exact in integers while they fit, or as logarithms.
    println!("C(50, 25) = {}, ln C(10000, 5000) = {:.6}", binomial_coefficient(50, 25), ln_binomial_coefficient(10_000, 5_000));
}
