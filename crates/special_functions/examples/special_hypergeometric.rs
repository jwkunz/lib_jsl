//! Hypergeometric functions.
//!
//! Run with `cargo run -p lib_jsl_special_functions --example special_hypergeometric`.

use lib_jsl_special_functions::{hypergeometric, hypergeometric_0f1, hypergeometric_1f1};

fn main() {
    // ₂F₁ contains many elementary functions: arcsin z = z ₂F₁(½, ½; 3/2; z²).
    let z: f64 = 0.6;
    println!("z ₂F₁(½, ½; 3/2; z²) = {:.15}, arcsin z = {:.15}", z * hypergeometric(0.5, 0.5, 1.5, z * z), z.asin());
    // Gauss's theorem at z = 1: ₂F₁(a, b; c; 1) = Γ(c)Γ(c−a−b) / (Γ(c−a)Γ(c−b)).
    println!("₂F₁(1, 1; 3; 1) = {} (exactly 2)", hypergeometric(1.0, 1.0, 3.0, 1.0));
    // Kummer's function at a negative integer a is a (Laguerre) polynomial.
    println!("₁F₁(−2; 1; x) at x = 3: {} (L₂(3) = 1 − 6 + 4.5 = −0.5)", hypergeometric_1f1(-2.0, 1.0, 3.0));
    println!("₀F₁(; 1; −4) = J₀(4) = {:.15}", hypergeometric_0f1(1.0, -4.0));
}
