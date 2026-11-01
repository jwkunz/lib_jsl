//! The Lambert W function.
//!
//! Run with `cargo run -p lib_jsl_special_functions --example special_lambert_w`.

use lib_jsl_special_functions::{inverse_x_log_x, lambert_w, lambert_w0, lambert_w_m1};
use num_complex::Complex;

fn main() {
    // The omega constant Ω = W(1), with Ω e^Ω = 1.
    let omega = lambert_w0(1.0);
    println!("Ω = W₀(1) = {omega:.15}; Ω e^Ω = {:.15}", omega * omega.exp());
    // Both real solutions of x eˣ = −0.2.
    println!("W₀(−0.2) = {:.15}, W₋₁(−0.2) = {:.15}", lambert_w0(-0.2), lambert_w_m1(-0.2));
    // Solving x ln x = 10, and 2ˣ = 5x (x = −W(−ln 2 / 5) / ln 2).
    println!("x ln x = 10 at x = {:.15}", inverse_x_log_x(10.0));
    let l = 2f64.ln();
    for branch in [lambert_w0, lambert_w_m1] {
        let x = -branch(-l / 5.0) / l;
        println!("2^x = 5x at x = {x:.12} (check: {:.3e})", 2f64.powf(x) - 5.0 * x);
    }
    // Complex branches.
    let w = lambert_w(1, Complex::new(1.0, 0.0));
    println!("W₁(1) = {:.12} {:+.12}i", w.re, w.im);
}
