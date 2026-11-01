//! Bessel and Airy functions.
//!
//! Run with `cargo run -p lib_jsl_special_functions --example special_bessel`.

use lib_jsl_special_functions::{airy_ai, bessel_fractional, bessel_ik, bessel_j_integer, bessel_jy, bessel_spherical};
use lib_jsl_root_finding::{bracket_roots_scan, brent_method};

fn main() {
    // The first zeros of J₀ give the vibration frequencies of a circular drum.
    let zeros: Vec<f64> = bracket_roots_scan(|x| bessel_j_integer(0, x), 0.5, 15.0, 300)
        .into_iter()
        .map(|(a, b)| brent_method(|x| bessel_j_integer(0, x), a, b, 1e-14).unwrap())
        .collect();
    println!("zeros of J₀: {zeros:.10?}");
    let (j, y) = bessel_jy(2.5, 7.0);
    println!("J_2.5(7) = {j:.15}, Y_2.5(7) = {y:.15}");
    // Half-integer orders are elementary: J_½(x) = √(2/(πx)) sin x, and j₀(x) = sin x / x.
    let x: f64 = 3.0;
    println!("J_½(3) = {:.15} = {:.15}", bessel_fractional(0.5, x), (2.0 / (std::f64::consts::PI * x)).sqrt() * x.sin());
    println!("j₀(3) = {:.15} = {:.15}", bessel_spherical(0, x), x.sin() / x);
    let (i, k) = bessel_ik(1.0, 2.0);
    println!("I₁(2) = {i:.15}, K₁(2) = {k:.15}");
    println!("Ai(−2.338107410459767) = {:.2e} (the first zero)", airy_ai(-2.338107410459767));
}
