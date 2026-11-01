//! The Riemann zeta function, the polylogarithm and Fermi-Dirac integrals.
//!
//! Run with `cargo run -p lib_jsl_special_functions --example special_zeta`.

use lib_jsl_special_functions::{fermi_dirac_integral, hurwitz_zeta, polylog, zeta};

fn main() {
    let pi = std::f64::consts::PI;
    println!("ζ(2) = {:.15} (π²/6 = {:.15})", zeta(2.0), pi * pi / 6.0);
    println!("ζ(−1) = {:.15} (−1/12), ζ(−2) = {}", zeta(-1.0), zeta(-2.0));
    println!("ζ(3, 2) = ζ(3) − 1 = {:.15}", hurwitz_zeta(3.0, 2.0));
    println!("Li₂(½) = {:.15} (π²/12 − ln²2/2 = {:.15})", polylog(2.0, 0.5), pi * pi / 12.0 - 2f64.ln().powi(2) / 2.0);
    // The electron density of a semiconductor: F_½ of the reduced Fermi level.
    for eta in [-4.0, 0.0, 4.0, 20.0] {
        println!("F_½({eta}) = {:.10}", fermi_dirac_integral(0.5, eta));
    }
}
