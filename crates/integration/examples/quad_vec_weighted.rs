//! Vector integrands with `quad_vec`, and weighted integrals with `quad_weighted`.
//!
//! Run with `cargo run -p lib_jsl_integration --example quad_vec_weighted`.

use lib_jsl_integration::quad_vec::{QuadVecOptions, quad_vec};
use lib_jsl_integration::weighted::{LogFactor, Weight, WeightedOptions, quad_weighted};
use std::f64::consts::PI;

fn main() {
    // Bessel's integral J_n(x) = (1/π) ∫₀^π cos(nθ − x sin θ) dθ for n = 0..5 at x = 2.5, all at once:
    // the six components share one set of subintervals.
    let x = 2.5;
    let options = QuadVecOptions { epsrel: 1e-12, ..Default::default() };
    let r = quad_vec(|t| (0..6).map(|n| (n as f64 * t - x * t.sin()).cos() / PI).collect(), 0.0, PI, &options).unwrap();
    println!("J_0..5(2.5) = {:.12?}", r.value);
    println!("  {} evaluations, {} subintervals, error estimate {:.1e}", r.nfev, r.intervals, r.error);

    // A Fourier integral: ∫₀^∞ cos 3x / (1 + x²) dx = (π/2) e^{−3}.
    let o = WeightedOptions::default();
    let r = quad_weighted(|x| 1.0 / (1.0 + x * x), 0.0, f64::INFINITY, Weight::Cos(3.0), &o).unwrap();
    println!("Fourier:   {:.12}  error {:.1e}", r.value, (r.value - PI / 2.0 * (-3.0f64).exp()).abs());

    // An algebraic-logarithmic weight: ∫₀¹ ln x / √x dx = −4.
    let w = Weight::Algebraic { alpha: -0.5, beta: 0.0, log: LogFactor::Left };
    let r = quad_weighted(|_| 1.0, 0.0, 1.0, w, &o).unwrap();
    println!("Algebraic: {:.12}  error {:.1e}", r.value, (r.value + 4.0).abs());

    // A principal value: ⨍₀² dx / (x − 1/2) = ln 3.
    let r = quad_weighted(|_| 1.0, 0.0, 2.0, Weight::Cauchy(0.5), &o).unwrap();
    println!("Cauchy:    {:.12}  error {:.1e}", r.value, (r.value - 3f64.ln()).abs());
}
