//! Gauss-Legendre quadrature: exactness, mapping to an interval, and fast convergence.
//!
//! Run with `cargo run -p lib_jsl_integration --example quadrature_gauss_legendre`.

use lib_jsl_integration::can_integrate::CanIntegrate1D;
use lib_jsl_integration::gaussian_quadrature::{GaussLegendreEngine, gauss_legendre};

fn main() {
    // A 3-point rule on [−1, 1]: nodes 0 and ±√(3/5), weights 8/9 and 5/9.
    let rule = gauss_legendre(3).unwrap();
    println!("nodes   {:?}", rule.nodes);
    println!("weights {:?}", rule.weights);

    // Exact for every polynomial of degree ≤ 5 = 2·3 − 1 ...
    let quintic = |x: f64| 3.0 * x.powi(5) - x.powi(4) + 2.0 * x * x + 1.0;
    println!("∫ quintic = {:.15} (exact {:.15})", rule.apply(quintic), -2.0 / 5.0 + 4.0 / 3.0 + 2.0);
    // ... but not degree 6.
    println!("∫ x⁶      = {:.15} (exact {:.15})", rule.apply(|x| x.powi(6)), 2.0 / 7.0);

    // On a general interval, the engine maps the nodes affinely.
    for n in [4, 8, 16] {
        let engine = GaussLegendreEngine::new(n, 0.0, std::f64::consts::PI).unwrap();
        let value = engine.integrate(f64::sin).unwrap();
        println!("n = {n:2}: ∫₀^π sin x dx ≈ {value:.16}, error {:.1e}", (value - 2.0).abs());
    }
}
