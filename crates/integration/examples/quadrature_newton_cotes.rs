//! The Newton-Cotes family: midpoint, trapezoid and Simpson rules, and the adaptive trapezoid.
//!
//! Run with `cargo run -p lib_jsl_integration --example quadrature_newton_cotes`.

use lib_jsl_integration::adaptive_trapezoid_integration_engine::AdaptiveTrapezoidIntegrationEngine;
use lib_jsl_integration::can_integrate::CanIntegrate1D;
use lib_jsl_integration::midpoint_integration_engine::MidpointIntegrationEngine;
use lib_jsl_integration::simpsons_1_3_integration_engine::Simpson13Engine;
use lib_jsl_integration::trapezoidal_integration_engine::TrapezoidalIntegrationEngine;
use std::cell::Cell;

fn main() {
    let exact = std::f64::consts::E - 1.0; // ∫₀¹ eˣ dx

    // Doubling the panels divides the midpoint and trapezoid errors by 4 and Simpson's by 16.
    // (The midpoint engine counts panels; the trapezoid and Simpson engines count points.)
    println!("panels   midpoint     trapezoid    Simpson");
    for n in [2, 4, 8, 16, 32] {
        let m = MidpointIntegrationEngine::new(n, 0.0, 1.0, false).integrate(f64::exp).unwrap();
        let t = TrapezoidalIntegrationEngine::new(n + 1, 0.0, 1.0, false).integrate(f64::exp).unwrap();
        let s = Simpson13Engine::new(n + 1, 0.0, 1.0, false).integrate(f64::exp).unwrap();
        println!("{n:6} {:12.3e} {:12.3e} {:12.3e}", m - exact, t - exact, s - exact);
    }

    // The midpoint error is about −½ the trapezoid error, so (2M + T)/3 cancels both: Simpson.
    let m = MidpointIntegrationEngine::new(4, 0.0, 1.0, false).integrate(f64::exp).unwrap();
    let t = TrapezoidalIntegrationEngine::new(5, 0.0, 1.0, false).integrate(f64::exp).unwrap();
    let s = Simpson13Engine::new(9, 0.0, 1.0, false).integrate(f64::exp).unwrap();
    println!("(2M₄ + T₄)/3 = {:.15}, Simpson with 8 panels = {s:.15}", (2.0 * m + t) / 3.0);

    // The adaptive trapezoid rule concentrates samples at a kink, where the uniform rule's
    // errors no longer cancel; on smooth integrands the uniform rule is better.
    let c = 1.0 / std::f64::consts::PI;
    let kink = |x: f64| (x - c).abs().sqrt();
    let exact_kink = 2.0 / 3.0 * (c.powf(1.5) + (1.0 - c).powf(1.5));
    let count = Cell::new(0);
    let counted = |x: f64| {
        count.set(count.get() + 1);
        kink(x)
    };
    let value = AdaptiveTrapezoidIntegrationEngine::new(1e-8, 0.0, 1.0).integrate(counted).unwrap();
    println!("adaptive trapezoid on √|x − 1/π|: error {:.1e} with {} evaluations", (value - exact_kink).abs(), count.get());
    let uniform = TrapezoidalIntegrationEngine::new(count.get(), 0.0, 1.0, false).integrate(kink).unwrap();
    println!("uniform trapezoid, same evaluations: error {:.1e}", (uniform - exact_kink).abs());
}
