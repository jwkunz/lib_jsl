//! Globally adaptive Gauss-Kronrod quadrature with error estimates.
//!
//! Run with `cargo run -p lib_jsl_integration --example quadrature_gauss_kronrod`.

use lib_jsl_integration::adaptive_gl7k15_integration_engine::AdaptiveGL7K15IntegrationEngine;
use lib_jsl_integration::can_integrate::CanIntegrate1D;

/// A test integrand: its name, the function on [0, 1] and the exact integral.
type Case = (&'static str, fn(f64) -> f64, f64);

fn main() {
    let engine = AdaptiveGL7K15IntegrationEngine::new(1e-10, 0.0, 1.0);
    let cases: [Case; 4] = [
        ("e^x", f64::exp, std::f64::consts::E - 1.0),
        ("ln x (singular at 0)", f64::ln, -1.0),
        ("x^-0.9 (strongly singular)", |x| x.powf(-0.9), 10.0),
        ("|x - 0.5| (kink)", |x| (x - 0.5).abs(), 0.25),
    ];
    for (name, f, exact) in cases {
        let (value, estimate, evaluations) = engine.integrate_with_error(f).unwrap();
        println!("{name:28} {value:.15}  true error {:.1e}  estimate {estimate:.1e}  {evaluations} evaluations", (value - exact).abs());
    }

    // `integrate` fails rather than return a value that misses the tolerance.
    let mut limited = AdaptiveGL7K15IntegrationEngine::new(1e-14, 0.0, 1.0);
    limited.max_subintervals = 5;
    println!("with 5 panels at most: {:?}", limited.integrate(|x: f64| x.powf(-0.9)));
}
