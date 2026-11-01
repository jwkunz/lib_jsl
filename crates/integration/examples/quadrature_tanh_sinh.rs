//! Tanh-sinh quadrature on integrands with endpoint singularities.
//!
//! Run with `cargo run -p lib_jsl_integration --example quadrature_tanh_sinh`.

use lib_jsl_integration::tanh_sinh::TanhSinhEngine;
use std::f64::consts::PI;

/// An integrand on [0, 1] with its name and exact value.
type Case = (&'static str, fn(f64) -> f64, f64);

fn main() {
    let engine = TanhSinhEngine::new(0.0, 1.0);
    let cases: [Case; 3] = [
        ("ln x", f64::ln, -1.0),
        ("1/√x", |x| 1.0 / x.sqrt(), 2.0),
        ("ln(x) ln(1 − x)", |x| x.ln() * (-x).ln_1p(), 2.0 - PI * PI / 6.0),
    ];
    for (name, f, exact) in cases {
        let (value, estimate) = engine.integrate_with_error(f).unwrap();
        println!("∫₀¹ {name:16} ≈ {value:+.15}  error {:.1e}  (estimate {estimate:.1e})", (value - exact).abs());
    }

    // A singularity at an endpoint away from zero: 1/√(1 − x²) on [−1, 1]. Computing 1 − x
    // near x = 1 cancels catastrophically; the complement form passes the exact offset instead.
    let (value, _) = TanhSinhEngine::new(-1.0, 1.0)
        .integrate_with_complement(|x, offset| {
            let (one_minus, one_plus) = if offset < 0.0 { (-offset, 1.0 + x) } else { (1.0 - x, offset) };
            1.0 / (one_minus * one_plus).sqrt()
        })
        .unwrap();
    println!("∫₋₁¹ dx/√(1 − x²) ≈ {value:.15}  error {:.1e}", (value - PI).abs());
}
