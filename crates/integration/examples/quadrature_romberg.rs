//! Romberg integration: the extrapolation tableau and the adaptive driver.
//!
//! Run with `cargo run -p lib_jsl_integration --example quadrature_romberg`.

use lib_jsl_integration::romberg::RombergEngine;

fn main() {
    let exact = std::f64::consts::E - 1.0;
    let engine = RombergEngine::new(0.0, 1.0);

    // Row k uses the trapezoid rule with 2^k panels; column j removes the h^{2j} error term.
    let table = engine.tableau(f64::exp, 4).unwrap();
    for (k, row) in table.iter().enumerate() {
        let errors: Vec<String> = row.iter().map(|v| format!("{:9.1e}", (v - exact).abs())).collect();
        println!("k = {k}: {}", errors.join(" "));
    }

    // The driver adds rows until the diagonal stops changing.
    let (value, estimate, levels) = engine.integrate_with_error(f64::exp).unwrap();
    println!("∫₀¹ eˣ dx ≈ {value:.15} after {levels} levels (estimate {estimate:.1e}, true error {:.1e})", (value - exact).abs());
}
