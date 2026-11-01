//! Integrals over boxes in several dimensions: tensor Gauss rules and adaptive cubature.
//!
//! Run with `cargo run -p lib_jsl_integration --example quadrature_cubature`.

use lib_jsl_integration::cubature::{CubatureOptions, adaptive_cubature, tensor_gauss_legendre};

fn main() {
    // A smooth integrand: ∫ e^{x+y+z} over the unit cube = (e − 1)³.
    let exact = (std::f64::consts::E - 1.0).powi(3);
    let smooth = |x: &[f64]| (x[0] + x[1] + x[2]).exp();
    for points in [2, 4, 8] {
        let v = tensor_gauss_legendre(smooth, &[0.0; 3], &[1.0; 3], points).unwrap();
        println!("tensor Gauss, {points}³ = {:4} points: error {:.1e}", points * points * points, (v - exact).abs());
    }
    let r = adaptive_cubature(smooth, &[0.0; 3], &[1.0; 3], CubatureOptions::default()).unwrap();
    println!("adaptive Genz-Malik: error {:.1e} (estimate {:.1e}) with {} evaluations", (r.value - exact).abs(), r.error, r.evaluations);

    // A corner singularity: ∫∫ 1/√(x² + y²) over the unit square = 2 ln(1 + √2).
    let exact = 2.0 * (1.0 + 2f64.sqrt()).ln();
    let singular = |x: &[f64]| 1.0 / (x[0] * x[0] + x[1] * x[1]).sqrt();
    let v = tensor_gauss_legendre(singular, &[0.0; 2], &[1.0; 2], 300).unwrap();
    println!("corner singularity, tensor Gauss with 90000 points: error {:.1e}", (v - exact).abs());
    let options = CubatureOptions { absolute_tolerance: 0.0, relative_tolerance: 1e-9, max_evaluations: 1_000_000 };
    let r = adaptive_cubature(singular, &[0.0; 2], &[1.0; 2], options).unwrap();
    println!("corner singularity, adaptive: error {:.1e} with {} evaluations (converged: {})", (r.value - exact).abs(), r.evaluations, r.converged);
}
