//! Polynomial interpolation: Newton and barycentric forms, Chebyshev points and series.
//!
//! Run with `cargo run -p lib_jsl_interpolation --example interpolation_polynomial`.

use lib_jsl_core::R1D;
use lib_jsl_interpolation::interpolation_trait::InterpolationTrait;
use lib_jsl_interpolation::one_d::chebyshev_approximations::ChebyshevApproximation;
use lib_jsl_interpolation::one_d::lagrange_interpolant::LagrangeInterpolant;
use lib_jsl_interpolation::one_d::polynomial_interpolator::PolynomialInterpolator;
use std::f64::consts::PI;

fn main() {
    // The same cubic through four points, in Newton and barycentric form.
    let x = R1D::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    let y = R1D::from_vec(vec![1.0, 2.0, 0.0, -1.0]);
    let mut newton = PolynomialInterpolator::new(x.view(), y.view()).unwrap();
    let mut barycentric = LagrangeInterpolant::new(x.view(), y.view()).unwrap();
    println!("p(1.5): Newton {:.12}, barycentric {:.12}", newton.interpolate_at(1.5).unwrap(), barycentric.interpolate_at(1.5).unwrap());

    // The Runge function at 21 points: equally spaced against Chebyshev.
    let runge = |x: f64| 1.0 / (1.0 + 25.0 * x * x);
    let max_error = |nodes: Vec<f64>| {
        let (x, y) = (R1D::from_vec(nodes.clone()), R1D::from_vec(nodes.iter().map(|&v| runge(v)).collect()));
        let mut p = LagrangeInterpolant::new(x.view(), y.view()).unwrap();
        (0..=1000).map(|i| { let t = -1.0 + i as f64 / 500.0; (p.interpolate_at(t).unwrap() - runge(t)).abs() }).fold(0.0, f64::max)
    };
    let n = 21;
    println!("21 equally spaced points: max error {:.2e}", max_error((0..n).map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64).collect()));
    println!("21 Chebyshev points:      max error {:.2e}", max_error((0..n).map(|j| (PI * j as f64 / (n - 1) as f64).cos()).collect()));

    // A Chebyshev series for a smooth function: fit 60 terms, drop the negligible tail, differentiate.
    let mut c = ChebyshevApproximation::new(|x: f64| (x * x).cos() * x.exp(), 0.0, 2.0, 60).unwrap();
    c.truncate(1e-15);
    println!("Chebyshev series for cos(x²)eˣ on [0, 2]: {} of 60 coefficients exceed 1e-15", c.coefficients().len());
    let d = c.derivative();
    let exact = (-2.0 * (1.0f64).sin() + 1f64.cos()) * 1f64.exp();
    println!("derivative at x = 1: {:.13} (exact {exact:.13})", d.evaluate(1.0));
}
