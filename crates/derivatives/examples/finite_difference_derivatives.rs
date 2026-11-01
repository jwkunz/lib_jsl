//! Finite difference derivatives, gradients, Hessians and Jacobians.
//!
//! Run with `cargo run -p lib_jsl_derivatives --example finite_difference_derivatives`.

use lib_jsl_derivatives::derivatives_1d_scalar::{DerivativeStencilOrder, derivative_2_point, derivative_stencil};
use lib_jsl_derivatives::derivatives_vectors::{gradient_2_point, hessian_2_point, jacobian_2_point};

fn main() {
    // d/dx of sin x at x = 1, with the default (balanced) step and with steps too large and too small.
    let exact = 1f64.cos();
    for step in [None, Some(1e-2), Some(1e-12)] {
        let d = derivative_2_point(&f64::sin, 1.0, step).unwrap();
        println!("central difference, step {step:?}: error {:.1e}", (d - exact).abs());
    }
    let d = derivative_stencil(&f64::sin, 1.0, None, DerivativeStencilOrder::FirstOrder).unwrap()[0];
    println!("five-point stencil, default step: error {:.1e}", (d - exact).abs());

    // The Rosenbrock function f = (1 − x)² + 100 (y − x²)² at (−1.2, 1).
    let rosenbrock = |v: &[f64]| (1.0 - v[0]).powi(2) + 100.0 * (v[1] - v[0] * v[0]).powi(2);
    let x = [-1.2, 1.0];
    let g = gradient_2_point(&rosenbrock, &x, None).unwrap();
    println!("gradient {g:.6?} (exact [-215.6, -88])");
    let h = hessian_2_point(&rosenbrock, &x, None).unwrap();
    println!("Hessian  {h:.4?} (exact [[1330, 480], [480, 200]])");

    // The Jacobian of polar-to-Cartesian coordinates (r, θ) ↦ (r cos θ, r sin θ); its determinant is r.
    let polar = |v: &[f64]| vec![v[0] * v[1].cos(), v[0] * v[1].sin()];
    let j = jacobian_2_point(&polar, &[2.0, 0.5], None).unwrap();
    println!("Jacobian determinant {:.10} (exact 2)", j[0][0] * j[1][1] - j[0][1] * j[1][0]);
}
