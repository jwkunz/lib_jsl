//! Linear least squares: normal equations, QR and the production solver.
//!
//! Run with `cargo run -p lib_jsl_linalg_reference --example linalg_least_squares`.

use lib_jsl_linalg::LeastSquaresSvd;
use lib_jsl_linalg_reference::least_squares::{normal_equations, qr_least_squares};
use ndarray::{Array1, Array2, array};

fn main() {
    // Fit y = c₀ + c₁ t + c₂ t² to noisy measurements.
    let t: Array1<f64> = Array1::linspace(0.0, 2.0, 9);
    let y = array![1.02, 1.29, 1.84, 2.47, 3.33, 4.24, 5.36, 6.61, 8.04];
    let a = Array2::from_shape_fn((9, 3), |(i, j)| t[i].powi(j as i32));

    let fit = a.least_squares(&y).unwrap();
    println!("coefficients {:.4}, rank {}, residual sum of squares {:.4}", fit.solution, fit.rank, fit.residual_sum_of_squares.unwrap());
    println!("condition number σ₁/σ₃ = {:.1}", fit.singular_values[0] / fit.singular_values[2]);
    println!("normal equations: {:.4}", normal_equations(&a, &y).unwrap());
    println!("reference QR:     {:.4}", qr_least_squares(&a, &y).unwrap());

    // A rank-deficient system has many minimisers; least_squares returns the one of least norm.
    let dependent = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let r = dependent.least_squares(&array![2.0, 4.0, 6.0]).unwrap();
    println!("rank {}: minimum-norm solution {:.4}", r.rank, r.solution);
}
