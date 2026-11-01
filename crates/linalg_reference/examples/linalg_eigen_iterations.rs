//! Textbook eigenvalue algorithms: power iteration, Jacobi rotations and the QR algorithm.
//!
//! Run with `cargo run -p lib_jsl_linalg_reference --example linalg_eigen_iterations`.

use lib_jsl_linalg_reference::jacobi_eigenvalue::jacobi_eigen;
use lib_jsl_linalg_reference::power_iteration::{inverse_iteration, power_iteration, rayleigh_quotient_iteration};
use lib_jsl_linalg_reference::qr_algorithm::{shifted_qr, unshifted_qr};
use ndarray::{Array1, Array2};

fn main() {
    // The second-difference matrix of a vibrating string with 8 interior points: eigenvalues
    // 2 − 2 cos(kπ/9), k = 1..8, the squared frequencies of its modes.
    let n = 8;
    let a = Array2::from_shape_fn((n, n), |(i, j)| if i == j { 2.0 } else if i.abs_diff(j) == 1 { -1.0 } else { 0.0 });
    let exact: Vec<f64> = (1..=n).map(|k| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / 9.0).cos()).collect();
    println!("exact eigenvalues: {exact:.6?}");

    let start = Array1::from_elem(n, 1.0) + Array1::from_shape_fn(n, |i| 0.01 * i as f64);
    let p = power_iteration(&a, &start, 1e-10, 10_000).unwrap();
    println!("power iteration: largest λ = {:.10} in {} steps", p.value, p.iterations);
    let i = inverse_iteration(&a, 0.1, &start, 1e-10, 1000).unwrap();
    println!("inverse iteration (shift 0.1): smallest λ = {:.10} in {} steps", i.value, i.iterations);
    let r = rayleigh_quotient_iteration(&a, &start, 1e-12, 100).unwrap();
    println!("Rayleigh quotient iteration: λ = {:.12} in {} steps", r.value, r.iterations);

    let j = jacobi_eigen(&a, 1e-14).unwrap();
    println!("Jacobi: {:.6} in {} sweeps", j.values, j.sweeps);
    let u = unshifted_qr(&a, 1e-12, 100_000).unwrap();
    let s = shifted_qr(&a, 1e-14, 1000).unwrap();
    println!("QR algorithm: {} unshifted steps, {} shifted steps; {:.6}", u.iterations, s.iterations, s.values);
}
