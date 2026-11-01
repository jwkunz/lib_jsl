//! Cholesky factorisation of symmetric positive-definite matrices.
//!
//! Run with `cargo run -p lib_jsl_linalg_reference --example linalg_cholesky`.

use lib_jsl_linalg::{Cholesky, Diag, SolveTriangular, UPLO};
use lib_jsl_linalg_reference::cholesky::cholesky;
use ndarray::array;

fn main() {
    // A covariance matrix: symmetric and positive definite.
    let c = array![[4.0, 1.2, 0.8], [1.2, 2.0, 0.3], [0.8, 0.3, 1.0]];
    let l = c.cholesky(UPLO::Lower).unwrap();
    println!("L =\n{l:.6}");
    println!("reference factor agrees: {}", (&l - &cholesky(&c).unwrap()).iter().all(|v| v.abs() < 1e-14));

    // Solve C x = b with two triangular solves: L y = b, then Lᵀ x = y.
    let b = array![1.0, 2.0, 3.0];
    let y = l.solve_triangular(UPLO::Lower, Diag::NonUnit, &b).unwrap();
    let x = l.t().solve_triangular(UPLO::Upper, Diag::NonUnit, &y).unwrap();
    println!("x = {x:.6}, residual {:.1e}", (c.dot(&x) - &b).iter().fold(0.0f64, |m, v| m.max(v.abs())));

    // Correlated random samples: if z has independent standard normal entries, L z has covariance C.
    let z = array![0.3, -1.1, 0.7];
    println!("correlated sample L z = {:.6}", l.dot(&z));

    // Cholesky is also the cheapest test of positive definiteness.
    println!("indefinite matrix: {:?}", array![[1.0, 2.0], [2.0, 1.0]].cholesky(UPLO::Lower));
}
