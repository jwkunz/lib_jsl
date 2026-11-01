//! Solving linear systems: textbook Gaussian elimination and production LU.
//!
//! Run with `cargo run -p lib_jsl_linalg_reference --example linalg_lu`.

use lib_jsl_linalg::{Determinant, Factorize, Inverse, Solve};
use lib_jsl_linalg_reference::{gaussian_elimination, lu};
use ndarray::{Array1, array};

fn main() {
    // Currents in a three-loop resistor network (Kirchhoff's laws).
    let a = array![[10.0, -2.0, -3.0], [-2.0, 12.0, -4.0], [-3.0, -4.0, 15.0]];
    let b = array![12.0, 0.0, 6.0];

    // The textbook algorithm, from the reference crate.
    let x_ref = gaussian_elimination::solve_with_partial_pivoting(&a, &b).unwrap();
    println!("reference elimination: {x_ref:.6}");

    // The production solver: LU with partial pivoting, for real or complex matrices.
    let x = a.solve(&b).unwrap();
    println!("production solve:      {x:.6}");

    // Factor once, then solve for many right-hand sides at O(n²) each.
    let f = a.factorize().unwrap();
    for rhs in [array![1.0, 0.0, 0.0], array![0.0, 1.0, 0.0]] {
        println!("A x = {rhs}: x = {:.6}", f.solve(&rhs).unwrap());
    }
    println!("det A = {:.6} (reference {:.6})", a.det().unwrap(), lu::lu(&a).unwrap().determinant());
    let check = a.dot(&a.inv().unwrap());
    println!("max |A A⁻¹ − I| = {:.1e}", (&check - &ndarray::Array2::<f64>::eye(3)).iter().fold(0.0f64, |m, v| m.max(v.abs())));

    // A singular matrix is reported, not divided by zero.
    println!("singular: {:?}", array![[1.0, 2.0], [2.0, 4.0]].solve(&Array1::from(vec![1.0, 1.0])));
}
