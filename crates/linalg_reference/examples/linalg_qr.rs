//! QR factorisation: Gram-Schmidt against Householder reflections.
//!
//! Run with `cargo run -p lib_jsl_linalg_reference --example linalg_qr`.

use lib_jsl_linalg_reference::qr::{classical_gram_schmidt, householder, modified_gram_schmidt, orthogonality_error};
use ndarray::Array2;

fn main() {
    // The columns of a Vandermonde matrix in [0, 1] are nearly parallel: κ ≈ 10⁹ for degree 9.
    let a = Array2::from_shape_fn((30, 10), |(i, j)| (i as f64 / 29.0).powi(j as i32));
    for (name, (q, r)) in [
        ("classical Gram-Schmidt", classical_gram_schmidt(&a)),
        ("modified Gram-Schmidt", modified_gram_schmidt(&a)),
        ("Householder", householder(&a)),
    ] {
        let residual = (q.dot(&r) - &a).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        println!("{name:>24}: ‖QᵀQ − I‖ = {:.1e}, max |QR − A| = {residual:.1e}", orthogonality_error(&q));
    }
    // All three reproduce A = QR to rounding error; only Householder keeps Q orthogonal.
}
