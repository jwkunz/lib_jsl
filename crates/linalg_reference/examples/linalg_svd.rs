//! The singular value decomposition: low-rank approximation, rank and the pseudoinverse.
//!
//! Run with `cargo run -p lib_jsl_linalg_reference --example linalg_svd`.

use lib_jsl_linalg::{SVD, SvdJacobi};
use lib_jsl_linalg_reference::naive_svd::svd_via_normal_matrix;
use ndarray::{Array2, array, s};

fn main() {
    // Ratings of 4 films by 5 viewers: two "tastes" explain most of the data.
    let ratings = array![[5.0, 4.0, 1.0, 1.0], [4.0, 5.0, 1.0, 2.0], [1.0, 1.0, 5.0, 4.0], [2.0, 1.0, 4.0, 5.0], [3.0, 3.0, 3.0, 3.0]];
    let (u, sigma, vt) = ratings.svd(true, true).unwrap();
    let (u, vt) = (u.unwrap(), vt.unwrap());
    println!("singular values: {sigma:.4}");

    // The best rank-2 approximation keeps the two largest singular triplets (Eckart-Young).
    let rank2 = u.slice(s![.., ..2]).dot(&Array2::from_diag(&sigma.slice(s![..2]).to_owned())).dot(&vt.slice(s![..2, ..]));
    println!("rank-2 approximation:\n{rank2:.2}");
    let error = (&ratings - &rank2).svd(false, false).unwrap().1[0];
    println!("‖A − A₂‖₂ = {error:.4} = σ₃ = {:.4}", sigma[2]);

    // The 2-norm condition number.
    println!("κ₂ = σ₁/σ₄ = {:.2}", sigma[0] / sigma[3]);

    // Jacobi and the textbook route through AᵀA agree with Golub-Kahan on this well-scaled matrix.
    println!("svd_jacobi: {:.4}", ratings.svd_jacobi(false, false).unwrap().1);
    println!("reference (AᵀA): {:.4}", svd_via_normal_matrix(&ratings).unwrap().singular_values);
}
