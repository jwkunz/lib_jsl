//! Orthogonal polynomials and spherical harmonics.
//!
//! Run with `cargo run -p lib_jsl_special_functions --example special_orthogonal_polynomials`.

use lib_jsl_special_functions::{chebyshev_t, hermite, jacobi, laguerre, legendre, real_spherical_harmonic, spherical_harmonic};

fn main() {
    let x = 0.3;
    println!("P₅({x}) = {:.15}", legendre(5, x));
    // T_n(cos t) = cos(n t).
    let t: f64 = 0.7;
    println!("T₇(cos 0.7) = {:.15}, cos 4.9 = {:.15}", chebyshev_t(7, t.cos()), (7.0 * t).cos());
    println!("H₄(1) = {}, L₃^(½)(2) = {:.15}, P₄^(1,2)(0.5) = {:.15}", hermite(4, 1.0), laguerre(3, 0.5, 2.0), jacobi(4, 1.0, 2.0, 0.5));
    // Orthogonality of Legendre polynomials, by midpoint-rule integration of P₃ P₅ and P₄².
    let n = 200_000;
    let h = 2.0 / n as f64;
    let (mut i35, mut i44) = (0.0, 0.0);
    for k in 0..n {
        let x = -1.0 + (k as f64 + 0.5) * h;
        i35 += legendre(3, x) * legendre(5, x) * h;
        i44 += legendre(4, x).powi(2) * h;
    }
    println!("∫P₃P₅ = {i35:.2e}, ∫P₄² = {i44:.8} (2/9 = {:.8})", 2.0 / 9.0);
    // Spherical harmonics: complex and real forms.
    let y = spherical_harmonic(2, 1, 0.8, 0.3);
    println!("Y₂¹(0.8, 0.3) = {:.12} {:+.12}i; real form {:.12}", y.re, y.im, real_spherical_harmonic(2, 1, 0.8, 0.3));
}
