//! Production eigensolvers: Hermitian and general, real and complex.
//!
//! Run with `cargo run -p lib_jsl_linalg_reference --example linalg_eigen`.

use lib_jsl_linalg::{Eig, EigVals, Eigh, UPLO};
use ndarray::{Array2, array};
use num_complex::Complex;

fn main() {
    // Normal modes of three masses joined by springs: K v = ω² M v with M = I.
    let k = array![[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]];
    let (omega2, modes) = k.eigh(UPLO::Lower).unwrap();
    println!("ω² = {omega2:.6}");
    println!("mode shapes (columns):\n{modes:.4}");

    // A rotation by 90° has complex eigenvalues ±i.
    println!("eigenvalues of a rotation: {:.6}", array![[0.0, -1.0], [1.0, 0.0]].eigvals().unwrap());

    // A general real matrix: eigenvalues and eigenvectors, checked by A v = λ v.
    let a = array![[4.0, 1.0, -2.0], [0.5, 3.0, 1.0], [2.0, -1.0, 1.0]];
    let (w, v) = a.eig().unwrap();
    let ac = a.mapv(|x| Complex::new(x, 0.0));
    let residual = (ac.dot(&v) - &v * &w).iter().map(|z| z.norm()).fold(0.0, f64::max);
    println!("eigenvalues {w:.6}; max |A v − λ v| = {residual:.1e}");

    // Hermitian complex matrices have real eigenvalues.
    let h: Array2<Complex<f64>> = array![[Complex::new(2.0, 0.0), Complex::new(0.0, 1.0)], [Complex::new(0.0, -1.0), Complex::new(2.0, 0.0)]];
    println!("Hermitian eigenvalues: {:.6}", h.eigh(UPLO::Lower).unwrap().0);
}
