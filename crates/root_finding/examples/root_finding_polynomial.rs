//! Roots of polynomials, real and complex.
//!
//! Run with `cargo run -p lib_jsl_root_finding --example root_finding_polynomial`.

use lib_jsl_root_finding::{polynomial_roots, polynomial_roots_complex};

fn main() {
    // Coefficients run from the highest power down: x⁴ − 10x² + 9 = (x² − 1)(x² − 9).
    println!("real roots: {:?}", polynomial_roots(&[1.0, 0.0, -10.0, 0.0, 9.0]));

    // x³ − 1: one real root and two complex ones.
    println!("real roots of x³ − 1: {:?}", polynomial_roots(&[1.0, 0.0, 0.0, -1.0]));
    for z in polynomial_roots_complex(&[1.0, 0.0, 0.0, -1.0]).unwrap() {
        println!("  complex root {:.6} {:+.6}i", z.re, z.im);
    }

    // A repeated root is reported once by polynomial_roots, but with its multiplicity (as a
    // small cluster) by polynomial_roots_complex: (x − 2)³ = x³ − 6x² + 12x − 8.
    println!("(x − 2)³: {:?}", polynomial_roots(&[1.0, -6.0, 12.0, -8.0]));
    println!("as eigenvalues: {:?}", polynomial_roots_complex(&[1.0, -6.0, 12.0, -8.0]).unwrap());
}
