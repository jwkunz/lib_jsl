//! Polynomials by their coefficients and roots, and decimal-string integers of any size.

use lib_jsl_number_theory::polynomial::{polynomial_eval, polynomial_from_roots, polynomial_roots};
use lib_jsl_number_theory::text_coded_integer::TextCodedInteger;
use num_complex::Complex;

fn main() {
    // x³ − 6x² + 11x − 6 = (x − 1)(x − 2)(x − 3), coefficients in descending powers.
    let p = [1.0, -6.0, 11.0, -6.0];
    println!("p(4) = {}", polynomial_eval(&p, Complex::new(4.0, 0.0)));
    let mut roots: Vec<f64> = polynomial_roots(&p).unwrap().iter().map(|r| r.re).collect();
    roots.sort_by(f64::total_cmp);
    println!("roots of p: {:.12?}", roots);
    let q = polynomial_from_roots(&[Complex::new(0.0, 1.0), Complex::new(0.0, -1.0)]);
    println!("(x − i)(x + i) has coefficients {:?}", q.iter().map(|c| c.re).collect::<Vec<_>>());
    // 2^100 and 30! in decimal-string arithmetic.
    let mut power = TextCodedInteger::new(1);
    for _ in 0..100 {
        power = power * TextCodedInteger::new(2);
    }
    let mut factorial = TextCodedInteger::new(1);
    for k in 2..=30 {
        factorial = factorial * TextCodedInteger::new(k);
    }
    println!("2^100 = {power}");
    println!("30! = {factorial}");
    println!("30! / 2^100 = {} remainder {}", factorial.clone() / power.clone(), factorial % power);
}
