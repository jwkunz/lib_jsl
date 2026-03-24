use ndarray::{Array1, Array2};
use ndarray_linalg::Eig;
use num::Complex;

use crate::prelude::{C1D, ErrorsJSL, IsAnalytic};

/// Evaluate a polynomial at `x` using Horner's rule.
///
/// The coefficient slice is interpreted in descending powers, so
/// `[a0, a1, ..., an]` represents:
///
/// `a0*x^n + a1*x^(n-1) + ... + an`
pub fn polynomial_eval<T: IsAnalytic>(coeffs: &[T], x: Complex<f64>) -> Complex<f64> {
    let mut acc = Complex::new(0.0, 0.0);
    for coeff in coeffs {
        acc = acc * x + coeff.to_complex();
    }
    acc
}

/// Form a monic polynomial from its roots.
///
/// The returned coefficients are in descending powers.
pub fn polynomial_from_roots(roots: &[Complex<f64>]) -> C1D {
    let mut coeffs = vec![Complex::new(1.0, 0.0)];
    for &root in roots {
        let mut next = vec![Complex::new(0.0, 0.0); coeffs.len() + 1];
        for (i, &coeff) in coeffs.iter().enumerate() {
            next[i] += coeff;
            next[i + 1] -= coeff * root;
        }
        coeffs = next;
    }
    Array1::from_vec(coeffs)
}

/// Compute the roots of a polynomial using a companion-matrix eigenvalue solve.
///
/// The coefficient slice is interpreted in descending powers, so
/// `[a0, a1, ..., an]` represents:
///
/// `a0*x^n + a1*x^(n-1) + ... + an`
pub fn polynomial_roots<T: IsAnalytic>(coeffs: &[T]) -> Result<C1D, ErrorsJSL> {
    if coeffs.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange(
            "polynomial coefficients must be non-empty",
        ));
    }
    if coeffs.len() == 1 {
        return Ok(Array1::from_vec(vec![]));
    }

    let lead = coeffs[0].to_complex();
    if lead.norm_sqr() <= 1e-24 {
        return Err(ErrorsJSL::InvalidInputRange(
            "leading polynomial coefficient must be non-zero",
        ));
    }

    let order = coeffs.len() - 1;
    let mut companion = Array2::from_elem((order, order), Complex::new(0.0, 0.0));
    for row in 1..order {
        companion[[row, row - 1]] = Complex::new(1.0, 0.0);
    }
    for row in 0..order {
        companion[[row, order - 1]] = -coeffs[order - row].to_complex() / lead;
    }

    let (evals, _) = companion
        .eig()
        .map_err(|_| ErrorsJSL::RuntimeError("failed to compute polynomial roots"))?;
    Ok(evals)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sort_by_real_then_imag(values: &mut [Complex<f64>]) {
        values.sort_by(|a, b| {
            a.re
                .partial_cmp(&b.re)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a.im
                        .partial_cmp(&b.im)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
    }

    #[test]
    fn test_polynomial_eval_quadratic() {
        let coeffs = [1.0, -3.0, 2.0];
        let value = polynomial_eval(&coeffs, Complex::new(2.0, 0.0));
        assert!(value.norm() < 1e-12);
    }

    #[test]
    fn test_polynomial_from_roots() {
        let roots = [Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let coeffs = polynomial_from_roots(&roots);
        assert_eq!(coeffs.len(), 3);
        assert!((coeffs[0] - Complex::new(1.0, 0.0)).norm() < 1e-12);
        assert!((coeffs[1] - Complex::new(-3.0, 0.0)).norm() < 1e-12);
        assert!((coeffs[2] - Complex::new(2.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_polynomial_roots_quadratic() {
        let coeffs = [1.0, -3.0, 2.0];
        let mut roots = polynomial_roots(&coeffs).unwrap().to_vec();
        let mut expected = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        sort_by_real_then_imag(&mut roots);
        sort_by_real_then_imag(&mut expected);

        for (actual, expected) in roots.iter().zip(expected.iter()) {
            assert!((*actual - *expected).norm() < 1e-9);
        }
    }
}
