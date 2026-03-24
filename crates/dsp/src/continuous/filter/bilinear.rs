/// Bilinear transform utility similar to `scipy.signal.bilinear`.
///
/// This converts an analog transfer function
///
/// `H_a(s) = (b[0] s^Q + b[1] s^(Q-1) + ... + b[Q]) /`
/// `         (a[0] s^P + a[1] s^(P-1) + ... + a[P])`
///
/// into a digital IIR transfer function by substituting
///
/// `s = 2 fs (z - 1) / (z + 1)`
///
/// without pre-warping, matching SciPy's `bilinear`.
use ndarray::Array1;
use num::Complex;

use lib_jsl_core::{C1D, ErrorsJSL, IsAnalytic};

#[derive(Clone, Debug, PartialEq)]
pub struct BilinearResult {
    /// Numerator coefficients of the digital transfer function in powers of `z^-1`.
    pub beta: C1D,
    /// Denominator coefficients of the digital transfer function in powers of `z^-1`.
    pub alpha: C1D,
}

fn validate_inputs<T: IsAnalytic>(b: &[T], a: &[T], fs: f64) -> Result<(), ErrorsJSL> {
    if b.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange(
            "numerator coefficients must be non-empty",
        ));
    }
    if a.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange(
            "denominator coefficients must be non-empty",
        ));
    }
    if !fs.is_finite() || fs <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("fs must be finite and > 0"));
    }
    if a[0].to_complex().norm_sqr() <= 1e-24 {
        return Err(ErrorsJSL::InvalidInputRange(
            "leading denominator coefficient must be non-zero",
        ));
    }
    Ok(())
}

fn poly_mul(lhs: &[Complex<f64>], rhs: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut out = vec![Complex::new(0.0, 0.0); lhs.len() + rhs.len() - 1];
    for (i, &x) in lhs.iter().enumerate() {
        for (j, &h) in rhs.iter().enumerate() {
            out[i + j] += x * h;
        }
    }
    out
}

fn poly_add_assign(dst: &mut [Complex<f64>], src: &[Complex<f64>], scale: Complex<f64>) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += scale * *s;
    }
}

fn poly_pow(base: &[Complex<f64>], exponent: usize) -> Vec<Complex<f64>> {
    if exponent == 0 {
        return vec![Complex::new(1.0, 0.0)];
    }
    let mut out = vec![Complex::new(1.0, 0.0)];
    for _ in 0..exponent {
        out = poly_mul(&out, base);
    }
    out
}

fn build_digital_polynomial<T: IsAnalytic>(
    coeffs: &[T],
    order: usize,
    n: usize,
    kappa: f64,
) -> Vec<Complex<f64>> {
    let one_minus_x = [Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)];
    let one_plus_x = [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
    let mut out = vec![Complex::new(0.0, 0.0); n + 1];

    for (idx, coeff) in coeffs.iter().enumerate() {
        let power_s = order - idx;
        let left = poly_pow(&one_minus_x, power_s);
        let right = poly_pow(&one_plus_x, n - power_s);
        let basis = poly_mul(&left, &right);
        let scale = coeff.to_complex() * kappa.powi(power_s as i32);
        poly_add_assign(&mut out, &basis, scale);
    }

    out
}

/// Apply the bilinear transform to analog numerator/denominator polynomials.
///
/// Arguments:
/// - `b`: numerator coefficients in descending powers of `s`
/// - `a`: denominator coefficients in descending powers of `s`
/// - `fs`: sampling frequency in ordinary units such as hertz
///
/// Returns:
/// - [`BilinearResult`] containing digital numerator `beta` and denominator
///   `alpha`, both expressed in ascending powers of `z^-1`
pub fn bilinear<T: IsAnalytic>(b: &[T], a: &[T], fs: f64) -> Result<BilinearResult, ErrorsJSL> {
    validate_inputs(b, a, fs)?;

    let q = b.len() - 1;
    let p = a.len() - 1;
    let n = q.max(p);
    let kappa = 2.0 * fs;

    let mut beta = build_digital_polynomial(b, q, n, kappa);
    let mut alpha = build_digital_polynomial(a, p, n, kappa);

    let alpha0 = alpha[0];
    if alpha0.norm_sqr() <= 1e-24 {
        return Err(ErrorsJSL::RuntimeError(
            "digital denominator leading coefficient is zero",
        ));
    }

    for coeff in &mut beta {
        *coeff /= alpha0;
    }
    for coeff in &mut alpha {
        *coeff /= alpha0;
    }

    Ok(BilinearResult {
        beta: Array1::from_vec(beta),
        alpha: Array1::from_vec(alpha),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close_complex(actual: Complex<f64>, expected: Complex<f64>, tol: f64) {
        assert!(
            (actual - expected).norm() < tol,
            "actual={actual}, expected={expected}, tol={tol}"
        );
    }

    #[test]
    fn test_bilinear_first_order_lowpass() {
        let result = bilinear(&[1.0], &[1.0, 1.0], 2.0).unwrap();

        assert_eq!(result.beta.len(), 2);
        assert_eq!(result.alpha.len(), 2);
        assert_close_complex(result.beta[0], Complex::new(0.2, 0.0), 1e-12);
        assert_close_complex(result.beta[1], Complex::new(0.2, 0.0), 1e-12);
        assert_close_complex(result.alpha[0], Complex::new(1.0, 0.0), 1e-12);
        assert_close_complex(result.alpha[1], Complex::new(-0.6, 0.0), 1e-12);
    }

    #[test]
    fn test_bilinear_gain_only_is_constant() {
        let result = bilinear(&[2.5], &[1.0], 10.0).unwrap();
        assert_eq!(result.beta.len(), 1);
        assert_eq!(result.alpha.len(), 1);
        assert_close_complex(result.beta[0], Complex::new(2.5, 0.0), 1e-12);
        assert_close_complex(result.alpha[0], Complex::new(1.0, 0.0), 1e-12);
    }

    #[test]
    fn test_bilinear_supports_complex_coefficients() {
        let result = bilinear(
            &[Complex::new(1.0, 1.0)],
            &[Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)],
            2.0,
        )
        .unwrap();

        assert_close_complex(result.beta[0], Complex::new(0.2, 0.2), 1e-12);
        assert_close_complex(result.beta[1], Complex::new(0.2, 0.2), 1e-12);
        assert_close_complex(result.alpha[0], Complex::new(1.0, 0.0), 1e-12);
        assert_close_complex(result.alpha[1], Complex::new(-0.6, 0.0), 1e-12);
    }

    #[test]
    fn test_bilinear_rejects_bad_fs() {
        let err = bilinear(&[1.0], &[1.0], 0.0).unwrap_err();
        assert!(matches!(err, ErrorsJSL::InvalidInputRange(_)));
    }
}
