/// Bilinear transform in zero-pole-gain form, similar to
/// `scipy.signal.bilinear_zpk`.
///
/// This maps analog zeros and poles from the `s`-plane into the digital
/// `z`-plane using Tustin's substitution:
///
/// `s = 2 fs (z - 1) / (z + 1)`
///
/// No pre-warping is performed, matching SciPy's `bilinear_zpk`.
use ndarray::Array1;
use num::Complex;

use crate::prelude::{C1D, ErrorsJSL, IsAnalytic};

#[derive(Clone, Debug, PartialEq)]
pub struct BilinearZpkResult {
    /// Zeros of the transformed digital filter.
    pub z: C1D,
    /// Poles of the transformed digital filter.
    pub p: C1D,
    /// Gain of the transformed digital filter.
    pub k: Complex<f64>,
}

fn validate_inputs<T: IsAnalytic, U: IsAnalytic>(
    z: &[T],
    p: &[T],
    k: U,
    fs: f64,
) -> Result<(), ErrorsJSL> {
    if !fs.is_finite() || fs <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("fs must be finite and > 0"));
    }
    if !k.to_complex().re.is_finite() || !k.to_complex().im.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("k must be finite"));
    }
    if z.iter().any(|x| !x.to_complex().re.is_finite() || !x.to_complex().im.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("zeros must be finite"));
    }
    if p.iter().any(|x| !x.to_complex().re.is_finite() || !x.to_complex().im.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("poles must be finite"));
    }
    if z.len() > p.len() {
        return Err(ErrorsJSL::InvalidInputRange(
            "bilinear_zpk requires number of zeros to be <= number of poles",
        ));
    }
    Ok(())
}

/// Apply the bilinear transform to analog zero-pole-gain data.
///
/// Arguments:
/// - `z`: analog zeros
/// - `p`: analog poles
/// - `k`: analog system gain
/// - `fs`: sampling frequency in ordinary units such as hertz
pub fn bilinear_zpk<T: IsAnalytic, U: IsAnalytic>(
    z: &[T],
    p: &[T],
    k: U,
    fs: f64,
) -> Result<BilinearZpkResult, ErrorsJSL> {
    validate_inputs(z, p, k, fs)?;

    let fs2 = 2.0 * fs;
    let fs2c = Complex::new(fs2, 0.0);

    let z_d = z
        .iter()
        .map(|zi| {
            let zi = zi.to_complex();
            (fs2c + zi) / (fs2c - zi)
        })
        .collect::<Vec<_>>();

    let p_d = p
        .iter()
        .map(|pi| {
            let pi = pi.to_complex();
            (fs2c + pi) / (fs2c - pi)
        })
        .collect::<Vec<_>>();

    let degree = p.len() - z.len();
    let mut z_out = z_d;
    z_out.extend((0..degree).map(|_| Complex::new(-1.0, 0.0)));

    let num = z
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, zi| acc * (fs2c - zi.to_complex()));
    let den = p
        .iter()
        .fold(Complex::new(1.0, 0.0), |acc, pi| acc * (fs2c - pi.to_complex()));
    if den.norm_sqr() <= 1e-24 {
        return Err(ErrorsJSL::RuntimeError(
            "bilinear_zpk gain normalization encountered zero denominator",
        ));
    }
    let k_d = k.to_complex() * num / den;

    Ok(BilinearZpkResult {
        z: Array1::from_vec(z_out),
        p: Array1::from_vec(p_d),
        k: k_d,
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
    fn test_bilinear_zpk_first_order_lowpass() {
        let result = bilinear_zpk::<Complex<f64>, f64>(&[], &[Complex::new(-1.0, 0.0)], 1.0, 2.0)
            .unwrap();

        assert_eq!(result.z.len(), 1);
        assert_eq!(result.p.len(), 1);
        assert_close_complex(result.z[0], Complex::new(-1.0, 0.0), 1e-12);
        assert_close_complex(result.p[0], Complex::new(0.6, 0.0), 1e-12);
        assert_close_complex(result.k, Complex::new(0.2, 0.0), 1e-12);
    }

    #[test]
    fn test_bilinear_zpk_gain_only_is_constant() {
        let result = bilinear_zpk::<Complex<f64>, f64>(&[], &[], 2.5, 10.0).unwrap();
        assert!(result.z.is_empty());
        assert!(result.p.is_empty());
        assert_close_complex(result.k, Complex::new(2.5, 0.0), 1e-12);
    }

    #[test]
    fn test_bilinear_zpk_supports_complex_zero_and_gain() {
        let result = bilinear_zpk(
            &[Complex::new(-1.0, 2.0)],
            &[Complex::new(-3.0, 1.0)],
            Complex::new(1.0, -0.5),
            4.0,
        )
        .unwrap();

        assert_eq!(result.z.len(), 1);
        assert_eq!(result.p.len(), 1);
        assert_close_complex(
            result.z[0],
            (Complex::new(8.0, 0.0) + Complex::new(-1.0, 2.0))
                / (Complex::new(8.0, 0.0) - Complex::new(-1.0, 2.0)),
            1e-12,
        );
        assert_close_complex(
            result.p[0],
            (Complex::new(8.0, 0.0) + Complex::new(-3.0, 1.0))
                / (Complex::new(8.0, 0.0) - Complex::new(-3.0, 1.0)),
            1e-12,
        );
    }

    #[test]
    fn test_bilinear_zpk_rejects_improper_system() {
        let err = bilinear_zpk(
            &[Complex::new(-1.0, 0.0), Complex::new(-2.0, 0.0)],
            &[Complex::new(-1.0, 0.0)],
            1.0,
            2.0,
        )
        .unwrap_err();
        assert!(matches!(err, ErrorsJSL::InvalidInputRange(_)));
    }
}
