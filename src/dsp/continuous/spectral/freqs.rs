/// Analog frequency-response utility similar to `scipy.signal.freqs`.
///
/// This evaluates the transfer function of a continuous-time linear filter:
///
/// `H(jw) = (b[0] (jw)^M + b[1] (jw)^(M-1) + ... + b[M]) /`
/// `        (a[0] (jw)^N + a[1] (jw)^(N-1) + ... + a[N])`
///
/// Plotting support from SciPy is intentionally omitted here.
use ndarray::{Array1, Array2};
use ndarray_linalg::Eig;
use num::Complex;

use crate::prelude::{C1D, ErrorsJSL, IsAnalytic, R1D};

#[derive(Clone, Debug, PartialEq)]
pub struct FreqsResult {
    /// Angular frequency axis in rad/s.
    pub w: R1D,
    /// Complex analog frequency response evaluated at each frequency in `w`.
    pub h: C1D,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FreqsWorN {
    /// Automatically generate a frequency grid with the requested number of points.
    NumPoints(usize),
    /// Evaluate at the explicitly provided angular frequencies in rad/s.
    Frequencies(Vec<f64>),
}

fn validate_coefficients<T: IsAnalytic>(b: &[T], a: &[T]) -> Result<(), ErrorsJSL> {
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
    Ok(())
}

fn polyval<T: IsAnalytic>(coeffs: &[T], x: Complex<f64>) -> Complex<f64> {
    let mut acc = Complex::new(0.0, 0.0);
    for coeff in coeffs {
        acc = acc * x + coeff.to_complex();
    }
    acc
}

fn polynomial_roots<T: IsAnalytic>(coeffs: &[T]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
    if coeffs.len() <= 1 {
        return Ok(vec![]);
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
    Ok(evals.to_vec())
}

fn logspace(start_exp: f64, stop_exp: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![10.0_f64.powf(start_exp)];
    }
    (0..n)
        .map(|k| {
            let t = k as f64 / (n - 1) as f64;
            10.0_f64.powf(start_exp + t * (stop_exp - start_exp))
        })
        .collect()
}

fn auto_frequency_grid<T: IsAnalytic>(b: &[T], a: &[T], n: usize) -> Result<R1D, ErrorsJSL> {
    if n == 0 {
        return Err(ErrorsJSL::InvalidInputRange("worN must be > 0"));
    }

    let mut scales = Vec::new();
    for root in polynomial_roots(b)?.into_iter().chain(polynomial_roots(a)?.into_iter()) {
        let mag = root.norm();
        if mag.is_finite() && mag > 0.0 {
            scales.push(mag);
        }
    }

    let (w_min, w_max) = if scales.is_empty() {
        (0.1, 10.0)
    } else {
        let min_scale = scales
            .iter()
            .copied()
            .fold(f64::INFINITY, |acc, x| acc.min(x));
        let max_scale = scales
            .iter()
            .copied()
            .fold(0.0_f64, |acc, x| acc.max(x));
        let low = (min_scale / 10.0).max(1e-3);
        let high = (max_scale * 10.0).max(low * 10.0);
        (low, high)
    };

    Ok(Array1::from_vec(logspace(w_min.log10(), w_max.log10(), n)))
}

fn build_frequency_grid<T: IsAnalytic>(
    b: &[T],
    a: &[T],
    wor_n: Option<FreqsWorN>,
) -> Result<R1D, ErrorsJSL> {
    match wor_n.unwrap_or(FreqsWorN::NumPoints(200)) {
        FreqsWorN::NumPoints(n) => auto_frequency_grid(b, a, n),
        FreqsWorN::Frequencies(w) => {
            if w.is_empty() {
                return Err(ErrorsJSL::InvalidInputRange(
                    "frequency array must be non-empty",
                ));
            }
            if w.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return Err(ErrorsJSL::InvalidInputRange(
                    "frequency array values must be finite and >= 0",
                ));
            }
            Ok(Array1::from_vec(w))
        }
    }
}

/// Compute the analog frequency response of a continuous-time filter.
///
/// Arguments:
/// - `b`: numerator polynomial coefficients in descending powers of `s`
/// - `a`: denominator polynomial coefficients in descending powers of `s`
/// - `wor_n`: optional frequency-grid specification. `None` defaults to 200
///   automatically chosen frequencies near pole/zero magnitudes.
pub fn freqs<T: IsAnalytic>(
    b: &[T],
    a: &[T],
    wor_n: Option<FreqsWorN>,
) -> Result<FreqsResult, ErrorsJSL> {
    validate_coefficients(b, a)?;
    let w = build_frequency_grid(b, a, wor_n)?;

    let mut h = Vec::with_capacity(w.len());
    for &omega in &w {
        let s = Complex::new(0.0, omega);
        let numerator = polyval(b, s);
        let denominator = polyval(a, s);
        if denominator.norm_sqr() <= 1e-24 {
            return Err(ErrorsJSL::RuntimeError(
                "frequency response undefined because denominator is zero",
            ));
        }
        h.push(numerator / denominator);
    }

    Ok(FreqsResult {
        w,
        h: Array1::from_vec(h),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close_complex(a: Complex<f64>, b: Complex<f64>, tol: f64) {
        assert!((a - b).norm() < tol, "left={a}, right={b}, tol={tol}");
    }

    #[test]
    fn test_freqs_identity_is_flat() {
        let result = freqs(&[1.0], &[1.0], None).unwrap();
        assert_eq!(result.w.len(), 200);
        assert_eq!(result.h.len(), 200);
        for pair in result.w.iter().zip(result.w.iter().skip(1)) {
            assert!(pair.0 < pair.1);
        }
        for &h in &result.h {
            assert_close_complex(h, Complex::new(1.0, 0.0), 1e-12);
        }
    }

    #[test]
    fn test_freqs_matches_simple_rc_lowpass() {
        let result = freqs(
            &[1.0],
            &[1.0, 1.0],
            Some(FreqsWorN::Frequencies(vec![0.0, 1.0])),
        )
        .unwrap();

        assert_eq!(result.w.to_vec(), vec![0.0, 1.0]);
        assert_close_complex(result.h[0], Complex::new(1.0, 0.0), 1e-12);
        assert_close_complex(result.h[1], Complex::new(0.5, -0.5), 1e-12);
    }

    #[test]
    fn test_freqs_num_points_uses_requested_length() {
        let result = freqs(&[1.0], &[1.0, 1.0], Some(FreqsWorN::NumPoints(17))).unwrap();
        assert_eq!(result.w.len(), 17);
        assert_eq!(result.h.len(), 17);
    }

    #[test]
    fn test_freqs_rejects_negative_explicit_frequency() {
        let err = freqs(&[1.0], &[1.0], Some(FreqsWorN::Frequencies(vec![-1.0]))).unwrap_err();
        assert!(matches!(err, ErrorsJSL::InvalidInputRange(_)));
    }
}
