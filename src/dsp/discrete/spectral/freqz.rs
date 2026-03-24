/// Frequency-response utility similar to `scipy.signal.freqz`.
///
/// This implementation computes the complex frequency response of a digital
/// filter described by numerator coefficients `b` and denominator coefficients
/// `a`:
///
/// `H(e^{jw}) = (b[0] + b[1] e^{-jw} + ... + b[M] e^{-jwM}) /`
/// `            (a[0] + a[1] e^{-jw} + ... + a[N] e^{-jwN})`
///
/// The public API follows the most commonly used parts of SciPy's `freqz`:
/// - integer `worN` for an automatically generated grid
/// - explicit frequency arrays
/// - `whole`
/// - `fs`
/// - `include_nyquist`
///
/// Unlike the current SciPy `whole=True` behavior, this implementation returns
/// the full-circle frequency axis in ascending order from `-fs/2` to values
/// just below `fs/2`. This keeps the frequency axis sorted from lowest to
/// highest while still covering the whole unit circle.
use std::f64::consts::PI;

use ndarray::Array1;
use num::Complex;

use crate::prelude::{C1D, ErrorsJSL, IsAnalytic, R1D};

#[derive(Clone, Debug, PartialEq)]
pub struct FreqzResult {
    /// Frequency axis in the same units as `fs`.
    pub w: R1D,
    /// Complex frequency response evaluated at each frequency in `w`.
    pub h: C1D,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FreqzWorN {
    /// Generate a uniform frequency grid with the given number of points.
    NumPoints(usize),
    /// Evaluate at the explicitly provided frequencies, expressed in the same
    /// units as `fs`.
    Frequencies(Vec<f64>),
}

fn validate_fs(fs: f64) -> Result<(), ErrorsJSL> {
    if !fs.is_finite() || fs <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("fs must be finite and > 0"));
    }
    Ok(())
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

fn build_frequency_grid(
    wor_n: Option<FreqzWorN>,
    whole: bool,
    fs: f64,
    include_nyquist: bool,
) -> Result<R1D, ErrorsJSL> {
    match wor_n.unwrap_or(FreqzWorN::NumPoints(512)) {
        FreqzWorN::NumPoints(n) => {
            if n == 0 {
                return Err(ErrorsJSL::InvalidInputRange("worN must be > 0"));
            }

            let values = if whole {
                let step = fs / n as f64;
                (0..n)
                    .map(|k| -0.5 * fs + k as f64 * step)
                    .collect::<Vec<_>>()
            } else if n == 1 {
                vec![0.0]
            } else {
                let stop = 0.5 * fs;
                let denom = if include_nyquist {
                    (n - 1) as f64
                } else {
                    n as f64
                };
                (0..n)
                    .map(|k| stop * k as f64 / denom)
                    .collect::<Vec<_>>()
            };

            Ok(Array1::from_vec(values))
        }
        FreqzWorN::Frequencies(w) => {
            if w.is_empty() {
                return Err(ErrorsJSL::InvalidInputRange(
                    "frequency array must be non-empty",
                ));
            }
            if w.iter().any(|x| !x.is_finite()) {
                return Err(ErrorsJSL::InvalidInputRange(
                    "frequency array values must be finite",
                ));
            }
            Ok(Array1::from_vec(w))
        }
    }
}

fn eval_response_at<T: IsAnalytic>(b: &[T], a: &[T], omega: f64) -> Result<Complex<f64>, ErrorsJSL> {
    let z_inv = Complex::from_polar(1.0, -omega);

    let mut zk = Complex::new(1.0, 0.0);
    let mut num = Complex::new(0.0, 0.0);
    for coeff in b {
        num += coeff.to_complex() * zk;
        zk *= z_inv;
    }

    zk = Complex::new(1.0, 0.0);
    let mut den = Complex::new(0.0, 0.0);
    for coeff in a {
        den += coeff.to_complex() * zk;
        zk *= z_inv;
    }

    if den.norm_sqr() <= 1e-24 {
        return Err(ErrorsJSL::RuntimeError(
            "frequency response undefined because denominator is zero",
        ));
    }

    Ok(num / den)
}

/// Compute the frequency response of a digital filter.
///
/// Arguments:
/// - `b`: numerator coefficients
/// - `a`: optional denominator coefficients; defaults to `[1]` when omitted
/// - `wor_n`: optional frequency-grid specification; defaults to 512 points
/// - `whole`: optional; defaults to `false`. When `true` and `wor_n` is an
///   integer, evaluate the whole unit circle and return the axis in ascending
///   order on `[-fs/2, fs/2)`
/// - `fs`: sampling frequency used for the returned axis units; defaults to
///   `2*pi`, matching SciPy's radians/sample convention
/// - `include_nyquist`: optional; defaults to `true`. Only used for
///   non-`whole` integer grids
///
/// Notes:
/// - If `wor_n` is `FreqzWorN::Frequencies`, the explicit frequencies are used
///   as-is and `whole` is ignored, matching SciPy's behavior.
/// - This implementation does not include SciPy's plotting callback.
pub fn freqz<T: IsAnalytic>(
    b: &[T],
    a: Option<&[T]>,
    wor_n: Option<FreqzWorN>,
    whole: Option<bool>,
    fs: Option<f64>,
    include_nyquist: Option<bool>,
) -> Result<FreqzResult, ErrorsJSL> {
    let default_a = [T::one()];
    let a = a.unwrap_or(&default_a);
    let whole = whole.unwrap_or(false);
    let fs = fs.unwrap_or(2.0 * PI);
    let include_nyquist = include_nyquist.unwrap_or(true);

    validate_fs(fs)?;
    validate_coefficients(b, a)?;

    let w = build_frequency_grid(wor_n, whole, fs, include_nyquist)?;
    let mut h = Vec::with_capacity(w.len());
    for &freq in &w {
        let omega = (2.0 * PI / fs) * freq;
        h.push(eval_response_at(b, a, omega)?);
    }

    Ok(FreqzResult {
        w,
        h: Array1::from_vec(h),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close_complex(a: Complex<f64>, b: Complex<f64>, tol: f64) {
        assert!(
            (a - b).norm() < tol,
            "left={a}, right={b}, tol={tol}"
        );
    }

    #[test]
    fn test_freqz_identity_filter_is_flat() {
        let result = freqz(&[1.0], None, Some(FreqzWorN::NumPoints(8)), None, None, None)
            .unwrap();

        assert_eq!(result.w.len(), 8);
        assert_eq!(result.h.len(), 8);
        for &h in &result.h {
            assert_close_complex(h, Complex::new(1.0, 0.0), 1e-12);
        }
    }

    #[test]
    fn test_freqz_include_nyquist_hits_pi_for_default_fs() {
        let result = freqz(&[1.0], None, Some(FreqzWorN::NumPoints(5)), None, None, None)
            .unwrap();

        assert!((result.w[0] - 0.0).abs() < 1e-12);
        assert!((result.w[result.w.len() - 1] - PI).abs() < 1e-12);
    }

    #[test]
    fn test_freqz_whole_returns_sorted_axis() {
        let fs = 20.0;
        let result = freqz(
            &[1.0, -1.0],
            None,
            Some(FreqzWorN::NumPoints(8)),
            Some(true),
            Some(fs),
            None,
        )
        .unwrap();

        let expected = vec![-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5];
        assert_eq!(result.w.to_vec(), expected);
        for pair in result.w.iter().zip(result.w.iter().skip(1)) {
            assert!(pair.0 < pair.1);
        }
    }

    #[test]
    fn test_freqz_whole_reorders_response_with_sorted_axis() {
        let result = freqz(
            &[1.0, 1.0],
            None,
            Some(FreqzWorN::NumPoints(4)),
            Some(true),
            None,
            None,
        )
        .unwrap();

        assert_eq!(result.w.to_vec(), vec![-PI, -PI / 2.0, 0.0, PI / 2.0]);
        assert_close_complex(result.h[0], Complex::new(0.0, 0.0), 1e-12);
        assert_close_complex(result.h[1], Complex::new(1.0, 1.0), 1e-12);
        assert_close_complex(result.h[2], Complex::new(2.0, 0.0), 1e-12);
        assert_close_complex(result.h[3], Complex::new(1.0, -1.0), 1e-12);
    }

    #[test]
    fn test_freqz_explicit_frequency_array() {
        let freqs = vec![0.0, PI / 2.0, PI];
        let result = freqz(
            &[1.0],
            Some(&[1.0, -0.5]),
            Some(FreqzWorN::Frequencies(freqs.clone())),
            Some(true),
            None,
            None,
        )
        .unwrap();

        assert_eq!(result.w.to_vec(), freqs);
        assert_close_complex(result.h[0], Complex::new(2.0, 0.0), 1e-12);
        assert_close_complex(result.h[1], Complex::new(0.8, -0.4), 1e-12);
        assert_close_complex(result.h[2], Complex::new(2.0 / 3.0, 0.0), 1e-12);
    }
}
