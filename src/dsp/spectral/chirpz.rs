/// Efficient chirp-z transform (CZT) for one-dimensional analytic data.
///
/// The chirp-z transform evaluates the z-transform of a finite sequence along
/// a geometric contour in the complex plane:
///
/// `X[k] = sum_{n=0}^{N-1} x[n] * A^{-n} * W^{n k}`
///
/// for `k = 0..M-1`, where:
/// - `N` is the input length
/// - `M` is the number of output points
/// - `A` is the complex starting point on the contour
/// - `W` is the complex step ratio between successive contour points
///
/// This implementation uses the Bluestein/Rabiner reformulation so the CZT is
/// computed via FFT-based convolution instead of a direct `O(N*M)` sum.
use std::f64::consts::PI;

use num::Complex;

use crate::{
    ffts::{
        best_fft::BestFft,
        fft_engine_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
    },
    prelude::{C1D, ErrorsJSL, IsAnalytic},
};

fn complex_powf(base: Complex<f64>, exponent: f64) -> Result<Complex<f64>, ErrorsJSL> {
    if !base.re.is_finite() || !base.im.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange(
            "complex parameter values must be finite",
        ));
    }
    if base == Complex::new(0.0, 0.0) {
        return Err(ErrorsJSL::InvalidInputRange(
            "complex contour parameters must be non-zero",
        ));
    }
    Ok((base.ln() * exponent).exp())
}

/// Compute the chirp-z transform of a slice of analytic data.
///
/// Arguments:
/// - `data_in`: input sequence
/// - `m`: optional output length; defaults to `data_in.len()`
/// - `w`: optional complex ratio between successive contour samples; defaults
///   to `exp(-j*2*pi/m)`, which recovers the length-`m` DFT contour
/// - `a`: optional complex starting point on the contour; defaults to `1 + 0j`
///
/// Returns:
/// - A complex-valued vector of length `m`
///
/// Notes:
/// - When `m == data_in.len()`, `a == 1`, and `w == exp(-j*2*pi/m)`, this
///   computes the same values as the standard DFT.
/// - The input may be real or complex because the function accepts any type
///   implementing [`IsAnalytic`].
pub fn chirpz<T: IsAnalytic>(
    data_in: &[T],
    m: Option<usize>,
    w: Option<Complex<f64>>,
    a: Option<Complex<f64>>,
) -> Result<C1D, ErrorsJSL> {
    if data_in.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("data_in must be non-empty"));
    }

    let n = data_in.len();
    let m = m.unwrap_or(n);
    if m == 0 {
        return Err(ErrorsJSL::InvalidInputRange("m must be > 0"));
    }

    let a = a.unwrap_or(Complex::new(1.0, 0.0));
    if a == Complex::new(0.0, 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("a must be non-zero"));
    }

    let w = w.unwrap_or_else(|| {
        let theta = -2.0 * PI / m as f64;
        Complex::new(theta.cos(), theta.sin())
    });
    if w == Complex::new(0.0, 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("w must be non-zero"));
    }

    let g_len = m + n - 1;
    let conv_len = n + g_len - 1;
    let fft_len = conv_len.next_power_of_two();

    let mut y = vec![Complex::new(0.0, 0.0); fft_len];
    for idx in 0..n {
        let idxf = idx as f64;
        let a_term = complex_powf(a, -idxf)?;
        let w_term = complex_powf(w, 0.5 * idxf * idxf)?;
        y[idx] = data_in[idx].to_complex() * a_term * w_term;
    }

    let mut v = vec![Complex::new(0.0, 0.0); fft_len];
    for q in (-(n as isize) + 1)..=(m as isize - 1) {
        let qf = q as f64;
        v[(q + (n as isize - 1)) as usize] = complex_powf(w, -0.5 * qf * qf)?;
    }

    let mut fft = BestFft::new();
    fft.plan(
        fft_len,
        FftScaleFactor::None,
        FftDirection::Forward,
        FftOrdering::Standard,
    )?;
    let mut ifft = BestFft::new();
    ifft.plan(
        fft_len,
        FftScaleFactor::N,
        FftDirection::Inverse,
        FftOrdering::Standard,
    )?;

    let y_fft = fft.execute(&y)?;
    let v_fft = fft.execute(&v)?;
    let prod = y_fft
        .iter()
        .zip(v_fft.iter())
        .map(|(a, b)| *a * *b)
        .collect::<Vec<_>>();
    let conv = ifft.execute(&prod)?;

    let mut out = Vec::with_capacity(m);
    for k in 0..m {
        let kf = k as f64;
        let w_term = complex_powf(w, 0.5 * kf * kf)?;
        out.push(conv[k + n - 1] * w_term);
    }

    Ok(C1D::from_vec(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct_chirpz<T: IsAnalytic>(
        data_in: &[T],
        m: usize,
        w: Complex<f64>,
        a: Complex<f64>,
    ) -> Vec<Complex<f64>> {
        let mut out = Vec::with_capacity(m);
        for k in 0..m {
            let mut acc = Complex::new(0.0, 0.0);
            for (n, x) in data_in.iter().enumerate() {
                let term = complex_powf(a, -(n as f64)).unwrap()
                    * complex_powf(w, (n * k) as f64).unwrap();
                acc += x.to_complex() * term;
            }
            out.push(acc);
        }
        out
    }

    fn assert_close(actual: Complex<f64>, expected: Complex<f64>, tol: f64) {
        assert!(
            (actual - expected).norm() < tol,
            "actual={actual}, expected={expected}, tol={tol}"
        );
    }

    #[test]
    fn test_chirpz_matches_direct_dft_default_settings() {
        let x = [1.0, -0.5, 0.25, 2.0, -1.0];
        let actual = chirpz(&x, None, None, None).unwrap();
        let m = x.len();
        let theta = -2.0 * PI / m as f64;
        let w = Complex::new(theta.cos(), theta.sin());
        let expected = direct_chirpz(&x, m, w, Complex::new(1.0, 0.0));

        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_close(*a, *e, 1e-9);
        }
    }

    #[test]
    fn test_chirpz_matches_direct_custom_spiral() {
        let x = [
            Complex::new(1.0, 0.5),
            Complex::new(-0.25, 1.0),
            Complex::new(0.75, -0.5),
            Complex::new(0.1, 0.2),
        ];
        let m = 6;
        let radius = 0.93_f64;
        let angle = -0.37_f64;
        let w = Complex::new(radius * angle.cos(), radius * angle.sin());
        let a = Complex::new(0.85, 0.15);

        let actual = chirpz(&x, Some(m), Some(w), Some(a)).unwrap();
        let expected = direct_chirpz(&x, m, w, a);

        assert_eq!(actual.len(), m);
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_close(*a, *e, 1e-8);
        }
    }

    #[test]
    fn test_chirpz_rejects_empty_input() {
        let err = chirpz::<f64>(&[], None, None, None).unwrap_err();
        assert!(matches!(err, ErrorsJSL::InvalidInputRange(_)));
    }
}
