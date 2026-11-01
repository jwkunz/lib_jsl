//! FFT-based convolution, similar to `scipy.signal.fftconvolve`.
//!
//! Convolution in the time domain is multiplication in the frequency domain, so a length-`N`
//! signal convolved with a length-`M` kernel costs `O((N + M) log(N + M))` instead of `O(N M)`.
//! The 1D routine works for any [`IsAnalytic`] sample type (real or complex); the 2D routine works
//! on real images and is the basis for frequency-domain image filtering.

use crate::{
    discrete::convolve::{apply_mode, fft_full, ConvolveMode},
    ffts::{
        Fft,
        fft_engine_trait::{FftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
    },
    prelude::{ErrorsJSL, IsAnalytic},
};
use ndarray::Array2;
use num_complex::Complex;

/// 1D convolution of `in1` and `in2` computed with FFTs.
///
/// `mode` selects the output extent exactly as for [`convolve`](crate::discrete::convolve::convolve)
/// (`Full`, `Same` = the size of `in1`, `Valid`). The result matches direct convolution to rounding
/// error.
pub fn fft_convolve<T: IsAnalytic>(in1: &[T], in2: &[T], mode: ConvolveMode) -> Result<Vec<T>, ErrorsJSL> {
    if in1.is_empty() || in2.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("Input arrays must be non-empty"));
    }
    let full = fft_full(in1, in2)?;
    Ok(apply_mode(full, in1.len(), in2.len(), mode))
}

/// Transforms every row, then every column, of a complex matrix whose dimensions are powers of two.
pub(crate) fn fft2d(data: &Array2<Complex<f64>>, inverse: bool) -> Result<Array2<Complex<f64>>, ErrorsJSL> {
    let (rows, cols) = data.dim();
    let (direction, scale) = if inverse { (FftDirection::Inverse, FftScaleFactor::N) } else { (FftDirection::Forward, FftScaleFactor::None) };
    let mut row_fft = Fft::new();
    row_fft.plan(cols, scale, direction, FftOrdering::Standard)?;
    let mut col_fft = Fft::new();
    col_fft.plan(rows, scale, direction, FftOrdering::Standard)?;
    let mut tmp = Array2::<Complex<f64>>::zeros((rows, cols));
    for r in 0..rows {
        let row: Vec<Complex<f64>> = data.row(r).to_vec();
        for (c, v) in row_fft.execute(&row)?.into_iter().enumerate() {
            tmp[(r, c)] = v;
        }
    }
    let mut out = Array2::<Complex<f64>>::zeros((rows, cols));
    for c in 0..cols {
        let col: Vec<Complex<f64>> = tmp.column(c).to_vec();
        for (r, v) in col_fft.execute(&col)?.into_iter().enumerate() {
            out[(r, c)] = v;
        }
    }
    Ok(out)
}

/// 2D convolution of real images computed with FFTs.
///
/// `Full` has shape `(r1 + r2 - 1, c1 + c2 - 1)`; `Same` has the shape of `in1`, centred; `Valid`
/// has shape `(r1 - r2 + 1, c1 - c2 + 1)` and requires `in1` to be at least as large as `in2`
/// in both dimensions.
pub fn fft_convolve_2d(in1: &Array2<f64>, in2: &Array2<f64>, mode: ConvolveMode) -> Result<Array2<f64>, ErrorsJSL> {
    let ((r1, c1), (r2, c2)) = (in1.dim(), in2.dim());
    if r1 == 0 || c1 == 0 || r2 == 0 || c2 == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Input arrays must be non-empty"));
    }
    let (fr, fc) = (r1 + r2 - 1, c1 + c2 - 1);
    let (pr, pc) = (fr.next_power_of_two(), fc.next_power_of_two());
    let pad = |a: &Array2<f64>| {
        let mut p = Array2::<Complex<f64>>::zeros((pr, pc));
        for ((r, c), &v) in a.indexed_iter() {
            p[(r, c)] = Complex::new(v, 0.0);
        }
        p
    };
    let fa = fft2d(&pad(in1), false)?;
    let fb = fft2d(&pad(in2), false)?;
    let prod = &fa * &fb;
    let spatial = fft2d(&prod, true)?;
    let full = Array2::from_shape_fn((fr, fc), |(r, c)| spatial[(r, c)].re);
    match mode {
        ConvolveMode::Full => Ok(full),
        ConvolveMode::Same => {
            let (r0, c0) = ((r2 - 1) / 2, (c2 - 1) / 2);
            Ok(Array2::from_shape_fn((r1, c1), |(r, c)| full[(r + r0, c + c0)]))
        }
        ConvolveMode::Valid => {
            if r1 < r2 || c1 < c2 {
                return Err(ErrorsJSL::InvalidInputRange("Valid mode requires in1 to be at least as large as in2 in both dimensions"));
            }
            Ok(Array2::from_shape_fn((r1 - r2 + 1, c1 - c2 + 1), |(r, c)| full[(r + r2 - 1, c + c2 - 1)]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::convolve::{convolve, ConvolveMethod};

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    #[test]
    fn matches_direct_convolution_in_every_mode() {
        let mut seed = 5;
        for (n, m) in [(1usize, 1usize), (7, 3), (3, 7), (100, 31), (257, 64), (64, 257)] {
            let a: Vec<f64> = (0..n).map(|_| lcg(&mut seed)).collect();
            let b: Vec<f64> = (0..m).map(|_| lcg(&mut seed)).collect();
            for mode in [ConvolveMode::Full, ConvolveMode::Same, ConvolveMode::Valid] {
                let want = convolve(&a, &b, mode, ConvolveMethod::Direct).unwrap();
                let got = fft_convolve(&a, &b, mode).unwrap();
                assert_eq!(got.len(), want.len(), "{mode:?} {n}x{m}");
                for (g, w) in got.iter().zip(&want) {
                    assert!((g - w).abs() < 1e-11, "{mode:?} {n}x{m}: {g} vs {w}");
                }
            }
        }
    }

    #[test]
    fn complex_signals_and_identity_kernel() {
        let mut seed = 9;
        let a: Vec<Complex<f64>> = (0..50).map(|_| Complex::new(lcg(&mut seed), lcg(&mut seed))).collect();
        let delta = vec![Complex::new(1.0, 0.0)];
        let y = fft_convolve(&a, &delta, ConvolveMode::Full).unwrap();
        for (g, w) in y.iter().zip(&a) {
            assert!((g - w).norm() < 1e-12);
        }
        let b: Vec<Complex<f64>> = (0..9).map(|_| Complex::new(lcg(&mut seed), lcg(&mut seed))).collect();
        let want = convolve(&a, &b, ConvolveMode::Full, ConvolveMethod::Direct).unwrap();
        let got = fft_convolve(&a, &b, ConvolveMode::Full).unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).norm() < 1e-11);
        }
    }

    #[test]
    fn rejects_empty_inputs() {
        assert!(fft_convolve::<f64>(&[], &[1.0], ConvolveMode::Full).is_err());
        assert!(fft_convolve::<f64>(&[1.0], &[], ConvolveMode::Full).is_err());
    }

    fn direct_2d(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let ((r1, c1), (r2, c2)) = (a.dim(), b.dim());
        let mut out = Array2::<f64>::zeros((r1 + r2 - 1, c1 + c2 - 1));
        for ((i, j), &x) in a.indexed_iter() {
            for ((k, l), &h) in b.indexed_iter() {
                out[(i + k, j + l)] += x * h;
            }
        }
        out
    }

    #[test]
    fn two_dimensional_matches_direct() {
        let mut seed = 77;
        let a = Array2::from_shape_fn((13, 9), |_| lcg(&mut seed));
        let b = Array2::from_shape_fn((4, 5), |_| lcg(&mut seed));
        let want = direct_2d(&a, &b);
        let got = fft_convolve_2d(&a, &b, ConvolveMode::Full).unwrap();
        assert_eq!(got.dim(), want.dim());
        assert!(got.iter().zip(want.iter()).all(|(g, w)| (g - w).abs() < 1e-11));
        let same = fft_convolve_2d(&a, &b, ConvolveMode::Same).unwrap();
        assert_eq!(same.dim(), a.dim());
        for r in 0..13 {
            for c in 0..9 {
                assert!((same[(r, c)] - want[(r + 1, c + 2)]).abs() < 1e-11);
            }
        }
        let valid = fft_convolve_2d(&a, &b, ConvolveMode::Valid).unwrap();
        assert_eq!(valid.dim(), (10, 5));
        assert!((valid[(0, 0)] - want[(3, 4)]).abs() < 1e-11);
        assert!(fft_convolve_2d(&b, &a, ConvolveMode::Valid).is_err());
    }

    #[test]
    fn fft2d_round_trip() {
        let mut seed = 3;
        let x = Array2::from_shape_fn((8, 16), |_| Complex::new(lcg(&mut seed), lcg(&mut seed)));
        let back = fft2d(&fft2d(&x, false).unwrap(), true).unwrap();
        assert!(x.iter().zip(back.iter()).all(|(a, b)| (a - b).norm() < 1e-12));
        // A single impulse transforms to a flat spectrum.
        let mut imp = Array2::<Complex<f64>>::zeros((4, 4));
        imp[(0, 0)] = Complex::new(1.0, 0.0);
        assert!(fft2d(&imp, false).unwrap().iter().all(|v| (v - Complex::new(1.0, 0.0)).norm() < 1e-12));
    }
}
