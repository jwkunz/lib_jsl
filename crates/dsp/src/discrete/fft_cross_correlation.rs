//! FFT-based cross-correlation, similar to `scipy.signal.correlate(..., method="fft")`.
//!
//! See the book entry [Convolution and correlation](https://jwkunz.github.io/lib_jsl/book/filters/convolution.html#cross-correlation-and-delay-estimation).

use crate::{
    discrete::{
        convolve::ConvolveMode,
        cross_correlation::{cross_correlation_lags, CorrelationMode},
        fft_convolution::fft_convolve,
    },
    prelude::{ErrorsJSL, IsAnalytic},
};

fn map_mode(mode: CorrelationMode) -> ConvolveMode {
    match mode {
        CorrelationMode::Full => ConvolveMode::Full,
        CorrelationMode::Same => ConvolveMode::Same,
        CorrelationMode::Valid => ConvolveMode::Valid,
    }
}

/// Cross-correlation of `in1` with `in2` computed with FFTs: `correlate(a, b) = convolve(a,
/// reverse(conj(b)))`, so `out[k]` measures how well `in2` delayed by the corresponding lag
/// matches `in1` (see [`cross_correlation_lags`] for the lag of each output).
pub fn fft_cross_correlate<T: IsAnalytic>(in1: &[T], in2: &[T], mode: CorrelationMode) -> Result<Vec<T>, ErrorsJSL> {
    if in1.is_empty() || in2.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("Input arrays must be non-empty"));
    }
    let rhs: Vec<T> = in2.iter().rev().map(|x| x.f_conj()).collect();
    fft_convolve(in1, &rhs, map_mode(mode))
}

/// Autocorrelation of `x` with itself (full mode: lags `-(n-1)..=n-1`, peak at lag 0).
pub fn fft_autocorrelate<T: IsAnalytic>(x: &[T]) -> Result<Vec<T>, ErrorsJSL> {
    fft_cross_correlate(x, x, CorrelationMode::Full)
}

/// Finds the lag at which `in1` best matches `in2` (the peak of the magnitude of their full
/// cross-correlation). A positive lag means `in1` is `in2` delayed by that many samples.
/// Returns `(lag, correlation value at the peak)`.
pub fn fft_correlation_peak<T: IsAnalytic>(in1: &[T], in2: &[T]) -> Result<(isize, T), ErrorsJSL> {
    let c = fft_cross_correlate(in1, in2, CorrelationMode::Full)?;
    let lags = cross_correlation_lags(in1.len(), in2.len(), CorrelationMode::Full);
    let (idx, _) = c
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.f_abs().partial_cmp(&b.1.f_abs()).unwrap())
        .expect("non-empty result");
    Ok((lags[idx], c[idx]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::convolve::ConvolveMethod;
    use crate::discrete::cross_correlation::cross_correlate;
    use num_complex::Complex;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    #[test]
    fn matches_the_direct_method_in_every_mode() {
        let mut seed = 21;
        for (n, m) in [(50usize, 8usize), (8, 50), (33, 33)] {
            let a: Vec<f64> = (0..n).map(|_| lcg(&mut seed)).collect();
            let b: Vec<f64> = (0..m).map(|_| lcg(&mut seed)).collect();
            for mode in [CorrelationMode::Full, CorrelationMode::Same, CorrelationMode::Valid] {
                let want = cross_correlate(&a, &b, mode, ConvolveMethod::Direct).unwrap();
                let got = fft_cross_correlate(&a, &b, mode).unwrap();
                assert_eq!(got.len(), want.len());
                assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-11), "{mode:?} {n}x{m}");
            }
        }
    }

    #[test]
    fn complex_correlation_conjugates_the_second_input() {
        let mut seed = 4;
        let a: Vec<Complex<f64>> = (0..40).map(|_| Complex::new(lcg(&mut seed), lcg(&mut seed))).collect();
        let b: Vec<Complex<f64>> = (0..10).map(|_| Complex::new(lcg(&mut seed), lcg(&mut seed))).collect();
        let want = cross_correlate(&a, &b, CorrelationMode::Full, ConvolveMethod::Direct).unwrap();
        let got = fft_cross_correlate(&a, &b, CorrelationMode::Full).unwrap();
        assert!(got.iter().zip(&want).all(|(g, w)| (g - w).norm() < 1e-11));
    }

    #[test]
    fn finds_the_delay_of_a_buried_template() {
        let mut seed = 8;
        let template: Vec<f64> = (0..64).map(|_| lcg(&mut seed)).collect();
        let mut signal: Vec<f64> = (0..1000).map(|_| 0.3 * lcg(&mut seed)).collect();
        for (i, t) in template.iter().enumerate() {
            signal[371 + i] += t;
        }
        let (lag, value) = fft_correlation_peak(&signal, &template).unwrap();
        assert_eq!(lag, 371);
        let energy: f64 = template.iter().map(|t| t * t).sum();
        assert!((value - energy).abs() < 0.2 * energy, "peak {value} vs template energy {energy}");
        // And the other way round: the template is "advanced" by that lag relative to the signal.
        assert_eq!(fft_correlation_peak(&template, &signal).unwrap().0, -371);
    }

    #[test]
    fn autocorrelation_is_symmetric_with_its_peak_at_zero_lag() {
        let mut seed = 31;
        let x: Vec<f64> = (0..60).map(|_| lcg(&mut seed)).collect();
        let r = fft_autocorrelate(&x).unwrap();
        assert_eq!(r.len(), 119);
        let energy: f64 = x.iter().map(|v| v * v).sum();
        assert!((r[59] - energy).abs() < 1e-10);
        for k in 1..59 {
            assert!((r[59 - k] - r[59 + k]).abs() < 1e-10);
            assert!(r[59 + k].abs() <= energy + 1e-10);
        }
    }

    #[test]
    fn rejects_empty_inputs() {
        assert!(fft_cross_correlate::<f64>(&[], &[1.0], CorrelationMode::Full).is_err());
        assert!(fft_correlation_peak::<f64>(&[1.0], &[]).is_err());
    }
}
