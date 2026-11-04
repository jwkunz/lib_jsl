//! The Goertzel algorithm for single DFT bins.
//!
//! See the book entry [Spectra at chosen frequencies: Goertzel, chirp-z and Fourier integrals](https://jwkunz.github.io/lib_jsl/book/spectral/chosen-frequencies.html#goertzels-recursion).

use lib_jsl_core::ErrorsJSL;
use num_complex::Complex;
use std::f64::consts::TAU;

/// The DFT of `x` at one (possibly non-integer) bin `k`, `X(k) = Σ x_n e^{−2πi k n / N}`, by the
/// generalised Goertzel recursion `s_n = x_n + 2cos(ω) s_{n−1} − s_{n−2}` (`ω = 2πk/N`) and a final
/// phase correction. Costs one real multiply-add per sample, so it beats an FFT when only a few
/// bins are needed (tone detection).
pub fn goertzel(x: &[f64], k: f64) -> Result<Complex<f64>, ErrorsJSL> {
    let n = x.len();
    if n == 0 || !k.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("Need a non-empty signal and a finite bin."));
    }
    // ANCHOR: goertzel
    let omega = TAU * k / n as f64;
    let coefficient = 2.0 * omega.cos();
    let (mut s1, mut s2) = (0.0, 0.0);
    for &v in x {
        let s0 = v + coefficient * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    // y_{N−1} = s_{N−1} − e^{−iω} s_{N−2} = e^{iω(N−1)} X(k); undo the phase.
    let y = Complex::new(s1, 0.0) - Complex::from_polar(1.0, -omega) * s2;
    Ok(y * Complex::from_polar(1.0, -omega * (n - 1) as f64))
    // ANCHOR_END: goertzel
}

/// `|X(k)|²` at bin `k` without the final complex rotation:
/// `s_{N−1}² + s_{N−2}² − 2cos(ω) s_{N−1} s_{N−2}`.
pub fn goertzel_power(x: &[f64], k: f64) -> Result<f64, ErrorsJSL> {
    let n = x.len();
    if n == 0 || !k.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("Need a non-empty signal and a finite bin."));
    }
    let coefficient = 2.0 * (TAU * k / n as f64).cos();
    let (mut s1, mut s2) = (0.0, 0.0);
    for &v in x {
        let s0 = v + coefficient * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    Ok(s1 * s1 + s2 * s2 - coefficient * s1 * s2)
}

/// The DFT at frequency `f` (Hz) of `x` sampled at `fs`.
pub fn goertzel_frequency(x: &[f64], f: f64, fs: f64) -> Result<Complex<f64>, ErrorsJSL> {
    goertzel(x, f * x.len() as f64 / fs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dft(x: &[f64], k: f64) -> Complex<f64> {
        x.iter().enumerate().map(|(n, &v)| Complex::from_polar(v, -TAU * k * n as f64 / x.len() as f64)).sum()
    }

    #[test]
    fn matches_the_direct_dft_at_integer_and_fractional_bins() {
        let x: Vec<f64> = (0..137).map(|i| (0.3 * i as f64).sin() + 0.1 * ((i * i) % 13) as f64).collect();
        for k in [0.0, 1.0, 7.0, 21.5, 68.0, 100.25] {
            let g = goertzel(&x, k).unwrap();
            assert!((g - dft(&x, k)).norm() < 1e-9, "k={k}");
            assert!((goertzel_power(&x, k).unwrap() - g.norm_sqr()).abs() < 1e-8 * (1.0 + g.norm_sqr()));
        }
        assert!(goertzel(&[], 1.0).is_err());
    }

    #[test]
    fn detects_dtmf_digits() {
        // Digit 9 = 852 Hz + 1477 Hz at 8 kHz over 205 samples (the classic DTMF block).
        let fs = 8000.0;
        let x: Vec<f64> = (0..205).map(|i| { let t = i as f64 / fs; (TAU * 852.0 * t).sin() + (TAU * 1477.0 * t).sin() }).collect();
        let rows = [697.0, 770.0, 852.0, 941.0];
        let columns = [1209.0, 1336.0, 1477.0];
        let strongest = |freqs: &[f64]| (0..freqs.len()).max_by(|&i, &j| goertzel_frequency(&x, freqs[i], fs).unwrap().norm().total_cmp(&goertzel_frequency(&x, freqs[j], fs).unwrap().norm())).unwrap();
        assert_eq!((rows[strongest(&rows)], columns[strongest(&columns)]), (852.0, 1477.0));
    }
}
