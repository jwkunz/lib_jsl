//! Maximum entropy spectral estimation (Burg's method).
//!
//! Among all spectra consistent with a set of measured autocorrelation lags, the *maximum entropy*
//! one is the all-pole (autoregressive) spectrum `σ² / |A(e^{jω})|²`. Burg's algorithm estimates
//! the AR model directly from the data by minimising the forward and backward prediction error
//! together, stage by stage, without first forming an autocorrelation estimate. Compared with the
//! periodogram it resolves closely spaced sinusoids from short records and never windows the data,
//! at the price of needing a model order (and, for high orders, spurious peaks).

use lib_jsl_core::ErrorsJSL;
use std::f64::consts::TAU;

/// The result of Burg's AR estimation.
#[derive(Clone, Debug, PartialEq)]
pub struct BurgResult {
    /// `A(z)` coefficients `[1, a1, …, ap]`.
    pub coefficients: Vec<f64>,
    /// Reflection coefficients; always of magnitude below one, so the model is stable.
    pub reflection: Vec<f64>,
    /// Estimated variance of the driving white noise (the final prediction error power per sample).
    pub variance: f64,
}

/// Burg's method for an AR model of the given `order` (`1 <= order < x.len()`).
pub fn burg(x: &[f64], order: usize) -> Result<BurgResult, ErrorsJSL> {
    let n = x.len();
    if order == 0 || order >= n {
        return Err(ErrorsJSL::InvalidInputRange("Burg order must satisfy 1 <= order < signal length"));
    }
    let mut variance = x.iter().map(|v| v * v).sum::<f64>() / n as f64;
    if !(variance > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Signal must have nonzero power"));
    }
    let mut f = x.to_vec(); // forward prediction errors
    let mut b = x.to_vec(); // backward prediction errors
    let mut a = vec![1.0];
    let mut reflection = Vec::with_capacity(order);
    for m in 1..=order {
        let (mut num, mut den) = (0.0, 0.0);
        for i in m..n {
            num += f[i] * b[i - 1];
            den += f[i] * f[i] + b[i - 1] * b[i - 1];
        }
        let k = if den > 0.0 { -2.0 * num / den } else { 0.0 };
        // Update the error sequences (iterate downwards so b[i-1] is still the old value).
        for i in (m..n).rev() {
            let (fi, bi) = (f[i], b[i - 1]);
            f[i] = fi + k * bi;
            b[i] = bi + k * fi;
        }
        let mut next = a.clone();
        next.push(0.0);
        for i in 1..=a.len() {
            next[i] += k * a[a.len() - i];
        }
        a = next;
        reflection.push(k);
        variance *= 1.0 - k * k;
    }
    Ok(BurgResult { coefficients: a, reflection, variance })
}

/// Evaluates the all-pole power spectral density `σ² / (fs |A(e^{jω})|²)` at the given
/// frequencies (Hz), as a two-sided density per Hz.
pub fn ar_psd(coefficients: &[f64], variance: f64, frequencies: &[f64], sample_rate: f64) -> Vec<f64> {
    frequencies
        .iter()
        .map(|&f| {
            let w = TAU * f / sample_rate;
            let (mut re, mut im) = (0.0, 0.0);
            for (k, a) in coefficients.iter().enumerate() {
                re += a * (w * k as f64).cos();
                im -= a * (w * k as f64).sin();
            }
            variance / (sample_rate * (re * re + im * im))
        })
        .collect()
}

/// Maximum entropy power spectral density estimate of a real signal from an order-`order` Burg AR
/// model, on `num_points` evenly spaced frequencies from 0 to Nyquist inclusive.
///
/// Returns `(frequencies in Hz, one-sided PSD in units²/Hz)`. The one-sided PSD integrates (over
/// 0 to `sample_rate / 2`) to the signal power.
pub fn maximum_entropy_spectrum(x: &[f64], order: usize, num_points: usize, sample_rate: f64) -> Result<(Vec<f64>, Vec<f64>), ErrorsJSL> {
    if num_points < 2 || !(sample_rate > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Need at least 2 frequency points and a positive sample rate"));
    }
    let model = burg(x, order)?;
    let freqs: Vec<f64> = (0..num_points).map(|i| 0.5 * sample_rate * i as f64 / (num_points - 1) as f64).collect();
    let mut psd = ar_psd(&model.coefficients, model.variance, &freqs, sample_rate);
    for (i, p) in psd.iter_mut().enumerate() {
        if i != 0 && i != num_points - 1 {
            *p *= 2.0; // fold the negative frequencies in
        } else {
            *p *= 1.0;
        }
    }
    Ok((freqs, psd))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::spectral::linear_predictive_coding::lpc_synthesize;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    fn noise(n: usize, seed: &mut u64) -> Vec<f64> {
        (0..n).map(|_| (0..12).map(|_| lcg(seed)).sum::<f64>()).collect()
    }

    #[test]
    fn recovers_ar_coefficients_and_innovation_variance() {
        let mut seed = 5;
        let truth = [1.0, -1.5, 0.9];
        let x = lpc_synthesize(&noise(30_500, &mut seed), &truth)[500..].to_vec();
        let m = burg(&x, 2).unwrap();
        assert!((m.coefficients[1] - truth[1]).abs() < 0.02 && (m.coefficients[2] - truth[2]).abs() < 0.02, "{:?}", m.coefficients);
        assert!((m.variance - 1.0).abs() < 0.1, "variance {}", m.variance);
        assert!(m.reflection.iter().all(|k| k.abs() < 1.0));
    }

    #[test]
    fn resolves_two_close_sinusoids_from_a_short_record() {
        // 64 samples, tones 0.10 and 0.125 cycles/sample (spacing 0.025 is 1.6 DFT bins).
        let mut seed = 9;
        let n = 64;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                (TAU * 0.10 * t).sin() + (TAU * 0.125 * t + 1.0).sin() + 0.05 * lcg(&mut seed)
            })
            .collect();
        let (f, p) = maximum_entropy_spectrum(&x, 12, 2001, 1.0).unwrap();
        let peaks: Vec<f64> = (1..p.len() - 1).filter(|&i| p[i] > p[i - 1] && p[i] > p[i + 1] && p[i] > 0.01 * p.iter().cloned().fold(0.0, f64::max)).map(|i| f[i]).collect();
        assert!(peaks.iter().any(|&q| (q - 0.10).abs() < 0.004), "peaks {peaks:?}");
        assert!(peaks.iter().any(|&q| (q - 0.125).abs() < 0.004), "peaks {peaks:?}");
    }

    #[test]
    fn psd_integrates_to_the_signal_power() {
        let mut seed = 13;
        let x = lpc_synthesize(&noise(20_500, &mut seed), &[1.0, -0.9, 0.4])[500..].to_vec();
        let power = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let fs = 8000.0;
        let (f, p) = maximum_entropy_spectrum(&x, 4, 4001, fs).unwrap();
        let df = f[1] - f[0];
        let integral: f64 = p.windows(2).map(|w| 0.5 * (w[0] + w[1]) * df).sum();
        assert!((integral / power - 1.0).abs() < 0.05, "integral {integral} vs power {power}");
    }

    #[test]
    fn matches_the_true_spectrum_of_an_ar_process() {
        let mut seed = 17;
        let truth = [1.0, -1.2, 0.72];
        let x = lpc_synthesize(&noise(60_500, &mut seed), &truth)[500..].to_vec();
        let m = burg(&x, 2).unwrap();
        let freqs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.45];
        let est = ar_psd(&m.coefficients, m.variance, &freqs, 1.0);
        let exact = ar_psd(&truth, 1.0, &freqs, 1.0);
        for (e, t) in est.iter().zip(&exact) {
            assert!((e / t - 1.0).abs() < 0.1, "{e} vs {t}");
        }
    }

    #[test]
    fn invalid_arguments() {
        assert!(burg(&[1.0, 2.0, 3.0], 0).is_err());
        assert!(burg(&[1.0, 2.0, 3.0], 3).is_err());
        assert!(burg(&[0.0; 10], 2).is_err());
        assert!(maximum_entropy_spectrum(&[1.0, 2.0, 3.0, 2.0, 1.0], 2, 1, 1.0).is_err());
        assert!(maximum_entropy_spectrum(&[1.0, 2.0, 3.0, 2.0, 1.0], 2, 8, 0.0).is_err());
    }
}
