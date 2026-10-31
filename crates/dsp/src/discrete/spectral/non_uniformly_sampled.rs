//! Spectral estimation for non-uniformly sampled data.
//!
//! Ordinary FFT spectra assume evenly spaced samples. When samples fall at irregular times (missed
//! measurements, event-driven sensors, astronomical observations) the usual tools are:
//!
//! - the **Lomb-Scargle periodogram** ([`lomb_scargle`]), which fits a sinusoid by least squares at
//!   each trial frequency. It is equivalent to the classical periodogram for even sampling, copes
//!   with arbitrary gaps, and with irregular times is not subject to the usual Nyquist aliasing, so
//!   frequencies far above the *average* sampling rate can still be detected;
//! - the **non-uniform discrete Fourier transform** ([`non_uniform_dft`]), a direct evaluation of
//!   `Σ y[n] e^{-j2π f t[n]}` at arbitrary frequencies;
//! - a three-parameter **least-squares sinusoid fit** ([`fit_sinusoid`]) giving the amplitude and
//!   phase at one frequency.
//!
//! All routines cost `O(N · F)` (samples times frequencies).

use lib_jsl_core::ErrorsJSL;
use num::Complex;
use std::f64::consts::TAU;

fn check_inputs(t: &[f64], y: &[f64]) -> Result<(), ErrorsJSL> {
    if t.len() != y.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((t.len(), y.len())));
    }
    if t.len() < 3 {
        return Err(ErrorsJSL::InvalidInputRange("Need at least 3 samples"));
    }
    if t.iter().chain(y).any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("Sample times and values must be finite"));
    }
    Ok(())
}

/// The Lomb-Scargle periodogram of samples `y` taken at times `t`, at the given **angular**
/// frequencies (radians per unit time).
///
/// The mean is removed first. With `normalize` the result is the dimensionless power in `[0, 1]`
/// (the fraction of the variance explained by the sinusoid at that frequency, so a noise-free
/// sinusoid scores 1 at its frequency); otherwise it is the unnormalised power, which for a
/// sinusoid of amplitude `A` sampled `N` times is about `A² N / 4`. Zero frequencies give 0.
pub fn lomb_scargle(t: &[f64], y: &[f64], angular_frequencies: &[f64], normalize: bool) -> Result<Vec<f64>, ErrorsJSL> {
    check_inputs(t, y)?;
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let yc: Vec<f64> = y.iter().map(|v| v - mean).collect();
    let total: f64 = yc.iter().map(|v| v * v).sum();
    if angular_frequencies.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Frequencies must be finite and non-negative"));
    }
    Ok(angular_frequencies
        .iter()
        .map(|&w| {
            if w == 0.0 {
                return 0.0;
            }
            let (mut s2, mut c2) = (0.0, 0.0);
            for &ti in t {
                let (s, c) = (2.0 * w * ti).sin_cos();
                s2 += s;
                c2 += c;
            }
            let tau = s2.atan2(c2) / (2.0 * w);
            let (mut yc_c, mut yc_s, mut cc, mut ss) = (0.0, 0.0, 0.0, 0.0);
            for (&ti, &yi) in t.iter().zip(&yc) {
                let (s, c) = (w * (ti - tau)).sin_cos();
                yc_c += yi * c;
                yc_s += yi * s;
                cc += c * c;
                ss += s * s;
            }
            let mut p = 0.5 * (if cc > 0.0 { yc_c * yc_c / cc } else { 0.0 } + if ss > 0.0 { yc_s * yc_s / ss } else { 0.0 });
            if normalize {
                p = if total > 0.0 { p * 2.0 / total } else { 0.0 };
            }
            p
        })
        .collect())
}

/// Lomb-Scargle power at frequencies given in cycles per unit time (Hz when `t` is in seconds).
/// This is the spectrum of irregularly sampled data as a function of ordinary frequency.
pub fn non_uniformly_sampled_spectrum(t: &[f64], y: &[f64], frequencies_hz: &[f64], normalize: bool) -> Result<Vec<f64>, ErrorsJSL> {
    let w: Vec<f64> = frequencies_hz.iter().map(|f| f * TAU).collect();
    lomb_scargle(t, y, &w, normalize)
}

/// A sensible frequency grid (cycles per unit time, excluding zero) for [`lomb_scargle`]:
/// spacing `1 / (oversampling · T)` for a record of duration `T`, extending to `max_frequency`.
pub fn lomb_scargle_frequency_grid(t: &[f64], oversampling: f64, max_frequency: f64) -> Result<Vec<f64>, ErrorsJSL> {
    if t.len() < 2 || !(oversampling > 0.0) || !(max_frequency > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Need >= 2 times, positive oversampling and positive maximum frequency"));
    }
    let span = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - t.iter().cloned().fold(f64::INFINITY, f64::min);
    if !(span > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Sample times must span a nonzero interval"));
    }
    let df = 1.0 / (oversampling * span);
    let n = (max_frequency / df).floor() as usize;
    Ok((1..=n).map(|i| i as f64 * df).collect())
}

/// Non-uniform discrete Fourier transform: `X(f) = Σ y[n] e^{-j2π f t[n]}` at each frequency (in
/// cycles per unit time).
pub fn non_uniform_dft(t: &[f64], y: &[f64], frequencies_hz: &[f64]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
    if t.len() != y.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((t.len(), y.len())));
    }
    if t.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("Need at least 1 sample"));
    }
    Ok(frequencies_hz
        .iter()
        .map(|&f| t.iter().zip(y).map(|(&ti, &yi)| Complex::from_polar(yi, -TAU * f * ti)).sum())
        .collect())
}

/// The result of fitting `offset + amplitude · cos(2π f t + phase)` to the samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SinusoidFit {
    pub amplitude: f64,
    /// Phase in radians, in `(-π, π]`.
    pub phase: f64,
    pub offset: f64,
}

#[allow(clippy::needless_range_loop)]
/// Least-squares fit of a constant plus a sinusoid of known frequency `f` (cycles per unit time).
pub fn fit_sinusoid(t: &[f64], y: &[f64], f: f64) -> Result<SinusoidFit, ErrorsJSL> {
    check_inputs(t, y)?;
    if !(f > 0.0) || !f.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("Frequency must be positive and finite"));
    }
    // Normal equations for y ≈ c0 + c1 cos(wt) + c2 sin(wt).
    let w = TAU * f;
    let mut m = [[0.0f64; 3]; 3];
    let mut rhs = [0.0f64; 3];
    for (&ti, &yi) in t.iter().zip(y) {
        let (s, c) = (w * ti).sin_cos();
        let basis = [1.0, c, s];
        for i in 0..3 {
            rhs[i] += basis[i] * yi;
            for j in 0..3 {
                m[i][j] += basis[i] * basis[j];
            }
        }
    }
    // Gauss-Jordan with partial pivoting on the 3x3 system.
    let mut aug = [[0.0f64; 4]; 3];
    for i in 0..3 {
        aug[i][..3].copy_from_slice(&m[i]);
        aug[i][3] = rhs[i];
    }
    for col in 0..3 {
        let pivot = (col..3).max_by(|&a, &b| aug[a][col].abs().partial_cmp(&aug[b][col].abs()).unwrap()).unwrap();
        if aug[pivot][col].abs() < 1e-12 * (t.len() as f64) {
            return Err(ErrorsJSL::RuntimeError("Sinusoid fit is ill-conditioned at this frequency"));
        }
        aug.swap(col, pivot);
        for r in 0..3 {
            if r != col {
                let factor = aug[r][col] / aug[col][col];
                for k in col..4 {
                    aug[r][k] -= factor * aug[col][k];
                }
            }
        }
    }
    let (c0, c1, c2) = (aug[0][3] / aug[0][0], aug[1][3] / aug[1][1], aug[2][3] / aug[2][2]);
    // c1 cos + c2 sin = A cos(wt + phi)  =>  A cos(phi) = c1, A sin(phi) = -c2.
    Ok(SinusoidFit { amplitude: c1.hypot(c2), phase: (-c2).atan2(c1), offset: c0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Irregular sample times with the given mean spacing.
    fn irregular_times(n: usize, mean_dt: f64, seed: &mut u64) -> Vec<f64> {
        let mut t = 0.0;
        (0..n)
            .map(|_| {
                t += mean_dt * (0.2 + 1.6 * lcg(seed));
                t
            })
            .collect()
    }

    fn argmax(p: &[f64]) -> usize {
        p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    }

    #[test]
    fn evenly_sampled_sinusoid_scores_one_at_its_frequency() {
        let t: Vec<f64> = (0..200).map(|n| n as f64 * 0.01).collect();
        let y: Vec<f64> = t.iter().map(|&ti| 3.0 + 2.0 * (TAU * 12.5 * ti + 0.4).sin()).collect();
        let p = non_uniformly_sampled_spectrum(&t, &y, &[12.5, 10.0, 20.0, 31.0], true).unwrap();
        assert!((p[0] - 1.0).abs() < 1e-9, "{p:?}");
        assert!(p[1] < 0.2 && p[2] < 0.2 && p[3] < 0.2);
        // Unnormalised power is A^2 N / 4 for a full-cycle record.
        let raw = non_uniformly_sampled_spectrum(&t, &y, &[12.5], false).unwrap()[0];
        assert!((raw / (4.0 * 200.0 / 4.0) - 1.0).abs() < 0.05, "raw power {raw}");
    }

    #[test]
    fn recovers_a_frequency_from_irregular_samples() {
        let mut seed = 5;
        let t = irregular_times(300, 0.1, &mut seed);
        let f0 = 1.37;
        let y: Vec<f64> = t.iter().map(|&ti| (TAU * f0 * ti + 1.0).cos() + 0.1 * (lcg(&mut seed) - 0.5)).collect();
        let grid = lomb_scargle_frequency_grid(&t, 20.0, 4.0).unwrap();
        let p = non_uniformly_sampled_spectrum(&t, &y, &grid, true).unwrap();
        let est = grid[argmax(&p)];
        assert!((est - f0).abs() < 0.01, "estimated {est}");
        assert!(p[argmax(&p)] > 0.95);
    }

    #[test]
    fn detects_frequencies_far_above_the_average_sampling_rate() {
        // Average rate 10 samples/unit (nominal Nyquist 5), true frequency 13.2: irregular times
        // break the aliasing that would make uniform sampling report 3.2 or 6.8.
        let mut seed = 8;
        let t = irregular_times(400, 0.1, &mut seed);
        let f0 = 13.2;
        let y: Vec<f64> = t.iter().map(|&ti| (TAU * f0 * ti).sin()).collect();
        let grid = lomb_scargle_frequency_grid(&t, 10.0, 20.0).unwrap();
        let p = non_uniformly_sampled_spectrum(&t, &y, &grid, true).unwrap();
        let est = grid[argmax(&p)];
        assert!((est - f0).abs() < 0.05, "estimated {est}");
    }

    #[test]
    fn sinusoid_fit_recovers_amplitude_phase_and_offset_exactly() {
        let mut seed = 2;
        let t = irregular_times(60, 0.3, &mut seed);
        let y: Vec<f64> = t.iter().map(|&ti| -1.5 + 2.5 * (TAU * 0.8 * ti + 0.9).cos()).collect();
        let fit = fit_sinusoid(&t, &y, 0.8).unwrap();
        assert!((fit.amplitude - 2.5).abs() < 1e-9 && (fit.phase - 0.9).abs() < 1e-9 && (fit.offset + 1.5).abs() < 1e-9, "{fit:?}");
        let neg = fit_sinusoid(&t, &t.iter().map(|&ti| (TAU * 0.8 * ti - 2.0).cos()).collect::<Vec<_>>(), 0.8).unwrap();
        assert!((neg.phase + 2.0).abs() < 1e-9);
    }

    #[allow(clippy::needless_range_loop)]
    #[test]
    fn non_uniform_dft_matches_the_dft_for_even_sampling() {
        let n = 16;
        let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i * 7) % 5) as f64 - 2.0).collect();
        let freqs: Vec<f64> = (0..n).map(|k| k as f64 / n as f64).collect();
        let x = non_uniform_dft(&t, &y, &freqs).unwrap();
        for k in 0..n {
            let want: Complex<f64> = (0..n).map(|i| Complex::from_polar(y[i], -TAU * (i * k) as f64 / n as f64)).sum();
            assert!((x[k] - want).norm() < 1e-9);
        }
        // A single sample at time t0 gives a pure phase ramp.
        let one = non_uniform_dft(&[0.3], &[2.0], &[1.0]).unwrap();
        assert!((one[0] - Complex::from_polar(2.0, -TAU * 0.3)).norm() < 1e-12);
    }

    #[test]
    fn frequency_grid_shape_and_validation() {
        let g = lomb_scargle_frequency_grid(&[0.0, 10.0], 5.0, 1.0).unwrap();
        assert!((g[0] - 0.02).abs() < 1e-12 && g.len() == 50);
        assert!(lomb_scargle_frequency_grid(&[1.0], 5.0, 1.0).is_err());
        assert!(lomb_scargle_frequency_grid(&[1.0, 1.0], 5.0, 1.0).is_err());
        assert!(lomb_scargle_frequency_grid(&[0.0, 1.0], 0.0, 1.0).is_err());
        assert!(lomb_scargle(&[0.0, 1.0], &[1.0, 2.0], &[1.0], true).is_err(), "fewer than 3 samples");
        assert!(lomb_scargle(&[0.0, 1.0, 2.0], &[1.0, 2.0], &[1.0], true).is_err(), "length mismatch");
        assert!(lomb_scargle(&[0.0, 1.0, 2.0], &[1.0, 2.0, f64::NAN], &[1.0], true).is_err());
        assert!(lomb_scargle(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], &[-1.0], true).is_err());
        assert_eq!(lomb_scargle(&[0.0, 1.0, 2.0], &[1.0, 2.0, 4.0], &[0.0], true).unwrap(), vec![0.0]);
        assert!(fit_sinusoid(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], 0.0).is_err());
    }
}
