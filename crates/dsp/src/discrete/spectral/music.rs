//! MUSIC (MUltiple SIgnal Classification) spectral estimation.
//!
//! MUSIC is a subspace method. The covariance matrix of the data has `p` large eigenvalues
//! (signal plus noise) and `m - p` equal small ones (noise only). The eigenvectors of the small
//! eigenvalues span the *noise subspace*, which is orthogonal to the steering vectors `a(θ)` of the
//! true sources. The pseudospectrum
//!
//! `P(θ) = 1 / (a(θ)^H E_n E_n^H a(θ))`
//!
//! therefore has sharp peaks at the source parameters `θ`, with resolution well beyond the
//! periodogram's. The core, [`music_pseudospectrum`], is generic over the steering vector, so the
//! same machinery estimates the frequencies of sinusoids in a time series ([`music`],
//! [`music_real`]) or the arrival angles of plane waves on a uniform linear array
//! ([`music_doa`]).
//!
//! The number of sources must be known (or estimated separately), and for sinusoids in a *real*
//! signal each sinusoid counts as two complex exponentials (at `±f`).

use lib_jsl_core::ErrorsJSL;
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};
use num::Complex;
use std::f64::consts::TAU;

/// Estimates the `m × m` covariance matrix of a time series from its overlapping length-`m`
/// snapshots `[x[i], …, x[i+m-1]]`. With `forward_backward` the estimate is averaged with its
/// conjugate-reversed counterpart, which decorrelates coherent sources and halves the variance.
pub fn sample_covariance(x: &[Complex<f64>], m: usize, forward_backward: bool) -> Result<Array2<Complex<f64>>, ErrorsJSL> {
    if m < 2 || x.len() < m + 1 {
        return Err(ErrorsJSL::InvalidInputRange("Need a window of at least 2 samples and more samples than the window"));
    }
    let k = x.len() - m + 1;
    let mut r = Array2::<Complex<f64>>::zeros((m, m));
    for s in 0..k {
        for i in 0..m {
            for j in 0..m {
                r[(i, j)] += x[s + i] * x[s + j].conj();
            }
        }
    }
    r.mapv_inplace(|v| v / k as f64);
    if forward_backward {
        let jrj = Array2::from_shape_fn((m, m), |(i, j)| r[(m - 1 - i, m - 1 - j)].conj());
        r = (r + jrj) * 0.5;
    }
    Ok(r)
}

/// Covariance matrix `E[x x^H]` of array data given as `sensors × snapshots`.
pub fn covariance_from_snapshots(snapshots: &Array2<Complex<f64>>) -> Result<Array2<Complex<f64>>, ErrorsJSL> {
    let (m, k) = snapshots.dim();
    if m < 2 || k == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Need at least 2 sensors and 1 snapshot"));
    }
    let mut r = Array2::<Complex<f64>>::zeros((m, m));
    for s in 0..k {
        for i in 0..m {
            for j in 0..m {
                r[(i, j)] += snapshots[(i, s)] * snapshots[(j, s)].conj();
            }
        }
    }
    Ok(r.mapv(|v| v / k as f64))
}

/// Largest residual `|R v - λ v|` over the eigenpairs.
fn eigen_residual(r: &Array2<Complex<f64>>, values: &[f64], vectors: &Array2<Complex<f64>>) -> f64 {
    let m = values.len();
    (0..m)
        .map(|c| {
            (0..m)
                .map(|row| ((0..m).map(|k| r[(row, k)] * vectors[(k, c)]).sum::<Complex<f64>>() - vectors[(row, c)] * values[c]).norm_sqr())
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max)
}

/// Eigenvalues (descending) and the matching eigenvectors (columns) of a Hermitian matrix.
///
/// `ndarray-linalg` may hand a row-major array to LAPACK as its transpose, which for a Hermitian
/// matrix is its complex conjugate, returning the conjugated eigenvectors. The result is therefore
/// checked against `R v = λ v` and conjugated if that is what fixes it.
fn hermitian_eigen(r: &Array2<Complex<f64>>) -> Result<(Vec<f64>, Array2<Complex<f64>>), ErrorsJSL> {
    let (vals, vecs) = r.eigh(UPLO::Lower).map_err(|_| ErrorsJSL::RuntimeError("Hermitian eigendecomposition failed"))?;
    let m = vals.len();
    let order: Vec<usize> = (0..m).rev().collect(); // eigh returns ascending order
    let sorted: Vec<f64> = order.iter().map(|&i| vals[i]).collect();
    let mut vectors = Array2::from_shape_fn((m, m), |(r_, c)| vecs[(r_, order[c])]);
    let scale = sorted.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-300);
    if eigen_residual(r, &sorted, &vectors) > 1e-8 * scale {
        vectors.mapv_inplace(|v| v.conj());
        if eigen_residual(r, &sorted, &vectors) > 1e-8 * scale {
            return Err(ErrorsJSL::RuntimeError("Eigendecomposition did not satisfy R v = lambda v"));
        }
    }
    Ok((sorted, vectors))
}

/// The eigenvalues of a Hermitian covariance matrix, largest first.
pub fn covariance_eigenvalues(covariance: &Array2<Complex<f64>>) -> Result<Vec<f64>, ErrorsJSL> {
    Ok(hermitian_eigen(covariance)?.0)
}

/// MUSIC pseudospectrum of a covariance matrix.
///
/// `steering(θ)` returns the length-`m` steering vector for parameter `θ`; the pseudospectrum is
/// evaluated at every value in `grid`. `num_sources` must be less than `m`.
pub fn music_pseudospectrum<F>(covariance: &Array2<Complex<f64>>, num_sources: usize, steering: F, grid: &[f64]) -> Result<Vec<f64>, ErrorsJSL>
where
    F: Fn(f64) -> Vec<Complex<f64>>,
{
    let m = covariance.nrows();
    if covariance.ncols() != m {
        return Err(ErrorsJSL::IncompatibleArraySizes((m, covariance.ncols())));
    }
    if num_sources == 0 || num_sources >= m {
        return Err(ErrorsJSL::InvalidInputRange("MUSIC needs 1 <= num_sources < covariance size"));
    }
    let (_, vectors) = hermitian_eigen(covariance)?;
    let noise = vectors.slice(ndarray::s![.., num_sources..]).to_owned();
    grid.iter()
        .map(|&theta| {
            let a = Array1::from(steering(theta));
            if a.len() != m {
                return Err(ErrorsJSL::IncompatibleArraySizes((a.len(), m)));
            }
            let proj: f64 = (0..noise.ncols()).map(|c| (0..m).map(|r| noise[(r, c)].conj() * a[r]).sum::<Complex<f64>>().norm_sqr()).sum();
            Ok(1.0 / proj.max(f64::MIN_POSITIVE))
        })
        .collect()
}

/// A MUSIC frequency pseudospectrum.
#[derive(Clone, Debug, PartialEq)]
pub struct MusicSpectrum {
    /// Frequency grid in Hz.
    pub frequencies: Vec<f64>,
    /// Pseudospectrum on that grid (a sharpness measure, not a power).
    pub pseudospectrum: Vec<f64>,
    /// Covariance eigenvalues, largest first; a clear gap after `num_sources` confirms the order.
    pub eigenvalues: Vec<f64>,
}

impl MusicSpectrum {
    /// The `count` highest local maxima of the pseudospectrum, as frequencies in Hz sorted
    /// ascending, each refined by a parabolic fit of the log-pseudospectrum around its peak.
    pub fn peak_frequencies(&self, count: usize) -> Vec<f64> {
        let p = &self.pseudospectrum;
        let mut peaks: Vec<usize> = (1..p.len().saturating_sub(1)).filter(|&i| p[i] > p[i - 1] && p[i] >= p[i + 1]).collect();
        peaks.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());
        peaks.truncate(count);
        let mut freqs: Vec<f64> = peaks
            .into_iter()
            .map(|i| {
                let (l, c, r) = (p[i - 1].ln(), p[i].ln(), p[i + 1].ln());
                let denom = l - 2.0 * c + r;
                let delta = if denom.abs() > 1e-300 { 0.5 * (l - r) / denom } else { 0.0 };
                self.frequencies[i] + delta.clamp(-1.0, 1.0) * (self.frequencies[i + 1] - self.frequencies[i])
            })
            .collect();
        freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        freqs
    }
}

fn frequency_steering(m: usize, sample_rate: f64) -> impl Fn(f64) -> Vec<Complex<f64>> {
    move |f| (0..m).map(|k| Complex::from_polar(1.0, TAU * f / sample_rate * k as f64)).collect()
}

/// MUSIC frequency estimation for a complex signal containing `num_sources` complex exponentials.
///
/// `window` is the covariance size `m` (`num_sources < window`, and larger windows give sharper
/// peaks at higher cost). The pseudospectrum is evaluated at `num_points` frequencies spanning
/// `[-sample_rate/2, sample_rate/2)`.
pub fn music(x: &[Complex<f64>], num_sources: usize, window: usize, num_points: usize, sample_rate: f64) -> Result<MusicSpectrum, ErrorsJSL> {
    if num_points < 3 || !(sample_rate > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Need at least 3 grid points and a positive sample rate"));
    }
    let r = sample_covariance(x, window, true)?;
    let frequencies: Vec<f64> = (0..num_points).map(|i| (i as f64 / num_points as f64 - 0.5) * sample_rate).collect();
    let pseudospectrum = music_pseudospectrum(&r, num_sources, frequency_steering(window, sample_rate), &frequencies)?;
    Ok(MusicSpectrum { frequencies, pseudospectrum, eigenvalues: covariance_eigenvalues(&r)? })
}

/// MUSIC frequency estimation for a *real* signal containing `num_sinusoids` real sinusoids (each
/// is two complex exponentials, so the signal subspace has dimension `2 * num_sinusoids`). The
/// grid spans `[0, sample_rate/2]`.
pub fn music_real(x: &[f64], num_sinusoids: usize, window: usize, num_points: usize, sample_rate: f64) -> Result<MusicSpectrum, ErrorsJSL> {
    if num_points < 3 || !(sample_rate > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Need at least 3 grid points and a positive sample rate"));
    }
    let xc: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let r = sample_covariance(&xc, window, true)?;
    let frequencies: Vec<f64> = (0..num_points).map(|i| 0.5 * sample_rate * i as f64 / (num_points - 1) as f64).collect();
    let pseudospectrum = music_pseudospectrum(&r, 2 * num_sinusoids, frequency_steering(window, sample_rate), &frequencies)?;
    Ok(MusicSpectrum { frequencies, pseudospectrum, eigenvalues: covariance_eigenvalues(&r)? })
}

/// MUSIC direction-of-arrival pseudospectrum for a uniform linear array.
///
/// `covariance` is the sensor covariance; `spacing` is the element spacing in wavelengths (0.5 for
/// the usual half-wavelength array); `angles_deg` are the candidate arrival angles measured from
/// broadside. Returns the pseudospectrum at each angle.
pub fn music_doa(covariance: &Array2<Complex<f64>>, num_sources: usize, spacing: f64, angles_deg: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
    let m = covariance.nrows();
    music_pseudospectrum(
        covariance,
        num_sources,
        move |deg| (0..m).map(|k| Complex::from_polar(1.0, TAU * spacing * k as f64 * deg.to_radians().sin())).collect(),
        angles_deg,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    fn gauss(seed: &mut u64) -> f64 {
        (0..12).map(|_| lcg(seed)).sum::<f64>()
    }

    fn cgauss(seed: &mut u64, sigma: f64) -> Complex<f64> {
        Complex::new(gauss(seed), gauss(seed)) * (sigma / 2f64.sqrt())
    }

    #[test]
    fn estimates_complex_exponential_frequencies() {
        let mut seed = 1;
        let freqs = [0.12, -0.31]; // cycles per sample
        let x: Vec<Complex<f64>> = (0..256)
            .map(|n| freqs.iter().map(|f| Complex::from_polar(1.0, TAU * f * n as f64)).sum::<Complex<f64>>() + cgauss(&mut seed, 0.1))
            .collect();
        let spec = music(&x, 2, 24, 4096, 1.0).unwrap();
        let est = spec.peak_frequencies(2);
        assert!((est[0] + 0.31).abs() < 1e-3 && (est[1] - 0.12).abs() < 1e-3, "{est:?}");
        // The eigenvalues show two strong components over a noise floor.
        assert!(spec.eigenvalues[1] > 20.0 * spec.eigenvalues[2], "{:?}", &spec.eigenvalues[..4]);
    }

    #[test]
    fn resolves_two_close_real_sinusoids() {
        let mut seed = 2;
        let fs = 8000.0;
        let x: Vec<f64> = (0..400)
            .map(|n| {
                let t = n as f64 / fs;
                (TAU * 440.0 * t).sin() + (TAU * 460.0 * t + 0.7).sin() + 0.1 * gauss(&mut seed)
            })
            .collect();
        // 20 Hz apart is 1.0 DFT bin of this 400-sample record: the periodogram cannot resolve them.
        let spec = music_real(&x, 2, 60, 4001, fs).unwrap();
        let est = spec.peak_frequencies(2);
        assert!((est[0] - 440.0).abs() < 1.0 && (est[1] - 460.0).abs() < 1.0, "{est:?}");
    }

    #[test]
    fn finds_arrival_angles_on_a_uniform_linear_array() {
        let mut seed = 3;
        let (sensors, snapshots) = (8, 500);
        let angles: [f64; 2] = [-20.0, 35.0];
        let steer = |deg: f64| -> Vec<Complex<f64>> { (0..sensors).map(|k| Complex::from_polar(1.0, std::f64::consts::PI * k as f64 * deg.to_radians().sin())).collect() };
        let data = Array2::from_shape_fn((sensors, snapshots), |_| Complex::new(0.0, 0.0));
        let mut data = data;
        for s in 0..snapshots {
            let amps = [cgauss(&mut seed, 1.0), cgauss(&mut seed, 1.0)];
            for (a, &deg) in amps.iter().zip(&angles) {
                for (k, sv) in steer(deg).iter().enumerate() {
                    data[(k, s)] += a * sv;
                }
            }
            for k in 0..sensors {
                data[(k, s)] += cgauss(&mut seed, 0.1);
            }
        }
        let r = covariance_from_snapshots(&data).unwrap();
        let grid: Vec<f64> = (0..3601).map(|i| -90.0 + 0.05 * i as f64).collect();
        let p = music_doa(&r, 2, 0.5, &grid).unwrap();
        let spec = MusicSpectrum { frequencies: grid, pseudospectrum: p, eigenvalues: covariance_eigenvalues(&r).unwrap() };
        let est = spec.peak_frequencies(2);
        assert!((est[0] + 20.0).abs() < 0.5 && (est[1] - 35.0).abs() < 0.5, "{est:?}");
        assert!(spec.eigenvalues[1] > 10.0 * spec.eigenvalues[2]);
    }

    #[test]
    fn forward_backward_covariance_is_hermitian_and_persymmetric() {
        let mut seed = 4;
        let x: Vec<Complex<f64>> = (0..60).map(|_| cgauss(&mut seed, 1.0)).collect();
        let r = sample_covariance(&x, 6, true).unwrap();
        for i in 0..6 {
            for j in 0..6 {
                assert!((r[(i, j)] - r[(j, i)].conj()).norm() < 1e-12, "Hermitian");
                assert!((r[(i, j)] - r[(5 - j, 5 - i)]).norm() < 1e-12, "persymmetric");
            }
        }
        let plain = sample_covariance(&x, 6, false).unwrap();
        assert!(plain.diag().iter().all(|d| d.im.abs() < 1e-12 && d.re > 0.0));
    }

    #[test]
    fn a_noise_free_exponential_has_rank_one() {
        let x: Vec<Complex<f64>> = (0..100).map(|n| Complex::from_polar(2.0, TAU * 0.2 * n as f64)).collect();
        let ev = covariance_eigenvalues(&sample_covariance(&x, 8, false).unwrap()).unwrap();
        assert!((ev[0] - 8.0 * 4.0).abs() < 1e-8, "{ev:?}");
        assert!(ev[1].abs() < 1e-8);
    }

    #[test]
    fn invalid_arguments() {
        let x = vec![Complex::new(1.0, 0.0); 20];
        assert!(music(&x, 4, 4, 100, 1.0).is_err(), "sources must be fewer than the window");
        assert!(music(&x, 0, 4, 100, 1.0).is_err());
        assert!(music(&x, 1, 25, 100, 1.0).is_err(), "window longer than the data");
        assert!(music(&x, 1, 4, 2, 1.0).is_err());
        assert!(music_real(&[0.0; 20], 1, 6, 100, 0.0).is_err());
        assert!(sample_covariance(&x, 1, false).is_err());
        let r = Array2::<Complex<f64>>::eye(4);
        assert!(music_pseudospectrum(&r, 1, |_| vec![Complex::new(1.0, 0.0); 3], &[0.0]).is_err(), "wrong steering length");
    }
}
