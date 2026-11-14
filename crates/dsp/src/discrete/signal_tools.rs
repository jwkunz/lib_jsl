//! Signal-processing utilities following `scipy.signal`: filter analysis ([`group_delay`],
//! [`minimum_phase`]), least-squares FIR design ([`firls`]), rate changes ([`decimate`] and the
//! FFT [`resample`]), [`medfilt`], the amplitude [`envelope`], and cross-spectral analysis
//! ([`csd`], [`coherence`]).
//!
//! See the book entry [Signal-processing utilities](https://jwkunz.github.io/lib_jsl/book/spectral/signal-tools.html).

use crate::discrete::filters::iir::chebyshev1::cheby1;
use crate::discrete::filters::iir::butterworth::{ButterOutput, ButterOutputType};
use crate::discrete::filters::iir::sos::{sosfilt, zpk2sos};
use crate::discrete::filters::iir::zero_phase::{Padding, sosfiltfilt};
use crate::discrete::spectral::hilbert::analytic_signal;
use crate::discrete::spectral::welch::{Detrend, SpectralScaling, detrend_in_place};
use crate::continuous::filter::design::IirFilterBandType;
use lib_jsl_core::ErrorsJSL;
use lib_jsl_ffts::{Fft, FftDirection, FftEngine1D, FftOrdering, FftScaleFactor};
use lib_jsl_linalg::Solve;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use std::f64::consts::PI;

type C = Complex<f64>;

fn fft(x: &[C], inverse: bool) -> Result<Vec<C>, ErrorsJSL> {
    let (scale, dir) = if inverse { (FftScaleFactor::N, FftDirection::Inverse) } else { (FftScaleFactor::None, FftDirection::Forward) };
    Fft::with_plan(x.len(), scale, dir, FftOrdering::Standard)?.execute(x)
}

fn fft_real(x: &[f64], n: usize) -> Result<Vec<C>, ErrorsJSL> {
    let mut buf: Vec<C> = x.iter().map(|&v| C::new(v, 0.0)).collect();
    buf.resize(n, C::new(0.0, 0.0));
    fft(&buf, false)
}

// ANCHOR: group_delay
/// The group delay `−dφ/dω` of the filter `b(z)/a(z)` in samples at the normalised angular
/// frequencies `w` (radians per sample) (SciPy's `group_delay`). With `c = b ∗ reverse(conj(a))`,
/// `τ(ω) = Re(Σ k cₖ e^{−iωk} / Σ cₖ e^{−iωk}) − (len(a) − 1)`; at frequencies where the
/// denominator vanishes the delay is set to 0, as in SciPy.
pub fn group_delay(b: &[f64], a: &[f64], w: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
    if b.is_empty() || a.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("b and a must be non-empty"));
    }
    let mut c = vec![0.0; b.len() + a.len() - 1];
    for (i, bi) in b.iter().enumerate() {
        for (j, aj) in a.iter().rev().enumerate() {
            c[i + j] += bi * aj;
        }
    }
    Ok(w
        .iter()
        .map(|&wk| {
            let z = C::from_polar(1.0, -wk);
            let (mut num, mut den, mut zk) = (C::new(0.0, 0.0), C::new(0.0, 0.0), C::new(1.0, 0.0));
            for (k, ck) in c.iter().enumerate() {
                num += zk * ck * k as f64;
                den += zk * ck;
                zk *= z;
            }
            if den.norm() < 10.0 * f64::EPSILON * c.iter().map(|v| v.abs()).sum::<f64>() { 0.0 } else { (num / den).re - (a.len() as f64 - 1.0) }
        })
        .collect())
}
// ANCHOR_END: group_delay

// ANCHOR: minimum_phase
/// A minimum-phase FIR filter from a linear-phase one by the homomorphic method (SciPy's
/// `minimum_phase(h, 'homomorphic')`): the real cepstrum of `|H|` (or of `|H|^{1/2}` with
/// `half`, which preserves the magnitude of a filter that is the square of a minimum-phase one),
/// folded onto positive quefrencies and exponentiated back. Returns `len(h)/2 + len(h) mod 2`
/// coefficients with `half`, otherwise `len(h)`.
pub fn minimum_phase(h: &[f64], half: bool, n_fft: Option<usize>) -> Result<Vec<f64>, ErrorsJSL> {
    let n = h.len();
    if n <= 2 {
        return Err(ErrorsJSL::InvalidInputRange("h must have more than two taps"));
    }
    let n_fft = n_fft.unwrap_or(1 << (2.0 * (n - 1) as f64 / 0.01).log2().ceil() as u32);
    if n_fft < n {
        return Err(ErrorsJSL::InvalidInputRange("n_fft must be at least len(h)"));
    }
    let spectrum = fft_real(h, n_fft)?;
    let mut mag: Vec<f64> = spectrum.iter().map(|z| z.norm()).collect();
    let min_pos = mag.iter().copied().filter(|&v| v > 0.0).fold(f64::INFINITY, f64::min);
    mag.iter_mut().for_each(|v| *v += 1e-7 * min_pos);
    let mut logmag: Vec<C> = mag.iter().map(|v| C::new(v.ln() * if half { 0.5 } else { 1.0 }, 0.0)).collect();
    // Real cepstrum, folded: c₀, 2c₁, …, 2c_{N/2−1}, (c_{N/2}), 0, ….
    logmag = fft(&logmag, true)?;
    let stop = n_fft / 2;
    for (k, v) in logmag.iter_mut().enumerate() {
        let w = if k == 0 {
            1.0
        } else if k < stop {
            2.0
        } else if k == stop && !n_fft.is_multiple_of(2) {
            1.0
        } else {
            0.0
        };
        *v = C::new(v.re * w, 0.0);
    }
    let exp: Vec<C> = fft(&logmag, false)?.iter().map(|z| z.exp()).collect();
    let hmin = fft(&exp, true)?;
    let n_out = if half { n / 2 + n % 2 } else { n };
    Ok(hmin[..n_out].iter().map(|z| z.re).collect())
}
// ANCHOR_END: minimum_phase

// ANCHOR: firls
/// A linear-phase FIR filter of odd length `numtaps` minimising the weighted integrated squared
/// error to a piecewise-linear desired response (SciPy's `firls`). `bands` holds band edges in
/// pairs and `desired` the gains at those edges, as fractions of the Nyquist frequency when `fs`
/// is `None`.
///
/// For the cosine series `A(ω) = Σ aₙ cos(nω)`, the normal equations `Q a = b` have
/// `Q = T + H` (a Toeplitz plus a Hankel matrix of integrals of `cos(nω)` over the bands) and `b`
/// the integrals of the desired response times `cos(nω)`, all in closed form.
pub fn firls(numtaps: usize, bands: &[f64], desired: &[f64], weight: Option<&[f64]>, fs: Option<f64>) -> Result<Vec<f64>, ErrorsJSL> {
    if numtaps.is_multiple_of(2) || numtaps < 1 {
        return Err(ErrorsJSL::InvalidInputRange("numtaps must be odd"));
    }
    if !bands.len().is_multiple_of(2) || bands.len() != desired.len() {
        return Err(ErrorsJSL::InvalidInputRange("bands and desired need one value per band edge"));
    }
    let nyq = 0.5 * fs.unwrap_or(2.0);
    let f: Vec<f64> = bands.iter().map(|v| v / nyq).collect();
    if f.iter().any(|v| !(0.0..=1.0).contains(v)) || f.windows(2).any(|w| w[1] < w[0]) {
        return Err(ErrorsJSL::InvalidInputRange("band edges must be increasing within [0, Nyquist]"));
    }
    let nb = f.len() / 2;
    let ones = vec![1.0; nb];
    let wts = weight.unwrap_or(&ones);
    if wts.len() != nb {
        return Err(ErrorsJSL::IncompatibleArraySizes((wts.len(), nb)));
    }
    let m = (numtaps - 1) / 2;
    let sinc = |x: f64| if x == 0.0 { 1.0 } else { (PI * x).sin() / (PI * x) };
    // q[n] = Σ_bands w (f₂ sinc(f₂ n) − f₁ sinc(f₁ n)).
    let q: Vec<f64> = (0..numtaps).map(|n| (0..nb).map(|i| wts[i] * (f[2 * i + 1] * sinc(f[2 * i + 1] * n as f64) - f[2 * i] * sinc(f[2 * i] * n as f64))).sum()).collect();
    let qm = Array2::from_shape_fn((m + 1, m + 1), |(i, j)| q[i.abs_diff(j)] + q[i + j]);
    let mut b = vec![0.0; m + 1];
    for i in 0..nb {
        let (f1, f2) = (f[2 * i], f[2 * i + 1]);
        let (d1, d2) = (desired[2 * i], desired[2 * i + 1]);
        let slope = (d2 - d1) / (f2 - f1);
        let c = d1 - f1 * slope;
        for (n, bn) in b.iter_mut().enumerate() {
            let nf = n as f64;
            let term = |fe: f64| {
                let mut t = fe * (slope * fe + c) * sinc(fe * nf);
                if n == 0 {
                    t -= slope * fe * fe / 2.0;
                } else {
                    t += slope * (nf * PI * fe).cos() / (PI * nf).powi(2);
                }
                t
            };
            *bn += wts[i] * (term(f2) - term(f1));
        }
    }
    let a = qm.solve(&Array1::from(b)).map_err(|_| ErrorsJSL::RuntimeError("firls: the normal equations are singular"))?;
    let mut h = Vec::with_capacity(numtaps);
    h.extend(a.iter().skip(1).rev());
    h.push(2.0 * a[0]);
    h.extend(a.iter().skip(1));
    Ok(h)
}
// ANCHOR_END: firls

// ANCHOR: decimate
/// Downsampling by an integer factor `q` after anti-alias filtering (SciPy's `decimate` with
/// `ftype='iir'`): an order-8 Chebyshev type I lowpass with 0.05 dB ripple and cutoff `0.8/q` of
/// Nyquist, applied forwards and backwards (`zero_phase`) or forwards only, then every `q`-th
/// sample.
pub fn decimate(x: &[f64], q: usize, zero_phase: bool) -> Result<Vec<f64>, ErrorsJSL> {
    if q < 1 {
        return Err(ErrorsJSL::InvalidInputRange("the factor must be at least 1"));
    }
    let ButterOutput::Zpk(z) = cheby1(8, 0.05, &[0.8 / q as f64], IirFilterBandType::Lowpass, Some(ButterOutputType::Zpk), None)? else {
        return Err(ErrorsJSL::RuntimeError("cheby1 did not return zeros, poles and gain"));
    };
    let sos = zpk2sos(z.z.as_slice().unwrap_or(&[]), z.p.as_slice().unwrap_or(&[]), z.k.re)?;
    let y = if zero_phase { sosfiltfilt(&sos, x, Padding::Odd, None)? } else { sosfilt(&sos, x, None)?.0 };
    Ok(y.into_iter().step_by(q).collect())
}
// ANCHOR_END: decimate

/// Resamples the real signal `x` to `num` samples by the FFT (SciPy's `resample`): the spectrum is
/// truncated or zero-padded, with the unpaired Nyquist bin of an even length united when
/// downsampling and split when upsampling. Exact for band-limited periodic signals.
pub fn resample(x: &[f64], num: usize) -> Result<Vec<f64>, ErrorsJSL> {
    let nx = x.len();
    if nx == 0 || num == 0 {
        return Err(ErrorsJSL::InvalidInputRange("lengths must be positive"));
    }
    let spectrum = fft_real(x, nx)?;
    let m = num.min(nx);
    let m2 = m / 2 + 1;
    // The half spectrum of the output, as irfft(X[:m2], n=num) reads it.
    let mut half = vec![C::new(0.0, 0.0); num / 2 + 1];
    half[..m2.min(num / 2 + 1)].copy_from_slice(&spectrum[..m2.min(num / 2 + 1)]);
    if m.is_multiple_of(2) && num != nx {
        half[m / 2] *= if num < nx { 2.0 } else { 0.5 };
    }
    // Hermitian completion: real DC (and real Nyquist bin for even num).
    let mut y = vec![C::new(0.0, 0.0); num];
    y[0] = C::new(half[0].re, 0.0);
    for k in 1..num.div_ceil(2) {
        y[k] = half[k];
        y[num - k] = half[k].conj();
    }
    if num.is_multiple_of(2) {
        y[num / 2] = C::new(half[num / 2].re, 0.0);
    }
    let scale = num as f64 / nx as f64;
    Ok(fft(&y, true)?.iter().map(|z| z.re * scale).collect())
}

/// A median filter with an odd `kernel_size`, the signal extended by zeros (SciPy's `medfilt`).
pub fn medfilt(x: &[f64], kernel_size: usize) -> Result<Vec<f64>, ErrorsJSL> {
    if kernel_size.is_multiple_of(2) {
        return Err(ErrorsJSL::InvalidInputRange("the kernel size must be odd"));
    }
    let h = kernel_size / 2;
    let n = x.len() as isize;
    let mut window = vec![0.0; kernel_size];
    Ok((0..n)
        .map(|i| {
            for (k, w) in window.iter_mut().enumerate() {
                let j = i + k as isize - h as isize;
                *w = if j < 0 || j >= n { 0.0 } else { x[j as usize] };
            }
            window.sort_by(f64::total_cmp);
            window[h]
        })
        .collect())
}

/// The amplitude envelope `|x + i H{x}|`, the magnitude of the analytic signal (equal to SciPy's
/// `abs(hilbert(x))`).
pub fn envelope(x: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
    Ok(analytic_signal(x)?.iter().map(|z| z.norm()).collect())
}

/// A one-sided cross spectral density estimate.
#[derive(Clone, Debug)]
pub struct CrossSpectrum {
    pub frequencies: Vec<f64>,
    pub values: Vec<C>,
    pub segments: usize,
}

// ANCHOR: csd
/// The cross spectral density `P_xy(f)` by Welch's method (SciPy's `csd`): segments of
/// `window.len()` samples overlapping by `overlap`, detrended and windowed; the average of
/// `conj(X) Y`, scaled as [`welch`](crate::discrete::spectral::welch::welch) and doubled except at
/// DC and an even-length Nyquist bin.
pub fn csd(x: &[f64], y: &[f64], fs: f64, window: &[f64], overlap: usize, detrend: Detrend, scaling: SpectralScaling) -> Result<CrossSpectrum, ErrorsJSL> {
    let m = window.len();
    if x.len() != y.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((x.len(), y.len())));
    }
    if m == 0 || overlap >= m || x.len() < m || !(fs > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("need a positive rate and a window longer than the overlap and shorter than the signal"));
    }
    let step = m - overlap;
    let segments = (x.len() - m) / step + 1;
    let bins = m / 2 + 1;
    let mut acc = vec![C::new(0.0, 0.0); bins];
    let (mut sx, mut sy) = (vec![0.0; m], vec![0.0; m]);
    for s in 0..segments {
        sx.copy_from_slice(&x[s * step..s * step + m]);
        sy.copy_from_slice(&y[s * step..s * step + m]);
        detrend_in_place(&mut sx, detrend);
        detrend_in_place(&mut sy, detrend);
        sx.iter_mut().zip(window).for_each(|(v, w)| *v *= w);
        sy.iter_mut().zip(window).for_each(|(v, w)| *v *= w);
        let (fx, fy) = (fft_real(&sx, m)?, fft_real(&sy, m)?);
        for k in 0..bins {
            acc[k] += fx[k].conj() * fy[k];
        }
    }
    let scale = match scaling {
        SpectralScaling::Density => 1.0 / (fs * window.iter().map(|w| w * w).sum::<f64>()),
        SpectralScaling::Spectrum => 1.0 / window.iter().sum::<f64>().powi(2),
    } / segments as f64;
    let last_doubled = if m.is_multiple_of(2) { bins - 1 } else { bins };
    for (k, v) in acc.iter_mut().enumerate() {
        *v *= scale * if k > 0 && k < last_doubled { 2.0 } else { 1.0 };
    }
    Ok(CrossSpectrum { frequencies: (0..bins).map(|k| k as f64 * fs / m as f64).collect(), values: acc, segments })
}

/// The magnitude-squared coherence `|P_xy|² / (P_xx P_yy)` (SciPy's `coherence`): 1 where `y` is
/// a noise-free linear filtering of `x`, near 0 where they are unrelated.
pub fn coherence(x: &[f64], y: &[f64], fs: f64, window: &[f64], overlap: usize, detrend: Detrend) -> Result<(Vec<f64>, Vec<f64>), ErrorsJSL> {
    let pxy = csd(x, y, fs, window, overlap, detrend, SpectralScaling::Density)?;
    let pxx = csd(x, x, fs, window, overlap, detrend, SpectralScaling::Density)?;
    let pyy = csd(y, y, fs, window, overlap, detrend, SpectralScaling::Density)?;
    let c = (0..pxy.values.len()).map(|k| pxy.values[k].norm_sqr() / (pxx.values[k].re * pyy.values[k].re)).collect();
    Ok((pxy.frequencies, c))
}
// ANCHOR_END: csd

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_delay_of_a_pure_delay_and_a_symmetric_fir() {
        let w: Vec<f64> = (0..10).map(|i| 0.1 + i as f64 * 0.25).collect();
        let gd = group_delay(&[0.0, 0.0, 0.0, 1.0], &[1.0], &w).unwrap();
        assert!(gd.iter().all(|g| (g - 3.0).abs() < 1e-12));
        let gd = group_delay(&[0.2, 0.5, 1.0, 0.5, 0.2], &[1.0], &w).unwrap();
        assert!(gd.iter().all(|g| (g - 2.0).abs() < 1e-9), "{gd:?}");
    }

    #[test]
    fn resample_is_exact_for_band_limited_signals() {
        let n = 32;
        let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin()).collect();
        for num in [20, 48, 64] {
            let y = resample(&x, num).unwrap();
            for (i, v) in y.iter().enumerate() {
                assert!((v - (2.0 * PI * 3.0 * i as f64 / num as f64).sin()).abs() < 1e-12, "num = {num}");
            }
        }
    }

    #[test]
    fn firls_lowpass_is_symmetric_and_matches_scipy_at_dc() {
        let h = firls(31, &[0.0, 0.3, 0.4, 1.0], &[1.0, 1.0, 0.0, 0.0], None, None).unwrap();
        assert_eq!(h.len(), 31);
        assert!((0..15).all(|i| (h[i] - h[30 - i]).abs() < 1e-15));
        // SciPy: Σh = 1.0033195062747546 (the passband ripple at DC).
        assert!((h.iter().sum::<f64>() - 1.0033195062747546).abs() < 1e-12);
    }
}
