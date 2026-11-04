//! Half-band filters and factor-of-two decimation and interpolation.
//!
//! A half-band low-pass has its cutoff at a quarter of the sample rate and a response symmetric
//! about it, `H(f) + H(½ − f) = 1`. Its impulse response is then zero at every even offset from
//! the centre except the centre itself (`½`), so nearly half the taps vanish, and a decimator
//! or interpolator by two needs only about `N/4` multiplies per output.
//!
//! See the book entry [Resampling: polyphase, half-band and CIC filters](https://jwkunz.github.io/lib_jsl/book/spectral/resampling.html#half-band-filters).

use crate::discrete::windows::kaiser::kaiser;
use lib_jsl_core::ErrorsJSL;
use std::f64::consts::PI;

/// A half-band low-pass of `num_taps = 4k + 3` taps by the Kaiser-windowed sinc
/// `h[n] = ½ sinc(n/2) w[n]` (centred). Larger `beta` gives more stopband attenuation and a wider
/// transition (β ≈ 0.1102 (A − 8.7) for an attenuation of A dB).
pub fn halfband_lowpass(num_taps: usize, beta: f64) -> Result<Vec<f64>, ErrorsJSL> {
    if num_taps < 3 || num_taps % 4 != 3 {
        return Err(ErrorsJSL::InvalidInputRange("A half-band filter needs 4k + 3 taps."));
    }
    let window = kaiser(num_taps, beta, true);
    let centre = (num_taps / 2) as isize;
    Ok((0..num_taps)
        .map(|i| {
            let n = i as isize - centre;
            let value = if n == 0 {
                0.5
            } else if n % 2 == 0 {
                0.0
            } else {
                (PI * n as f64 / 2.0).sin() / (PI * n as f64)
            };
            value * window[i]
        })
        .collect())
}

/// Streaming decimation by two with a half-band filter: computes only every other output and
/// skips the zero taps.
#[derive(Debug, Clone)]
pub struct HalfBandDecimator {
    /// The non-zero odd-offset taps (offsets −c+1, −c+3, …).
    odd_taps: Vec<(usize, f64)>,
    centre: usize,
    centre_tap: f64,
    history: Vec<f64>,
    parity: bool,
}

impl HalfBandDecimator {
    pub fn new(taps: &[f64]) -> Result<Self, ErrorsJSL> {
        if taps.len() < 3 || taps.len() % 4 != 3 {
            return Err(ErrorsJSL::InvalidInputRange("A half-band filter needs 4k + 3 taps."));
        }
        let centre = taps.len() / 2;
        let odd_taps = taps.iter().enumerate().filter(|&(i, _)| (i as isize - centre as isize) % 2 != 0).map(|(i, &t)| (i, t)).collect();
        Ok(HalfBandDecimator { odd_taps, centre, centre_tap: taps[centre], history: vec![0.0; taps.len()], parity: false })
    }

    /// Consumes input samples and returns one output per two inputs (`y[m] = (h ∗ x)[2m + 1]`,
    /// i.e. the full-rate filter output at odd indices).
    pub fn process(&mut self, x: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(x.len() / 2 + 1);
        for &v in x {
            self.history.rotate_right(1);
            self.history[0] = v;
            self.parity = !self.parity;
            if !self.parity {
                // history[k] = x[n − k]; the full-rate output is Σ h[k] x[n − k].
                let mut y = self.centre_tap * self.history[self.centre];
                for &(k, t) in &self.odd_taps {
                    y += t * self.history[k];
                }
                out.push(y);
            }
        }
        out
    }
}

/// Streaming interpolation by two with a half-band filter (gain two to preserve the level):
/// the even phase is the delayed input times `2·h_c`, the odd phase uses only the odd taps.
#[derive(Debug, Clone)]
pub struct HalfBandInterpolator {
    taps: Vec<f64>,
    history: Vec<f64>,
}

impl HalfBandInterpolator {
    pub fn new(taps: &[f64]) -> Result<Self, ErrorsJSL> {
        if taps.len() < 3 || taps.len() % 4 != 3 {
            return Err(ErrorsJSL::InvalidInputRange("A half-band filter needs 4k + 3 taps."));
        }
        Ok(HalfBandInterpolator { taps: taps.to_vec(), history: vec![0.0; taps.len() / 2 + 1] })
    }

    /// Returns two outputs per input: the full-rate filtered, zero-stuffed signal times two.
    pub fn process(&mut self, x: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(2 * x.len());
        for &v in x {
            self.history.rotate_right(1);
            self.history[0] = v;
            // Zero-stuffed u[2j] = x[j]; output at 2j uses taps with even index, at 2j+1 odd ones.
            for phase in 0..2 {
                let mut y = 0.0;
                // Output 2j + phase sees x[j − (k − phase)/2] through every tap k ≡ phase (mod 2).
                for (k, &t) in self.taps.iter().enumerate().skip(phase).step_by(2) {
                    if t != 0.0 {
                        y += t * self.history[(k - phase) / 2];
                    }
                }
                out.push(2.0 * y);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(h: &[f64], f: f64) -> f64 {
        let (re, im) = h.iter().enumerate().fold((0.0, 0.0), |(r, i), (n, &t)| (r + t * (2.0 * PI * f * n as f64).cos(), i - t * (2.0 * PI * f * n as f64).sin()));
        (re * re + im * im).sqrt()
    }

    fn convolve(x: &[f64], h: &[f64]) -> Vec<f64> {
        (0..x.len()).map(|n| (0..h.len()).filter(|&k| k <= n).map(|k| h[k] * x[n - k]).sum()).collect()
    }

    #[test]
    fn design_has_the_half_band_structure() {
        let h = halfband_lowpass(31, 8.0).unwrap();
        let c = h.len() / 2;
        assert_eq!(h[c], 0.5);
        assert!((0..h.len()).filter(|&i| i != c && (i as isize - c as isize) % 2 == 0).all(|i| h[i] == 0.0));
        assert!((response(&h, 0.25) - 0.5).abs() < 1e-12);
        for f in [0.02, 0.1, 0.17] {
            assert!((response(&h, f) + response(&h, 0.5 - f) - 1.0).abs() < 1e-3, "f={f}");
        }
        assert!(response(&h, 0.4) < 1e-3 && (response(&h, 0.05) - 1.0).abs() < 1e-3);
        assert!(halfband_lowpass(29, 8.0).is_err());
    }

    #[test]
    fn decimator_and_interpolator_match_full_rate_filtering() {
        let h = halfband_lowpass(23, 7.0).unwrap();
        let x: Vec<f64> = (0..300).map(|i| (0.05 * i as f64).sin() + 0.3 * ((i * 13) % 7) as f64).collect();
        let full = convolve(&x, &h);
        let mut decimator = HalfBandDecimator::new(&h).unwrap();
        let mut y = decimator.process(&x[..101]);
        y.extend(decimator.process(&x[101..]));
        let reference: Vec<f64> = full.iter().skip(1).step_by(2).copied().collect();
        assert_eq!(y.len(), reference.len());
        assert!(y.iter().zip(&reference).all(|(a, b)| (a - b).abs() < 1e-12));
        let up = HalfBandInterpolator::new(&h).unwrap().process(&x[..50]);
        let stuffed: Vec<f64> = x[..50].iter().flat_map(|&v| [v, 0.0]).collect();
        let reference: Vec<f64> = convolve(&stuffed, &h).iter().map(|v| 2.0 * v).collect();
        assert!(up.iter().zip(&reference).all(|(a, b)| (a - b).abs() < 1e-12));
        assert!(HalfBandDecimator::new(&h[..5]).is_err());
    }
}
