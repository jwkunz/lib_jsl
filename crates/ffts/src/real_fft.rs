//! FFTs of real-valued signals.

use crate::fft::Fft;
use crate::fft_engine_trait::{FftDirection, FftEngine1D, FftOrdering, FftScaleFactor};
use lib_jsl_core::ErrorsJSL;
use num::Complex;
use std::f64::consts::TAU;

type C64 = Complex<f64>;

/// A planned FFT for real input of length `n`. The spectrum of a real signal is conjugate
/// symmetric, so only the `n/2 + 1` non-negative-frequency bins are returned.
///
/// For even `n` the signal is packed into a complex sequence of half the length (even samples as
/// real parts, odd samples as imaginary parts), transformed, and untangled with one twiddle per
/// bin, about twice as fast as a complex FFT of the full length. Odd lengths use a full complex
/// transform. The forward transform is unscaled; [`RealFft::inverse`] divides by `n`, so it
/// exactly inverts [`RealFft::forward`].
#[derive(Debug)]
pub struct RealFft {
    n: usize,
    forward: Fft,
    inverse: Fft,
    twiddles: Vec<C64>,
}

impl RealFft {
    pub fn new(n: usize) -> Result<Self, ErrorsJSL> {
        if n == 0 {
            return Err(ErrorsJSL::InvalidInputRange("The length must be positive."));
        }
        let inner = if n.is_multiple_of(2) { n / 2 } else { n };
        let forward = Fft::with_plan(inner, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard)?;
        let inverse = Fft::with_plan(inner, FftScaleFactor::N, FftDirection::Inverse, FftOrdering::Standard)?;
        let twiddles = (0..=n / 2).map(|k| C64::from_polar(1.0, -TAU * k as f64 / n as f64)).collect();
        Ok(RealFft { n, forward, inverse, twiddles })
    }

    /// The signal length.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The number of spectrum bins, `n/2 + 1`.
    pub fn spectrum_len(&self) -> usize {
        self.n / 2 + 1
    }

    /// The bins `X_0 ..= X_{n/2}` of the unscaled DFT `X_k = Σ x_m e^{-2πi km/n}`.
    pub fn forward(&mut self, x: &[f64]) -> Result<Vec<C64>, ErrorsJSL> {
        if x.len() != self.n {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.len(), self.n)));
        }
        if !self.n.is_multiple_of(2) {
            let full = self.forward.execute(&x.iter().map(|&v| C64::new(v, 0.0)).collect::<Vec<_>>())?;
            return Ok(full[..self.spectrum_len()].to_vec());
        }
        let m = self.n / 2;
        let packed: Vec<C64> = (0..m).map(|j| C64::new(x[2 * j], x[2 * j + 1])).collect();
        let z = self.forward.execute(&packed)?;
        let half = C64::new(0.5, 0.0);
        let minus_half_i = C64::new(0.0, -0.5);
        Ok((0..=m)
            .map(|k| {
                let a = z[k % m];
                let b = z[(m - k) % m].conj();
                let even = half * (a + b);
                let odd = minus_half_i * (a - b);
                even + self.twiddles[k] * odd
            })
            .collect())
    }

    /// The real signal whose spectrum bins are `spectrum` (`n/2 + 1` values; the imaginary parts
    /// of `X_0`, and of `X_{n/2}` for even `n`, are ignored as they must vanish).
    pub fn inverse(&mut self, spectrum: &[C64]) -> Result<Vec<f64>, ErrorsJSL> {
        if spectrum.len() != self.spectrum_len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((spectrum.len(), self.spectrum_len())));
        }
        let n = self.n;
        if !n.is_multiple_of(2) {
            let full: Vec<C64> = (0..n).map(|k| if k < spectrum.len() { spectrum[k] } else { spectrum[n - k].conj() }).collect();
            let mut full = full;
            full[0].im = 0.0;
            return Ok(self.inverse.execute(&full)?.iter().map(|v| v.re).collect());
        }
        let m = n / 2;
        let mut s = spectrum.to_vec();
        s[0].im = 0.0;
        s[m].im = 0.0;
        let z: Vec<C64> = (0..m)
            .map(|k| {
                let (a, b) = (s[k], s[m - k].conj());
                let even = (a + b) * 0.5;
                let odd = (a - b) * 0.5 * self.twiddles[k].conj();
                even + C64::new(0.0, 1.0) * odd
            })
            .collect();
        let packed = self.inverse.execute(&z)?;
        Ok(packed.iter().flat_map(|v| [v.re, v.im]).collect())
    }
}

/// One-shot real FFT: the `n/2 + 1` non-negative-frequency bins of a real signal.
pub fn rfft(x: &[f64]) -> Result<Vec<C64>, ErrorsJSL> {
    RealFft::new(x.len())?.forward(x)
}

/// One-shot inverse real FFT for a signal of length `n`.
pub fn irfft(spectrum: &[C64], n: usize) -> Result<Vec<f64>, ErrorsJSL> {
    RealFft::new(n)?.inverse(spectrum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dft(x: &[f64]) -> Vec<C64> {
        let n = x.len();
        (0..n).map(|k| x.iter().enumerate().map(|(m, &v)| C64::from_polar(v, -TAU * ((k * m) % n) as f64 / n as f64)).sum()).collect()
    }

    fn signal(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.37).sin() + 0.2 * (i as f64).powf(0.5) - 0.5 * ((i * 7) % 5) as f64).collect()
    }

    #[test]
    fn matches_the_complex_dft_for_even_odd_and_power_of_two_lengths() {
        for n in [1, 2, 3, 4, 5, 6, 8, 9, 12, 15, 16, 31, 64, 100, 127, 256] {
            let x = signal(n);
            let got = rfft(&x).unwrap();
            let want = dft(&x);
            assert_eq!(got.len(), n / 2 + 1);
            let worst = got.iter().zip(&want).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
            assert!(worst < 1e-9 * n as f64, "n = {n}: {worst}");
            let back = irfft(&got, n).unwrap();
            assert!(back.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-10), "round trip n = {n}");
        }
    }

    #[test]
    fn planned_reuse_and_validation() {
        let mut plan = RealFft::new(10).unwrap();
        assert_eq!((plan.len(), plan.spectrum_len()), (10, 6));
        for seed in 0..3 {
            let x: Vec<f64> = (0..10).map(|i| ((i + seed) as f64).cos()).collect();
            let s = plan.forward(&x).unwrap();
            assert!(s[0].im.abs() < 1e-12 && s[5].im.abs() < 1e-12); // DC and Nyquist are real
            assert!(plan.inverse(&s).unwrap().iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-12));
        }
        assert!(plan.forward(&[1.0; 3]).is_err() && plan.inverse(&[C64::new(0.0, 0.0); 3]).is_err());
        assert!(RealFft::new(0).is_err());
        // A pure cosine puts all its energy in one bin.
        let n = 64;
        let x: Vec<f64> = (0..n).map(|i| (TAU * 5.0 * i as f64 / n as f64).cos()).collect();
        let s = rfft(&x).unwrap();
        assert!((s[5].re - 32.0).abs() < 1e-9 && s.iter().enumerate().filter(|(k, _)| *k != 5).all(|(_, v)| v.norm() < 1e-9));
    }
}
