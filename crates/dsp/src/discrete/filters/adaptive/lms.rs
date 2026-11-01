//! Least-mean-squares adaptive FIR filters.

use lib_jsl_core::ErrorsJSL;
use num::Complex;

type C64 = Complex<f64>;

/// An adaptive FIR filter trained by LMS or normalised LMS.
///
/// With regressor `u = [x_n, x_{n−1}, …]` the output is `y = wᴴu`, the error `e = d − y`, and the
/// update `w ← w + μ conj(e) u` (LMS) or `w ← w + μ conj(e) u / (ε + ‖u‖²)` (NLMS). LMS converges
/// in the mean for `0 < μ < 2 / (M·P_x)`; NLMS normalises by the input power, so `0 < μ < 2`
/// regardless of the signal level. Works for real signals passed as complex numbers with zero
/// imaginary part.
#[derive(Debug, Clone)]
pub struct LmsFilter {
    weights: Vec<C64>,
    regressor: Vec<C64>,
    step_size: f64,
    normalized: bool,
    regularization: f64,
}

impl LmsFilter {
    pub fn new(taps: usize, step_size: f64, normalized: bool) -> Result<Self, ErrorsJSL> {
        if taps == 0 || step_size.is_nan() || step_size <= 0.0 || (normalized && step_size >= 2.0) {
            return Err(ErrorsJSL::InvalidInputRange("Need taps > 0 and a positive step size (below 2 for NLMS)."));
        }
        Ok(LmsFilter { weights: vec![C64::new(0.0, 0.0); taps], regressor: vec![C64::new(0.0, 0.0); taps], step_size, normalized, regularization: 1e-8 })
    }

    pub fn weights(&self) -> &[C64] {
        &self.weights
    }

    /// Filters one sample against the desired response, adapts, and returns `(y, e)`.
    pub fn adapt(&mut self, x: C64, desired: C64) -> (C64, C64) {
        self.regressor.rotate_right(1);
        self.regressor[0] = x;
        let y: C64 = self.weights.iter().zip(&self.regressor).map(|(w, u)| w.conj() * u).sum();
        let e = desired - y;
        let mu = if self.normalized {
            self.step_size / (self.regularization + self.regressor.iter().map(|u| u.norm_sqr()).sum::<f64>())
        } else {
            self.step_size
        };
        self.weights.iter_mut().zip(&self.regressor).for_each(|(w, u)| *w += mu * e.conj() * u);
        (y, e)
    }

    /// Filters one sample without adapting.
    pub fn filter(&mut self, x: C64) -> C64 {
        self.regressor.rotate_right(1);
        self.regressor[0] = x;
        self.weights.iter().zip(&self.regressor).map(|(w, u)| w.conj() * u).sum()
    }

    /// Runs [`LmsFilter::adapt`] over a block, returning the outputs and errors.
    pub fn adapt_block(&mut self, x: &[C64], desired: &[C64]) -> Result<(Vec<C64>, Vec<C64>), ErrorsJSL> {
        if x.len() != desired.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.len(), desired.len())));
        }
        Ok(x.iter().zip(desired).map(|(&a, &b)| self.adapt(a, b)).unzip())
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::C64;

    /// A deterministic white-ish real sequence in [−1, 1).
    pub fn noise(n: usize, seed: u64) -> Vec<C64> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                C64::new((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0, 0.0)
            })
            .collect()
    }

    /// The plant h applied to x (FIR), plus optional measurement noise.
    pub fn plant(x: &[C64], h: &[C64]) -> Vec<C64> {
        (0..x.len()).map(|n| (0..h.len()).filter(|&k| k <= n).map(|k| h[k] * x[n - k]).sum()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    #[test]
    fn identifies_an_unknown_fir_system() {
        let h: Vec<C64> = [0.8, -0.4, 0.25, 0.1, -0.05].iter().map(|&v| C64::new(v, 0.0)).collect();
        let x = noise(20_000, 1);
        let d = plant(&x, &h);
        for normalized in [false, true] {
            let mut f = LmsFilter::new(5, if normalized { 0.5 } else { 0.05 }, normalized).unwrap();
            let (_, e) = f.adapt_block(&x, &d).unwrap();
            // y = wᴴu, so the weights converge to conj(h) (= h for real plants).
            assert!(f.weights().iter().zip(&h).all(|(w, hk)| (w.conj() - hk).norm() < 1e-6), "normalized={normalized}: {:?}", f.weights());
            let tail: f64 = e[19_000..].iter().map(|v| v.norm_sqr()).sum::<f64>() / 1000.0;
            assert!(tail < 1e-12);
        }
    }

    #[test]
    fn cancels_a_correlated_interferer_and_validates() {
        // d = signal + filtered reference; the filter learns to remove the reference's echo.
        let n = 30_000;
        let reference = noise(n, 3);
        let echo = plant(&reference, &[C64::new(0.0, 0.6), C64::new(0.3, -0.2)]);
        let signal: Vec<C64> = (0..n).map(|i| C64::new((0.01 * i as f64).sin() * 0.1, 0.0)).collect();
        let d: Vec<C64> = signal.iter().zip(&echo).map(|(s, e)| s + e).collect();
        let mut f = LmsFilter::new(4, 0.02, true).unwrap();
        let (_, e) = f.adapt_block(&reference, &d).unwrap();
        // The error converges to the wanted signal, up to the misadjustment (about μ/2 times
        // the signal power, which acts as noise on the adaptation).
        let signal_power: f64 = signal.iter().map(|s| s.norm_sqr()).sum::<f64>() / n as f64;
        let residual: f64 = (n - 2000..n).map(|i| (e[i] - signal[i]).norm_sqr()).sum::<f64>() / 2000.0;
        assert!(residual < 0.05 * signal_power, "{residual} vs {signal_power}");
        assert!(LmsFilter::new(0, 0.1, false).is_err() && LmsFilter::new(3, 2.5, true).is_err());
        assert!(f.adapt_block(&reference[..3], &d[..2]).is_err());
    }
}
