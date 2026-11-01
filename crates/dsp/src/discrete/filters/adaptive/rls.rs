//! Recursive least-squares adaptive FIR filter.

use lib_jsl_core::ErrorsJSL;
use num::Complex;

type C64 = Complex<f64>;

/// An adaptive FIR filter trained by exponentially weighted recursive least squares.
///
/// It minimises `Σ λ^{n−i} |d_i − wᴴu_i|²` exactly at every step, updating the inverse
/// correlation matrix `P` by the matrix inversion lemma: with gain `k = P u / (λ + uᴴ P u)`,
/// `e = d − wᴴu`, `w ← w + k conj(e)` and `P ← (P − k uᴴ P) / λ`. Cost is `O(M²)` per sample, but
/// convergence takes a few multiples of `M` samples independently of the input's eigenvalue
/// spread, where LMS slows down. `λ` near 1 averages over about `1 / (1 − λ)` samples.
#[derive(Debug, Clone)]
pub struct RlsFilter {
    weights: Vec<C64>,
    regressor: Vec<C64>,
    p: Vec<Vec<C64>>,
    forgetting: f64,
}

impl RlsFilter {
    /// `initial_inverse_correlation` sets `P = δ I`; large `δ` (e.g. 100 / input power) means
    /// little prior confidence in the zero initial weights.
    pub fn new(taps: usize, forgetting: f64, initial_inverse_correlation: f64) -> Result<Self, ErrorsJSL> {
        if taps == 0 || !(forgetting > 0.0 && forgetting <= 1.0) || initial_inverse_correlation.is_nan() || initial_inverse_correlation <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("Need taps > 0, 0 < λ <= 1 and δ > 0."));
        }
        let zero = C64::new(0.0, 0.0);
        let p = (0..taps).map(|i| (0..taps).map(|j| if i == j { C64::new(initial_inverse_correlation, 0.0) } else { zero }).collect()).collect();
        Ok(RlsFilter { weights: vec![zero; taps], regressor: vec![zero; taps], p, forgetting })
    }

    pub fn weights(&self) -> &[C64] {
        &self.weights
    }

    /// Filters one sample against the desired response, adapts, and returns `(y, e)` (the a
    /// priori output and error).
    pub fn adapt(&mut self, x: C64, desired: C64) -> (C64, C64) {
        self.regressor.rotate_right(1);
        self.regressor[0] = x;
        let u = &self.regressor;
        let pu: Vec<C64> = self.p.iter().map(|row| row.iter().zip(u).map(|(a, b)| a * b).sum()).collect();
        let denominator = self.forgetting + u.iter().zip(&pu).map(|(a, b)| a.conj() * b).sum::<C64>().re;
        let k: Vec<C64> = pu.iter().map(|v| v / denominator).collect();
        let y: C64 = self.weights.iter().zip(u).map(|(w, v)| w.conj() * v).sum();
        let e = desired - y;
        self.weights.iter_mut().zip(&k).for_each(|(w, ki)| *w += ki * e.conj());
        // uᴴP = (P u)ᴴ for Hermitian P. Updating one triangle and mirroring it keeps P exactly
        // Hermitian; without this, rounding drives P indefinite and it diverges like λ^{−n}.
        for (i, ki) in k.iter().enumerate() {
            for (j, puj) in pu.iter().enumerate().skip(i) {
                let v = (self.p[i][j] - ki * puj.conj()) / self.forgetting;
                self.p[i][j] = v;
                self.p[j][i] = v.conj();
            }
        }
        (y, e)
    }

    /// Runs [`RlsFilter::adapt`] over a block.
    pub fn adapt_block(&mut self, x: &[C64], desired: &[C64]) -> Result<(Vec<C64>, Vec<C64>), ErrorsJSL> {
        if x.len() != desired.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.len(), desired.len())));
        }
        Ok(x.iter().zip(desired).map(|(&a, &b)| self.adapt(a, b)).unzip())
    }
}

#[cfg(test)]
mod tests {
    use super::super::lms::LmsFilter;
    use super::super::lms::test_support::*;
    use super::*;

    #[test]
    fn converges_in_a_few_multiples_of_the_filter_length_even_for_coloured_input() {
        // Strongly coloured input (one-pole low-pass) slows LMS; RLS is unaffected.
        let white = noise(4000, 5);
        let mut x = vec![C64::new(0.0, 0.0); white.len()];
        for n in 0..x.len() {
            x[n] = white[n] + if n > 0 { 0.95 * x[n - 1] } else { C64::new(0.0, 0.0) };
        }
        let h: Vec<C64> = [C64::new(1.0, 0.5), C64::new(-0.3, 0.2), C64::new(0.1, 0.0), C64::new(0.0, -0.4)].to_vec();
        let d = plant(&x, &h);
        // A large δ makes the zero-weight prior negligible, so the noise-free fit is exact.
        let mut rls = RlsFilter::new(4, 1.0, 1e8).unwrap();
        let (_, e_rls) = rls.adapt_block(&x[..60], &d[..60]).unwrap();
        assert!(rls.weights().iter().zip(&h).all(|(w, hk)| (w.conj() - hk).norm() < 1e-6), "{:?}", rls.weights());
        assert!(e_rls[50..].iter().all(|e| e.norm() < 1e-5));
        let mut nlms = LmsFilter::new(4, 0.5, true).unwrap();
        nlms.adapt_block(&x[..60], &d[..60]).unwrap();
        let nlms_error: f64 = nlms.weights().iter().zip(&h).map(|(w, hk)| (w.conj() - hk).norm()).sum();
        assert!(nlms_error > 1e-2, "NLMS should still be far from converged: {nlms_error}");
    }

    #[test]
    fn forgetting_tracks_a_changing_system_and_validates() {
        let x = noise(4000, 9);
        let (h1, h2) = ([C64::new(1.0, 0.0), C64::new(0.5, 0.0)], [C64::new(-0.7, 0.0), C64::new(0.2, 0.0)]);
        let mut d = plant(&x[..2000], &h1);
        d.extend(plant(&x, &h2)[2000..].iter());
        let mut rls = RlsFilter::new(2, 0.98, 100.0).unwrap();
        rls.adapt_block(&x, &d).unwrap();
        assert!(rls.weights().iter().zip(&h2).all(|(w, hk)| (w.conj() - hk).norm() < 1e-6), "{:?}", rls.weights());
        assert!(RlsFilter::new(2, 1.2, 1.0).is_err() && RlsFilter::new(2, 0.99, 0.0).is_err());
    }
}
