//! Decision feedback equalizer (DFE).
//!
//! A channel with memory smears each symbol over its neighbours (inter-symbol interference).
//! A plain linear equalizer fights this by inverting the channel, which amplifies noise where the
//! channel is weak. The DFE has two parts:
//!
//! - a **feed-forward** filter over recent received samples, which handles *pre-cursor* ISI and
//!   shapes the signal;
//! - a **feedback** filter over the receiver's own *past decisions*, which subtracts the
//!   *post-cursor* ISI exactly. Because it works from clean, already-decided symbols rather than
//!   noisy samples, it cancels interference without enhancing noise.
//!
//! The output before the slicer is `y[n] = Σ w_ff[k] x[n-k] + Σ w_fb[k] d[n-1-k]`, where
//! `e = d − y` is the error against the target symbol. The target is a known training symbol, or
//! (once converged) the slicer's own decision, which is *decision-directed* operation. For a
//! causal channel `h = [1, h₁, h₂, …]` the feedback taps converge to `−h₁, −h₂, …`.
//!
//! Two adaptation algorithms are offered:
//!
//! - **NLMS** (`w ← w + μ e conj(u) / (ε + ‖u‖²)`): cheap (`O(M)` per symbol) and robust, but its
//!   convergence along weakly excited directions is slow, and ISI makes the regressor strongly
//!   correlated, so it can need thousands of symbols.
//! - **RLS** (recursive least squares with forgetting factor `λ`): `O(M²)` per symbol but converges
//!   in a few multiples of `M` symbols regardless of the channel, which is why it is the usual
//!   choice for training.
//!
//! Error propagation (a wrong decision corrupting the next few outputs) is the DFE's known weakness
//! and is why training first, at a good signal-to-noise ratio, matters.

use super::constellations::Constellation;
use crate::discrete::stream_operator::{StreamOperator, StreamOperatorManagement};
use lib_jsl_core::ErrorsJSL;
use num_complex::Complex;

/// How the equalizer adapts its weights.
#[derive(Clone, Debug)]
enum Adaptation {
    Nlms { step_size: f64 },
    Rls { forgetting: f64, p: Vec<Vec<Complex<f64>>> },
}

/// An adaptive decision feedback equalizer for complex symbols.
#[derive(Clone, Debug)]
pub struct DecisionFeedbackEqualizer {
    constellation: Constellation,
    main_tap: usize,
    forward_len: usize,
    adaptation: Adaptation,
    /// Weights: the feed-forward taps followed by the feedback taps.
    weights: Vec<Complex<f64>>,
    /// Regressor: recent received samples then recent decisions, each newest first.
    regressor: Vec<Complex<f64>>,
}

const REGULARIZATION: f64 = 1e-6;
/// Initial inverse-correlation scale for RLS (large = trust the data quickly).
const RLS_INITIAL_P: f64 = 100.0;

impl DecisionFeedbackEqualizer {
    /// An NLMS equalizer with `forward_taps >= 1` and `feedback_taps >= 0` taps, step size
    /// `0 < step_size < 2` (0.01 to 0.2 is typical), deciding on `constellation`. The feed-forward
    /// main tap starts at index 0, so the output at time `n` estimates the symbol that arrived
    /// with sample `n`.
    pub fn new(constellation: Constellation, forward_taps: usize, feedback_taps: usize, step_size: f64) -> Result<Self, ErrorsJSL> {
        Self::with_main_tap(constellation, forward_taps, feedback_taps, step_size, 0)
    }

    /// Like [`new`](Self::new) but with the feed-forward main tap initialised at `main_tap`, so
    /// that `main_tap` samples of pre-cursor ISI can be handled; the output then estimates the
    /// symbol that arrived `main_tap` samples earlier.
    pub fn with_main_tap(constellation: Constellation, forward_taps: usize, feedback_taps: usize, step_size: f64, main_tap: usize) -> Result<Self, ErrorsJSL> {
        if !(step_size > 0.0 && step_size < 2.0) {
            return Err(ErrorsJSL::InvalidInputRange("NLMS step size must be in (0, 2)"));
        }
        Self::build(constellation, forward_taps, feedback_taps, main_tap, Adaptation::Nlms { step_size })
    }

    /// An RLS equalizer with forgetting factor `0.9 <= forgetting <= 1` (0.995 to 0.9995 is
    /// typical; 1 weights all history equally). Converges in a few multiples of the total tap count.
    pub fn new_rls(constellation: Constellation, forward_taps: usize, feedback_taps: usize, forgetting: f64) -> Result<Self, ErrorsJSL> {
        Self::with_main_tap_rls(constellation, forward_taps, feedback_taps, forgetting, 0)
    }

    /// Like [`new_rls`](Self::new_rls) with the feed-forward main tap at `main_tap`.
    pub fn with_main_tap_rls(constellation: Constellation, forward_taps: usize, feedback_taps: usize, forgetting: f64, main_tap: usize) -> Result<Self, ErrorsJSL> {
        if !(0.9..=1.0).contains(&forgetting) {
            return Err(ErrorsJSL::InvalidInputRange("RLS forgetting factor must be in [0.9, 1]"));
        }
        let m = forward_taps + feedback_taps;
        let mut p = vec![vec![Complex::new(0.0, 0.0); m]; m];
        for (i, row) in p.iter_mut().enumerate() {
            row[i] = Complex::new(RLS_INITIAL_P, 0.0);
        }
        Self::build(constellation, forward_taps, feedback_taps, main_tap, Adaptation::Rls { forgetting, p })
    }

    fn build(constellation: Constellation, forward_taps: usize, feedback_taps: usize, main_tap: usize, adaptation: Adaptation) -> Result<Self, ErrorsJSL> {
        if forward_taps == 0 || main_tap >= forward_taps {
            return Err(ErrorsJSL::InvalidInputRange("Need at least one feed-forward tap and main_tap < forward_taps"));
        }
        let total = forward_taps + feedback_taps;
        let mut weights = vec![Complex::new(0.0, 0.0); total];
        weights[main_tap] = Complex::new(1.0, 0.0);
        Ok(DecisionFeedbackEqualizer { constellation, main_tap, forward_len: forward_taps, adaptation, weights, regressor: vec![Complex::new(0.0, 0.0); total] })
    }

    pub fn forward_taps(&self) -> &[Complex<f64>] {
        &self.weights[..self.forward_len]
    }

    /// The feedback taps; for a causal channel they converge to minus the post-cursor ISI.
    pub fn feedback_taps(&self) -> &[Complex<f64>] {
        &self.weights[self.forward_len..]
    }

    /// The main tap index (the equalizer's decision delay in samples).
    pub fn main_tap(&self) -> usize {
        self.main_tap
    }

    /// Processes one received sample. With `Some(target)` the equalizer trains on that symbol;
    /// with `None` it uses its own decision. Returns `(equalized sample, decision)`.
    pub fn equalize_sample(&mut self, x: Complex<f64>, target: Option<Complex<f64>>) -> (Complex<f64>, Complex<f64>) {
        let fl = self.forward_len;
        // Shift the received-sample part of the regressor and insert the new sample.
        self.regressor[..fl].rotate_right(1);
        self.regressor[0] = x;
        let y: Complex<f64> = self.weights.iter().zip(&self.regressor).map(|(w, u)| w * u).sum();
        let decision = self.constellation.slice(y);
        let desired = target.unwrap_or(decision);
        let error = desired - y;
        match &mut self.adaptation {
            Adaptation::Nlms { step_size } => {
                let power: f64 = self.regressor.iter().map(|v| v.norm_sqr()).sum();
                let gain = *step_size / (REGULARIZATION + power);
                for (w, u) in self.weights.iter_mut().zip(&self.regressor) {
                    *w += error * u.conj() * gain;
                }
            }
            Adaptation::Rls { forgetting, p } => {
                let m = self.weights.len();
                // pq = P conj(u); denominator = lambda + u^T P conj(u).
                let q: Vec<Complex<f64>> = self.regressor.iter().map(|u| u.conj()).collect();
                let pq: Vec<Complex<f64>> = (0..m).map(|i| (0..m).map(|j| p[i][j] * q[j]).sum()).collect();
                let denom = *forgetting + self.regressor.iter().zip(&pq).map(|(u, v)| u * v).sum::<Complex<f64>>();
                let k: Vec<Complex<f64>> = pq.iter().map(|v| v / denom).collect();
                for (w, ki) in self.weights.iter_mut().zip(&k) {
                    *w += ki * error;
                }
                // P <- (P - k u^T P) / lambda, where u^T P is the row vector with entries sum_i u_i P[i][j].
                let utp: Vec<Complex<f64>> = (0..m).map(|j| (0..m).map(|i| self.regressor[i] * p[i][j]).sum()).collect();
                for i in 0..m {
                    for j in 0..m {
                        p[i][j] = (p[i][j] - k[i] * utp[j]) / *forgetting;
                    }
                }
            }
        }
        // Shift the decision part of the regressor, feeding back the target while training.
        if self.regressor.len() > fl {
            self.regressor[fl..].rotate_right(1);
            self.regressor[fl] = desired;
        }
        (y, decision)
    }

    /// Trains on known symbols: `targets[i]` is the desired output for `received[i]` (when the
    /// main tap is nonzero, shift the targets by that delay). Returns the squared error of each
    /// step so convergence can be inspected.
    pub fn train(&mut self, received: &[Complex<f64>], targets: &[Complex<f64>]) -> Result<Vec<f64>, ErrorsJSL> {
        if received.len() != targets.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((received.len(), targets.len())));
        }
        Ok(received
            .iter()
            .zip(targets)
            .map(|(&x, &t)| {
                let (y, _) = self.equalize_sample(x, Some(t));
                (t - y).norm_sqr()
            })
            .collect())
    }
}

impl StreamOperatorManagement for DecisionFeedbackEqualizer {
    /// Clears the delay lines and restarts the weights (a fresh equalizer).
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.regressor.iter_mut().for_each(|v| *v = Complex::new(0.0, 0.0));
        self.weights.iter_mut().for_each(|v| *v = Complex::new(0.0, 0.0));
        self.weights[self.main_tap] = Complex::new(1.0, 0.0);
        if let Adaptation::Rls { p, .. } = &mut self.adaptation {
            for (i, row) in p.iter_mut().enumerate() {
                row.iter_mut().for_each(|v| *v = Complex::new(0.0, 0.0));
                row[i] = Complex::new(RLS_INITIAL_P, 0.0);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

/// Decision-directed streaming: each input sample yields the equalized (pre-slicer) sample.
impl StreamOperator<Complex<f64>, Complex<f64>> for DecisionFeedbackEqualizer {
    fn process(&mut self, data_in: &[Complex<f64>]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        Ok(Some(data_in.iter().map(|&x| self.equalize_sample(x, None).0).collect()))
    }

    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        Ok(None)
    }
}

/// Builds an equalizer, trains it on the first `training.len()` symbols of `received` against the
/// known `training` symbols, then equalizes the rest decision-directed. Returns the equalized
/// samples for the whole record.
pub fn decision_feedback_equalizer(
    received: &[Complex<f64>],
    training: &[Complex<f64>],
    constellation: Constellation,
    forward_taps: usize,
    feedback_taps: usize,
    step_size: f64,
) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
    if training.len() > received.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((training.len(), received.len())));
    }
    let mut eq = DecisionFeedbackEqualizer::new(constellation, forward_taps, feedback_taps, step_size)?;
    let mut out = Vec::with_capacity(received.len());
    for (i, &x) in received.iter().enumerate() {
        out.push(eq.equalize_sample(x, training.get(i).copied()).0);
    }
    Ok(out)
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

    fn symbols(c: &Constellation, n: usize, seed: &mut u64) -> Vec<Complex<f64>> {
        (0..n).map(|_| c.points()[((lcg(seed) + 0.5) * c.size() as f64) as usize % c.size()]).collect()
    }

    fn through_channel(x: &[Complex<f64>], h: &[Complex<f64>], noise: f64, seed: &mut u64) -> Vec<Complex<f64>> {
        (0..x.len())
            .map(|n| {
                let s: Complex<f64> = h.iter().enumerate().filter(|(k, _)| *k <= n).map(|(k, c)| c * x[n - k]).sum();
                s + Complex::new(gauss(seed), gauss(seed)) * noise
            })
            .collect()
    }

    fn errors(c: &Constellation, y: &[Complex<f64>], want: &[Complex<f64>]) -> usize {
        y.iter().zip(want).filter(|(a, b)| c.slice(**a) != **b).count()
    }

    #[test]
    fn cancels_post_cursor_isi_and_learns_the_negated_channel_taps() {
        let c = Constellation::qpsk();
        let mut seed = 3;
        let h = [Complex::new(1.0, 0.0), Complex::new(0.5, 0.2), Complex::new(-0.3, 0.1)];
        let tx = symbols(&c, 10_000, &mut seed);
        let rx = through_channel(&tx, &h, 0.02, &mut seed);
        // The uncompensated channel makes many errors.
        assert!(errors(&c, &rx[3000..], &tx[3000..]) > 100);

        let mut eq = DecisionFeedbackEqualizer::new_rls(c.clone(), 3, 2, 0.9995).unwrap();
        let mse = eq.train(&rx[..3000], &tx[..3000]).unwrap();
        let early: f64 = mse[..20].iter().sum::<f64>() / 20.0;
        let late: f64 = mse[2900..].iter().sum::<f64>() / 100.0;
        assert!(late < early * 0.05 && late < 2e-3, "MSE {early} -> {late}");
        // Feedback taps converge to minus the post-cursors; the first forward tap to ~1.
        let fb = eq.feedback_taps();
        assert!((fb[0] + h[1]).norm() < 0.02 && (fb[1] + h[2]).norm() < 0.02, "{fb:?}");
        assert!((eq.forward_taps()[0] - Complex::new(1.0, 0.0)).norm() < 0.02, "{:?}", eq.forward_taps());
        // Now decision-directed on the rest: no symbol errors.
        let y = eq.process(&rx[3000..]).unwrap().unwrap();
        assert_eq!(errors(&c, &y, &tx[3000..]), 0);
    }

    #[test]
    fn rls_converges_where_nlms_is_still_crawling() {
        // The ISI makes the regressor ill-conditioned (eigenvalue spread in the hundreds), so NLMS
        // learns the weakly excited directions very slowly; RLS is unaffected.
        let c = Constellation::qpsk();
        let mut seed = 31;
        let h = [Complex::new(1.0, 0.0), Complex::new(0.5, 0.2), Complex::new(-0.3, 0.1)];
        let tx = symbols(&c, 1500, &mut seed);
        let rx = through_channel(&tx, &h, 0.01, &mut seed);
        let tap_error = |eq: &DecisionFeedbackEqualizer| (eq.feedback_taps()[0] + h[1]).norm() + (eq.feedback_taps()[1] + h[2]).norm();
        let mut rls = DecisionFeedbackEqualizer::new_rls(c.clone(), 3, 2, 0.999).unwrap();
        let mut nlms = DecisionFeedbackEqualizer::new(c, 3, 2, 0.05).unwrap();
        rls.train(&rx, &tx).unwrap();
        nlms.train(&rx, &tx).unwrap();
        assert!(tap_error(&rls) < 0.02, "RLS tap error {}", tap_error(&rls));
        assert!(tap_error(&nlms) > 5.0 * tap_error(&rls), "NLMS {} vs RLS {}", tap_error(&nlms), tap_error(&rls));
    }

    #[test]
    fn nlms_still_equalizes_adequately_for_decision_making() {
        let c = Constellation::qpsk();
        let mut seed = 17;
        let h = [Complex::new(1.0, 0.0), Complex::new(0.5, 0.2), Complex::new(-0.3, 0.1)];
        let tx = symbols(&c, 12_000, &mut seed);
        let rx = through_channel(&tx, &h, 0.02, &mut seed);
        let mut eq = DecisionFeedbackEqualizer::new(c.clone(), 3, 2, 0.1).unwrap();
        let mse = eq.train(&rx[..6000], &tx[..6000]).unwrap();
        let early: f64 = mse[..100].iter().sum::<f64>() / 100.0;
        let late: f64 = mse[5900..].iter().sum::<f64>() / 100.0;
        assert!(late < early * 0.05, "MSE {early} -> {late}");
        let y = eq.process(&rx[6000..]).unwrap().unwrap();
        assert_eq!(errors(&c, &y, &tx[6000..]), 0);
    }

    #[test]
    fn handles_pre_cursor_isi_with_a_delayed_main_tap() {
        let c = Constellation::qpsk();
        let mut seed = 5;
        // Precursor 0.35, main 1.0, post-cursors 0.4 and -0.2.
        let h = [Complex::new(0.35, 0.0), Complex::new(1.0, 0.0), Complex::new(0.4, 0.0), Complex::new(-0.2, 0.0)];
        let tx = symbols(&c, 9000, &mut seed);
        let rx = through_channel(&tx, &h, 0.02, &mut seed);
        let delay = 1; // the main path is the second tap, so the output lags the input by 1
        let mut eq = DecisionFeedbackEqualizer::with_main_tap_rls(c.clone(), 5, 2, 0.999, delay).unwrap();
        // The output at step n estimates the symbol sent at n - delay.
        let targets: Vec<Complex<f64>> = (0..2000).map(|n| if n >= delay { tx[n - delay] } else { Complex::new(0.0, 0.0) }).collect();
        eq.train(&rx[..2000], &targets[..2000]).unwrap();
        let y = eq.process(&rx[2000..]).unwrap().unwrap();
        let want: Vec<Complex<f64>> = (2000..9000).map(|n| tx[n - delay]).collect();
        assert_eq!(errors(&c, &y, &want), 0);
        assert_eq!(eq.main_tap(), 1);
    }

    #[test]
    fn works_with_a_qam_constellation_and_the_one_shot_helper() {
        let c = Constellation::qam(16).unwrap();
        let mut seed = 9;
        let h = [Complex::new(1.0, 0.0), Complex::new(0.25, -0.1)];
        let tx = symbols(&c, 10_000, &mut seed);
        let rx = through_channel(&tx, &h, 0.01, &mut seed);
        let y = decision_feedback_equalizer(&rx, &tx[..3000], c.clone(), 3, 3, 0.1).unwrap();
        assert_eq!(y.len(), rx.len());
        assert_eq!(errors(&c, &y[6000..], &tx[6000..]), 0);
        assert!(decision_feedback_equalizer(&rx[..10], &tx[..20], c, 3, 3, 0.03).is_err());
    }

    #[test]
    fn without_isi_the_equalizer_leaves_the_signal_alone() {
        let c = Constellation::bpsk();
        let mut seed = 2;
        let tx = symbols(&c, 500, &mut seed);
        let mut eq = DecisionFeedbackEqualizer::new(c.clone(), 2, 2, 0.1).unwrap();
        let y = eq.process(&tx).unwrap().unwrap();
        assert!(y.iter().zip(&tx).all(|(a, b)| (a - b).norm() < 1e-9));
        assert!(eq.feedback_taps().iter().all(|w| w.norm() < 1e-9));
    }

    #[test]
    fn reset_restores_the_initial_weights_and_streaming_is_chunk_independent() {
        let c = Constellation::qpsk();
        let mut seed = 12;
        let h = [Complex::new(1.0, 0.0), Complex::new(0.4, 0.0)];
        let tx = symbols(&c, 800, &mut seed);
        let rx = through_channel(&tx, &h, 0.01, &mut seed);
        let make = || DecisionFeedbackEqualizer::new(c.clone(), 3, 2, 0.05).unwrap();
        let whole = make().process(&rx).unwrap().unwrap();
        let mut eq = make();
        let mut parts = Vec::new();
        for ch in rx.chunks(33) {
            parts.extend(eq.process(ch).unwrap().unwrap());
        }
        assert!(whole.iter().zip(&parts).all(|(a, b)| (a - b).norm() < 1e-12));
        StreamOperatorManagement::reset(&mut eq).unwrap();
        assert!(eq.feedback_taps().iter().all(|w| w.norm() == 0.0));
        assert_eq!(eq.forward_taps()[0], Complex::new(1.0, 0.0));
        assert!(eq.process(&[]).unwrap().is_none() && eq.flush().unwrap().is_none());
    }

    #[test]
    fn invalid_parameters() {
        let c = Constellation::bpsk();
        assert!(DecisionFeedbackEqualizer::new(c.clone(), 0, 2, 0.1).is_err());
        assert!(DecisionFeedbackEqualizer::new(c.clone(), 3, 2, 0.0).is_err());
        assert!(DecisionFeedbackEqualizer::new(c.clone(), 3, 2, 2.0).is_err());
        assert!(DecisionFeedbackEqualizer::with_main_tap(c.clone(), 3, 2, 0.1, 3).is_err());
        assert!(DecisionFeedbackEqualizer::new_rls(c.clone(), 3, 2, 0.5).is_err());
        assert!(DecisionFeedbackEqualizer::new_rls(c.clone(), 3, 2, 1.1).is_err());
        assert!(DecisionFeedbackEqualizer::new_rls(c.clone(), 0, 2, 0.99).is_err());
        let mut eq = DecisionFeedbackEqualizer::new(c, 3, 0, 0.1).unwrap();
        assert!(eq.train(&[Complex::new(1.0, 0.0)], &[]).is_err());
        assert!(eq.feedback_taps().is_empty());
    }
}
