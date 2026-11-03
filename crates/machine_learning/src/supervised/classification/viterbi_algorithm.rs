//! The Viterbi algorithm and hidden Markov models, with Baum-Welch training.
//!
//! See the book entry [Hidden Markov models: Viterbi, forward-backward and Baum-Welch](https://jwkunz.github.io/lib_jsl/book/learning/hidden-markov-models.html).

use crate::common::MlRng;
use lib_jsl_core::{ErrorsJSL, R2D};

/// The most likely state path through a trellis, by dynamic programming in log space.
///
/// `log_initial[s]` scores starting in state `s`; `log_transition(from, to)` scores a step;
/// `log_emission(state, t)` scores state `state` explaining observation `t`. Use
/// `f64::NEG_INFINITY` for impossible events. Returns the best path of length `steps` and its
/// total log score (`-inf` if every path is impossible, in which case the path is all zeros).
pub fn viterbi_path(
    log_initial: &[f64],
    steps: usize,
    log_transition: impl Fn(usize, usize) -> f64,
    log_emission: impl Fn(usize, usize) -> f64,
) -> (Vec<usize>, f64) {
    let n = log_initial.len();
    if n == 0 || steps == 0 {
        return (Vec::new(), 0.0);
    }
    // ANCHOR: viterbi
    let mut score: Vec<f64> = (0..n).map(|s| log_initial[s] + log_emission(s, 0)).collect();
    let mut back: Vec<Vec<usize>> = Vec::with_capacity(steps);
    for t in 1..steps {
        let mut next = vec![f64::NEG_INFINITY; n];
        let mut from = vec![0usize; n];
        for to in 0..n {
            let (best_from, best) = (0..n).map(|f| (f, score[f] + log_transition(f, to))).fold((0, f64::NEG_INFINITY), |b, c| if c.1 > b.1 { c } else { b });
            next[to] = best + log_emission(to, t);
            from[to] = best_from;
        }
        back.push(from);
        score = next;
    }
    let (mut state, best) = score.iter().enumerate().fold((0, f64::NEG_INFINITY), |b, (s, &v)| if v > b.1 { (s, v) } else { b });
    let mut path = vec![0; steps];
    path[steps - 1] = state;
    for t in (1..steps).rev() {
        state = back[t - 1][state];
        path[t - 1] = state;
    }
    // ANCHOR_END: viterbi
    (path, best)
}

/// A discrete hidden Markov model: hidden states with start probabilities, a transition matrix and
/// an emission matrix over a finite alphabet of observation symbols (`0..symbols`).
#[derive(Debug, Clone)]
pub struct HiddenMarkovModel {
    initial: Vec<f64>,
    transition: R2D,
    emission: R2D,
}

fn log(p: f64) -> f64 {
    if p > 0.0 { p.ln() } else { f64::NEG_INFINITY }
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let m = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY { m } else { m + values.iter().map(|v| (v - m).exp()).sum::<f64>().ln() }
}

impl HiddenMarkovModel {
    /// `initial[s]`, `transition[[from, to]]` and `emission[[state, symbol]]` must be probability
    /// distributions (each row sums to one).
    pub fn new(initial: Vec<f64>, transition: R2D, emission: R2D) -> Result<Self, ErrorsJSL> {
        let n = initial.len();
        if n == 0 || transition.dim() != (n, n) || emission.nrows() != n || emission.ncols() == 0 {
            return Err(ErrorsJSL::IncompatibleArraySizes((n, transition.nrows())));
        }
        let is_dist = |row: &[f64]| row.iter().all(|&p| (0.0..=1.0).contains(&p)) && (row.iter().sum::<f64>() - 1.0).abs() < 1e-9;
        if !is_dist(&initial) || !transition.rows().into_iter().all(|r| is_dist(&r.to_vec())) || !emission.rows().into_iter().all(|r| is_dist(&r.to_vec())) {
            return Err(ErrorsJSL::InvalidInputRange("Initial, transition and emission rows must be probability distributions."));
        }
        Ok(HiddenMarkovModel { initial, transition, emission })
    }

    pub fn states(&self) -> usize {
        self.initial.len()
    }

    fn check(&self, observations: &[usize]) -> Result<(), ErrorsJSL> {
        if observations.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("The observation sequence is empty."));
        }
        if observations.iter().any(|&o| o >= self.emission.ncols()) {
            return Err(ErrorsJSL::InvalidInputRange("An observation is outside the symbol alphabet."));
        }
        Ok(())
    }

    /// The most likely hidden state sequence for the observations and its log probability.
    pub fn viterbi(&self, observations: &[usize]) -> Result<(Vec<usize>, f64), ErrorsJSL> {
        self.check(observations)?;
        let log_init: Vec<f64> = self.initial.iter().map(|&p| log(p)).collect();
        Ok(viterbi_path(&log_init, observations.len(), |f, t| log(self.transition[[f, t]]), |s, t| log(self.emission[[s, observations[t]]])))
    }

    /// The log-likelihood `log P(observations)` summed over all state paths (forward algorithm).
    pub fn log_likelihood(&self, observations: &[usize]) -> Result<f64, ErrorsJSL> {
        self.check(observations)?;
        let n = self.states();
        let mut alpha: Vec<f64> = (0..n).map(|s| log(self.initial[s]) + log(self.emission[[s, observations[0]]])).collect();
        for &o in &observations[1..] {
            alpha = (0..n).map(|to| log_sum_exp(&(0..n).map(|f| alpha[f] + log(self.transition[[f, to]])).collect::<Vec<_>>()) + log(self.emission[[to, o]])).collect();
        }
        Ok(log_sum_exp(&alpha))
    }

    #[allow(clippy::needless_range_loop)]
    /// Posterior state probabilities `P(state_t | all observations)` by forward-backward, as a
    /// `(steps, states)` matrix (each row sums to one).
    pub fn posteriors(&self, observations: &[usize]) -> Result<R2D, ErrorsJSL> {
        self.check(observations)?;
        let (n, steps) = (self.states(), observations.len());
        let mut alpha = vec![vec![f64::NEG_INFINITY; n]; steps];
        let mut beta = vec![vec![0.0; n]; steps];
        for s in 0..n {
            alpha[0][s] = log(self.initial[s]) + log(self.emission[[s, observations[0]]]);
        }
        for t in 1..steps {
            for to in 0..n {
                alpha[t][to] = log_sum_exp(&(0..n).map(|f| alpha[t - 1][f] + log(self.transition[[f, to]])).collect::<Vec<_>>()) + log(self.emission[[to, observations[t]]]);
            }
        }
        for t in (0..steps - 1).rev() {
            for f in 0..n {
                beta[t][f] = log_sum_exp(&(0..n).map(|to| log(self.transition[[f, to]]) + log(self.emission[[to, observations[t + 1]]]) + beta[t + 1][to]).collect::<Vec<_>>());
            }
        }
        let mut out = R2D::zeros((steps, n));
        for t in 0..steps {
            let row: Vec<f64> = (0..n).map(|s| alpha[t][s] + beta[t][s]).collect();
            let z = log_sum_exp(&row);
            for s in 0..n {
                out[[t, s]] = (row[s] - z).exp();
            }
        }
        Ok(out)
    }
}

/// The outcome of Baum-Welch training.
#[derive(Debug, Clone)]
pub struct BaumWelchResult {
    pub model: HiddenMarkovModel,
    /// Total log-likelihood of the training sequences at the start of each iteration, before its
    /// re-estimation (non-decreasing). The returned `model` has been re-estimated once more, so its
    /// likelihood is at least the last entry.
    pub log_likelihood: Vec<f64>,
    pub converged: bool,
}

impl HiddenMarkovModel {
    pub fn initial(&self) -> &[f64] {
        &self.initial
    }

    pub fn transition(&self) -> &R2D {
        &self.transition
    }

    pub fn emission(&self) -> &R2D {
        &self.emission
    }

    /// A random model with `states` states and `symbols` symbols (rows drawn uniformly from the
    /// simplex and kept away from zero), as a starting point for [`HiddenMarkovModel::baum_welch`].
    pub fn random(states: usize, symbols: usize, seed: u64) -> Result<Self, ErrorsJSL> {
        if states == 0 || symbols == 0 {
            return Err(ErrorsJSL::InvalidInputRange("Need at least one state and one symbol."));
        }
        let mut rng = MlRng::new(seed);
        let mut row = |k: usize| {
            let v: Vec<f64> = (0..k).map(|_| 0.5 + rng.uniform()).collect();
            let total: f64 = v.iter().sum();
            v.into_iter().map(|p| p / total).collect::<Vec<f64>>()
        };
        let initial = row(states);
        let transition = R2D::from_shape_vec((states, states), (0..states).flat_map(|_| row(states)).collect()).expect("shape");
        let emission = R2D::from_shape_vec((states, symbols), (0..states).flat_map(|_| row(symbols)).collect()).expect("shape");
        HiddenMarkovModel::new(initial, transition, emission)
    }

    /// Samples a hidden state path and its observations of the given length.
    pub fn sample(&self, steps: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
        let mut rng = MlRng::new(seed);
        let mut draw = |p: &mut dyn Iterator<Item = f64>| {
            let u = rng.uniform();
            let mut acc = 0.0;
            let mut last = 0;
            for (i, pi) in p.enumerate() {
                acc += pi;
                last = i;
                if u < acc {
                    return i;
                }
            }
            last
        };
        let (mut states, mut symbols) = (Vec::with_capacity(steps), Vec::with_capacity(steps));
        for t in 0..steps {
            let s = if t == 0 { draw(&mut self.initial.iter().copied()) } else { draw(&mut self.transition.row(states[t - 1]).iter().copied()) };
            states.push(s);
            symbols.push(draw(&mut self.emission.row(s).iter().copied()));
        }
        (states, symbols)
    }

    /// Scaled forward-backward pass: returns `α̂`, `β̂` (rescaled per step) and the scale factors
    /// `c_t`, with `log P(O) = Σ log c_t`.
    #[allow(clippy::type_complexity, clippy::needless_range_loop)]
    fn scaled_forward_backward(&self, o: &[usize]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
        let (n, steps) = (self.states(), o.len());
        let mut alpha = vec![vec![0.0; n]; steps];
        let mut scale = vec![0.0; steps];
        for s in 0..n {
            alpha[0][s] = self.initial[s] * self.emission[[s, o[0]]];
        }
        for t in 0..steps {
            if t > 0 {
                for to in 0..n {
                    alpha[t][to] = (0..n).map(|f| alpha[t - 1][f] * self.transition[[f, to]]).sum::<f64>() * self.emission[[to, o[t]]];
                }
            }
            scale[t] = alpha[t].iter().sum::<f64>();
            let c = if scale[t] > 0.0 { scale[t] } else { 1.0 };
            alpha[t].iter_mut().for_each(|a| *a /= c);
        }
        let mut beta = vec![vec![1.0; n]; steps];
        for t in (0..steps - 1).rev() {
            let c = if scale[t + 1] > 0.0 { scale[t + 1] } else { 1.0 };
            for f in 0..n {
                beta[t][f] = (0..n).map(|to| self.transition[[f, to]] * self.emission[[to, o[t + 1]]] * beta[t + 1][to]).sum::<f64>() / c;
            }
        }
        (alpha, beta, scale)
    }

    /// Fits the model to observation sequences by the Baum-Welch (expectation-maximisation)
    /// algorithm, starting from `self`. Each iteration runs a scaled forward-backward pass per
    /// sequence, accumulates expected transition and emission counts, and re-estimates the
    /// parameters; the likelihood never decreases. Stops when the log-likelihood improves by less
    /// than `tolerance` (relative) or after `max_iterations`. EM finds a local optimum, so several
    /// random starts ([`HiddenMarkovModel::random`]) may be worth trying.
    #[allow(clippy::needless_range_loop)]
    pub fn baum_welch(&self, sequences: &[Vec<usize>], max_iterations: usize, tolerance: f64) -> Result<BaumWelchResult, ErrorsJSL> {
        if sequences.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("Need at least one training sequence."));
        }
        for o in sequences {
            self.check(o)?;
        }
        let (n, m) = (self.states(), self.emission.ncols());
        let mut model = self.clone();
        let mut history = Vec::new();
        let mut converged = false;
        for _ in 0..max_iterations {
            let mut initial = vec![0.0; n];
            let mut transitions = R2D::zeros((n, n));
            let mut emissions = R2D::zeros((n, m));
            let mut log_likelihood = 0.0;
            // ANCHOR: baum_welch
            for o in sequences {
                let (alpha, beta, scale) = model.scaled_forward_backward(o);
                if scale.iter().any(|&c| c <= 0.0) {
                    return Err(ErrorsJSL::RuntimeError("A training sequence has zero probability under the model."));
                }
                log_likelihood += scale.iter().map(|c| c.ln()).sum::<f64>();
                for t in 0..o.len() {
                    // γ_t(s) = α̂_t(s) β̂_t(s) (already normalised by the scaling).
                    for s in 0..n {
                        let gamma = alpha[t][s] * beta[t][s];
                        if t == 0 {
                            initial[s] += gamma;
                        }
                        emissions[[s, o[t]]] += gamma;
                    }
                    if t + 1 < o.len() {
                        for f in 0..n {
                            for to in 0..n {
                                transitions[[f, to]] += alpha[t][f] * model.transition[[f, to]] * model.emission[[to, o[t + 1]]] * beta[t + 1][to] / scale[t + 1];
                            }
                        }
                    }
                }
            }
            // ANCHOR_END: baum_welch
            let improved = history.last().is_none_or(|&last: &f64| log_likelihood - last > tolerance * last.abs().max(1.0));
            history.push(log_likelihood);
            // M-step: normalise the expected counts (rows with no mass keep their old values).
            let total: f64 = initial.iter().sum();
            let new_initial: Vec<f64> = initial.iter().map(|v| v / total).collect();
            let normalise = |counts: &R2D, old: &R2D| {
                let mut out = counts.clone();
                for (r, mut row) in out.rows_mut().into_iter().enumerate() {
                    let s: f64 = row.sum();
                    if s > 0.0 {
                        row.mapv_inplace(|v| v / s);
                    } else {
                        row.assign(&old.row(r));
                    }
                }
                out
            };
            model = HiddenMarkovModel { initial: new_initial, transition: normalise(&transitions, &model.transition), emission: normalise(&emissions, &model.emission) };
            if !improved {
                converged = true;
                break;
            }
        }
        Ok(BaumWelchResult { model, log_likelihood: history, converged })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// The textbook Healthy/Fever model; symbols 0 = normal, 1 = cold, 2 = dizzy.
    fn clinic() -> HiddenMarkovModel {
        HiddenMarkovModel::new(vec![0.6, 0.4], arr2(&[[0.7, 0.3], [0.4, 0.6]]), arr2(&[[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])).unwrap()
    }

    #[test]
    fn textbook_viterbi_path_and_probability() {
        let (path, logp) = clinic().viterbi(&[0, 1, 2]).unwrap();
        assert_eq!(path, vec![0, 0, 1]); // Healthy, Healthy, Fever
        assert!((logp.exp() - 0.01512).abs() < 1e-12);
    }

    #[test]
    fn viterbi_agrees_with_brute_force_and_forward_is_a_sum_over_paths() {
        let m = clinic();
        let obs = [2, 2, 0, 1, 2];
        let n = m.states();
        let mut best = (vec![], f64::NEG_INFINITY);
        let mut total = 0.0;
        for code in 0..n.pow(obs.len() as u32) {
            let path: Vec<usize> = (0..obs.len()).map(|t| (code / n.pow(t as u32)) % n).collect();
            let mut p = m.initial[path[0]] * m.emission[[path[0], obs[0]]];
            for t in 1..obs.len() {
                p *= m.transition[[path[t - 1], path[t]]] * m.emission[[path[t], obs[t]]];
            }
            total += p;
            if p.ln() > best.1 {
                best = (path, p.ln());
            }
        }
        let (path, logp) = m.viterbi(&obs).unwrap();
        assert_eq!(path, best.0);
        assert!((logp - best.1).abs() < 1e-12);
        assert!((m.log_likelihood(&obs).unwrap() - total.ln()).abs() < 1e-12);
    }

    #[test]
    fn posteriors_are_distributions_and_agree_with_intuition() {
        let m = clinic();
        let post = m.posteriors(&[2, 2, 2, 0, 0, 0]).unwrap();
        assert!(post.rows().into_iter().all(|r| (r.sum() - 1.0).abs() < 1e-12));
        assert!(post[[1, 1]] > 0.5 && post[[5, 0]] > 0.5); // dizzy streak -> Fever, normal streak -> Healthy
        // A single observation: posterior ∝ initial × emission.
        let one = m.posteriors(&[1]).unwrap();
        let (a, b) = (0.6 * 0.4, 0.4 * 0.3);
        assert!((one[[0, 0]] - a / (a + b)).abs() < 1e-12);
    }

    #[test]
    fn impossible_events_and_validation() {
        // State 0 can never emit symbol 1, and state 1 never emits symbol 0.
        let m = HiddenMarkovModel::new(vec![1.0, 0.0], arr2(&[[0.5, 0.5], [0.0, 1.0]]), arr2(&[[1.0, 0.0], [0.0, 1.0]])).unwrap();
        assert_eq!(m.viterbi(&[0, 0, 1, 1]).unwrap().0, vec![0, 0, 1, 1]);
        let (_, lp) = m.viterbi(&[1, 0]).unwrap();
        assert_eq!(lp, f64::NEG_INFINITY);
        assert!(m.viterbi(&[]).is_err() && m.viterbi(&[5]).is_err());
        assert!(HiddenMarkovModel::new(vec![0.5, 0.4], arr2(&[[0.5, 0.5], [0.5, 0.5]]), arr2(&[[1.0], [1.0]])).is_err());
        assert!(HiddenMarkovModel::new(vec![1.0], arr2(&[[0.5, 0.5]]), arr2(&[[1.0]])).is_err());
        // The generic trellis function handles a degenerate empty input.
        assert_eq!(viterbi_path(&[], 3, |_, _| 0.0, |_, _| 0.0), (vec![], 0.0));
    }

    #[test]
    fn baum_welch_recovers_a_generating_model() {
        let truth = HiddenMarkovModel::new(vec![0.8, 0.2], arr2(&[[0.9, 0.1], [0.2, 0.8]]), arr2(&[[0.7, 0.2, 0.1], [0.05, 0.25, 0.7]])).unwrap();
        let sequences: Vec<Vec<usize>> = (0..20).map(|k| truth.sample(500, k).1).collect();
        let start = HiddenMarkovModel::random(2, 3, 11).unwrap();
        let fit = start.baum_welch(&sequences, 500, 1e-10).unwrap();
        assert!(fit.converged);
        // EM never decreases the likelihood.
        assert!(fit.log_likelihood.windows(2).all(|w| w[1] >= w[0] - 1e-6), "{:?}", &fit.log_likelihood[..5]);
        // The returned model is at least as good as the last recorded likelihood.
        let final_ll: f64 = sequences.iter().map(|o| fit.model.log_likelihood(o).unwrap()).sum();
        assert!(final_ll >= fit.log_likelihood.last().unwrap() - 1e-6);
        // Match the learned states to the true ones (EM may swap labels).
        let m = &fit.model;
        let swap = m.emission()[[0, 2]] > m.emission()[[1, 2]];
        let perm = |s: usize| if swap { 1 - s } else { s };
        for f in 0..2 {
            for t in 0..2 {
                assert!((m.transition()[[perm(f), perm(t)]] - truth.transition()[[f, t]]).abs() < 0.03, "transition {f}->{t}");
            }
            for o in 0..3 {
                assert!((m.emission()[[perm(f), o]] - truth.emission()[[f, o]]).abs() < 0.03, "emission {f}:{o}");
            }
        }
        // The fitted model explains the data at least as well as the truth (it is the MLE).
        let total = |h: &HiddenMarkovModel| sequences.iter().map(|o| h.log_likelihood(o).unwrap()).sum::<f64>();
        assert!(total(m) >= total(&truth) - 1e-6);
        assert!((total(m) - fit.log_likelihood.last().unwrap()).abs() < 1.0);
        assert!(start.baum_welch(&[], 5, 1e-6).is_err() && start.baum_welch(&[vec![5]], 5, 1e-6).is_err());
    }
}
