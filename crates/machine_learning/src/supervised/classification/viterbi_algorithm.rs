//! The Viterbi algorithm and hidden Markov models.

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
}
