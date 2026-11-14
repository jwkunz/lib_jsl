//! Global optimisation of functions with many local minima, following `scipy.optimize`: basin
//! hopping (`basinhopping`), generalised simulated annealing with local search
//! (`dual_annealing`), and the deterministic DIRECT algorithm (`direct`, the locally biased
//! DIRECT-L variant that is SciPy's default).
//!
//! See the book entry [Assignment, trust regions and global optimisation](https://jwkunz.github.io/lib_jsl/book/optimization/assignment-trust-global.html#basin-hopping).

use crate::bounded::{LbfgsbOptions, minimize_lbfgsb};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};
use std::f64::consts::PI;

/// The result of a global optimisation.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GlobalResult {
    pub x: Vec<f64>,
    pub fun: f64,
    /// Objective evaluations, including those for finite-difference gradients.
    pub nfev: usize,
    pub iterations: usize,
}

/// Counts evaluations and applies SciPy's two-point finite differences (`h = √ε max(1, |xᵢ|)`,
/// stepping backwards at an upper bound).
struct Counted<'a> {
    f: &'a mut dyn FnMut(&[f64]) -> f64,
    nfev: usize,
}

impl Counted<'_> {
    fn eval(&mut self, x: &[f64]) -> f64 {
        self.nfev += 1;
        (self.f)(x)
    }
    fn value_and_gradient(&mut self, x: &[f64], ub: Option<&[f64]>) -> (f64, Vec<f64>) {
        let fx = self.eval(x);
        let mut g = vec![0.0; x.len()];
        let mut xh = x.to_vec();
        for i in 0..x.len() {
            let mut h = f64::EPSILON.sqrt() * x[i].abs().max(1.0) * if x[i] < 0.0 { -1.0 } else { 1.0 };
            if let Some(ub) = ub
                && x[i] + h > ub[i]
            {
                h = -h.abs();
            }
            xh[i] = x[i] + h;
            let dh = xh[i] - x[i];
            g[i] = (self.eval(&xh) - fx) / dh;
            xh[i] = x[i];
        }
        (fx, g)
    }
    /// L-BFGS-B from `x0` with finite-difference gradients.
    fn local(&mut self, x0: &[f64], lb: &[f64], ub: &[f64], max_iterations: usize) -> (Vec<f64>, f64) {
        let ub_owned = ub.to_vec();
        let bounded = ub.iter().any(|v| v.is_finite());
        let options = LbfgsbOptions { max_iterations, ..Default::default() };
        match minimize_lbfgsb(|x| self.value_and_gradient(x, if bounded { Some(&ub_owned) } else { None }), x0, lb, ub, options) {
            Ok(r) => (r.x, r.fun),
            Err(_) => (x0.to_vec(), f64::INFINITY),
        }
    }
}

fn standard_normal(rng: &mut UniformRNG) -> f64 {
    // Box-Muller.
    let u1 = 1.0 - rng.next_f64();
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

// ------------------------------------------------------------------------------- basin hopping

/// Options for [`basinhopping`] (SciPy's defaults).
#[derive(Clone, Copy, Debug)]
pub struct BasinhoppingOptions {
    pub iterations: usize,
    /// The Metropolis temperature `T`.
    pub temperature: f64,
    /// The initial maximum displacement of a random step.
    pub stepsize: f64,
    /// Adjust the step size every `interval` steps towards `target_accept_rate`.
    pub interval: usize,
    pub target_accept_rate: f64,
    pub stepwise_factor: f64,
    pub seed: u64,
}

impl Default for BasinhoppingOptions {
    fn default() -> Self {
        BasinhoppingOptions { iterations: 100, temperature: 1.0, stepsize: 0.5, interval: 50, target_accept_rate: 0.5, stepwise_factor: 0.9, seed: 0 }
    }
}

// ANCHOR: basinhopping
/// Basin hopping (Wales and Doye, 1997; SciPy's `basinhopping`): a random walk on the local
/// minima. Each step displaces the current minimum uniformly by up to `stepsize` in every
/// coordinate, minimises locally (L-BFGS-B with finite differences), and accepts the new minimum
/// by the Metropolis rule `exp(−(f_new − f_old)/T) > u`. Every `interval` steps the step size is
/// divided (or multiplied) by `stepwise_factor` if the acceptance rate is above (below) the
/// target. Returns the lowest minimum seen.
pub fn basinhopping(mut f: impl FnMut(&[f64]) -> f64, x0: &[f64], options: BasinhoppingOptions) -> GlobalResult {
    let n = x0.len();
    let (lb, ub) = (vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n]);
    let mut c = Counted { f: &mut f, nfev: 0 };
    let mut rng = UniformRNG::from_seed(options.seed);
    let (mut x, mut fx) = c.local(x0, &lb, &ub, 15000);
    let (mut best_x, mut best_f) = (x.clone(), fx);
    let mut stepsize = options.stepsize;
    let (mut nstep, mut naccept) = (0usize, 0usize);
    for _ in 0..options.iterations {
        let trial: Vec<f64> = x.iter().map(|v| v + stepsize * (2.0 * rng.next_f64() - 1.0)).collect();
        let (xn, fn_) = c.local(&trial, &lb, &ub, 15000);
        nstep += 1;
        let accept = fn_ < fx || (-(fn_ - fx) / options.temperature).exp() > rng.next_f64();
        if accept {
            naccept += 1;
            x = xn;
            fx = fn_;
            if fx < best_f {
                best_f = fx;
                best_x = x.clone();
            }
        }
        if nstep % options.interval == 0 {
            let rate = naccept as f64 / nstep as f64;
            if rate > options.target_accept_rate {
                stepsize /= options.stepwise_factor;
            } else {
                stepsize *= options.stepwise_factor;
            }
        }
    }
    GlobalResult { x: best_x, fun: best_f, nfev: c.nfev, iterations: options.iterations }
}
// ANCHOR_END: basinhopping

// ------------------------------------------------------------------------------- dual annealing

/// Options for [`dual_annealing`] (SciPy's defaults).
#[derive(Clone, Copy, Debug)]
pub struct DualAnnealingOptions {
    pub max_iterations: usize,
    pub initial_temp: f64,
    pub restart_temp_ratio: f64,
    /// The visiting distribution parameter `q_v ∈ (1, 3]`.
    pub visit: f64,
    /// The acceptance parameter `q_a < 1`.
    pub accept: f64,
    pub max_fun: usize,
    pub no_local_search: bool,
    pub seed: u64,
}

impl Default for DualAnnealingOptions {
    fn default() -> Self {
        DualAnnealingOptions { max_iterations: 1000, initial_temp: 5230.0, restart_temp_ratio: 2e-5, visit: 2.62, accept: -5.0, max_fun: 10_000_000, no_local_search: false, seed: 0 }
    }
}

/// The Tsallis visiting distribution of generalised simulated annealing.
struct Visiting {
    q: f64,
    lower: Vec<f64>,
    range: Vec<f64>,
    factor4_p: f64,
    factor6: f64,
}

impl Visiting {
    const TAIL_LIMIT: f64 = 1e8;
    const MIN_VISIT_BOUND: f64 = 1e-10;

    fn new(lower: &[f64], upper: &[f64], q: f64) -> Visiting {
        let factor2 = ((4.0 - q) * (q - 1.0).ln()).exp();
        let factor3 = ((2.0 - q) * 2f64.ln() / (q - 1.0)).exp();
        let factor4_p = PI.sqrt() * factor2 / (factor3 * (3.0 - q));
        let factor5 = 1.0 / (q - 1.0) - 0.5;
        let d1 = 2.0 - factor5;
        let factor6 = PI * (1.0 - factor5) / (PI * (1.0 - factor5)).sin() / lib_jsl_special_functions::ln_gamma(d1).exp();
        Visiting { q, lower: lower.to_vec(), range: upper.iter().zip(lower).map(|(u, l)| u - l).collect(), factor4_p, factor6 }
    }

    /// One draw of the distortion: Mantegna's ratio of normals scaled by the temperature.
    fn draw(&self, temperature: f64, rng: &mut UniformRNG) -> f64 {
        let (x, y) = (standard_normal(rng), standard_normal(rng));
        let factor1 = (temperature.ln() / (self.q - 1.0)).exp();
        let factor4 = self.factor4_p * factor1;
        let sigmax = (-(self.q - 1.0) * (self.factor6 / factor4).ln() / (3.0 - self.q)).exp();
        let den = ((self.q - 1.0) * y.abs().ln() / (3.0 - self.q)).exp();
        x * sigmax / den
    }

    fn wrap(&self, v: f64, i: usize) -> f64 {
        let a = v - self.lower[i];
        let b = a % self.range[i] + self.range[i];
        let mut w = b % self.range[i] + self.lower[i];
        if (w - self.lower[i]).abs() < Self::MIN_VISIT_BOUND {
            w += Self::MIN_VISIT_BOUND;
        }
        w
    }

    /// A new point: every coordinate moved for the first `n` steps of a chain, one coordinate at
    /// a time for the next `n`.
    fn visit(&self, x: &[f64], step: usize, temperature: f64, rng: &mut UniformRNG) -> Vec<f64> {
        let n = x.len();
        let clip = |v: f64, rng: &mut UniformRNG| if v > Self::TAIL_LIMIT { Self::TAIL_LIMIT * rng.next_f64() } else if v < -Self::TAIL_LIMIT { -Self::TAIL_LIMIT * rng.next_f64() } else { v };
        if step < n {
            let visits: Vec<f64> = (0..n).map(|_| self.draw(temperature, rng)).collect();
            visits.iter().enumerate().map(|(i, &v)| { let v = clip(v, rng); self.wrap(v + x[i], i) }).collect()
        } else {
            let mut out = x.to_vec();
            let i = step - n;
            let v = clip(self.draw(temperature, rng), rng);
            out[i] = self.wrap(v + x[i], i);
            out
        }
    }
}

// ANCHOR: dual_annealing
/// Dual annealing (Xiang et al., 1997; SciPy's `dual_annealing`): generalised simulated
/// annealing whose heavy-tailed Tsallis visiting distribution makes long jumps likely at high
/// temperature, with the temperature falling as `T(t) = T₀ (2^{q_v−1} − 1)/((1 + t)^{q_v−1} − 1)`,
/// acceptance of uphill moves with probability `[1 − (1 − q_a) ΔE/T]^{1/(1−q_a)}`, a local
/// L-BFGS-B search whenever the best point improves (or nothing has improved for a long time), and
/// re-annealing from a random point once the temperature falls below `restart_temp_ratio · T₀`.
pub fn dual_annealing(mut f: impl FnMut(&[f64]) -> f64, lower: &[f64], upper: &[f64], options: DualAnnealingOptions) -> Result<GlobalResult, ErrorsJSL> {
    let n = lower.len();
    if upper.len() != n || lower.iter().zip(upper).any(|(l, u)| u <= l || u.is_nan() || !l.is_finite() || !u.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("need finite bounds with lower < upper"));
    }
    let o = options;
    let mut c = Counted { f: &mut f, nfev: 0 };
    let mut rng = UniformRNG::from_seed(o.seed);
    let visiting = Visiting::new(lower, upper, o.visit);
    let ls_max_iter = (n * 6).clamp(100, 1000);
    let random_point = |rng: &mut UniformRNG| -> Vec<f64> { lower.iter().zip(upper).map(|(l, u)| l + (u - l) * rng.next_f64()).collect() };
    // Energy state.
    let mut current = random_point(&mut rng);
    let mut e_current = c.eval(&current);
    let mut reinit = 0;
    while !e_current.is_finite() {
        reinit += 1;
        if reinit > 1000 {
            return Err(ErrorsJSL::RuntimeError("could not find a finite starting value"));
        }
        current = random_point(&mut rng);
        e_current = c.eval(&current);
    }
    let (mut xbest, mut ebest) = (current.clone(), e_current);
    // Strategy chain state.
    let (mut xmin, mut emin) = (current.clone(), e_current);
    let mut not_improved = 0usize;
    let mut not_improved_max = 1000usize;
    let temperature_restart = o.initial_temp * o.restart_temp_ratio;
    let t1 = ((o.visit - 1.0) * 2f64.ln()).exp() - 1.0;
    let mut iteration = 0usize;
    'outer: loop {
        for i in 0..o.max_iterations {
            let s = i as f64 + 2.0;
            let t2 = ((o.visit - 1.0) * s.ln()).exp() - 1.0;
            let temperature = o.initial_temp * t1 / t2;
            if iteration >= o.max_iterations {
                break 'outer;
            }
            if temperature < temperature_restart {
                current = random_point(&mut rng);
                e_current = c.eval(&current);
                continue 'outer;
            }
            // One strategy chain.
            let temperature_step = temperature / (i as f64 + 1.0);
            not_improved += 1;
            let mut improved = i == 0;
            for j in 0..2 * n {
                let x_visit = visiting.visit(&current, j, temperature, &mut rng);
                let e = c.eval(&x_visit);
                if e < e_current {
                    current = x_visit;
                    e_current = e;
                    if e < ebest {
                        ebest = e;
                        xbest = current.clone();
                        improved = true;
                        not_improved = 0;
                    }
                } else {
                    let r = rng.next_f64();
                    let pqv_temp = 1.0 - (1.0 - o.accept) * (e - e_current) / temperature_step;
                    let pqv = if pqv_temp <= 0.0 { 0.0 } else { (pqv_temp.ln() / (1.0 - o.accept)).exp() };
                    if r <= pqv {
                        current = x_visit;
                        e_current = e;
                        xmin = current.clone();
                    }
                    if not_improved >= not_improved_max && (j == 0 || e_current < emin) {
                        emin = e_current;
                        xmin = current.clone();
                    }
                }
                if c.nfev >= o.max_fun {
                    break 'outer;
                }
            }
            if !o.no_local_search {
                if improved {
                    let (x, e) = c.local(&xbest, lower, upper, ls_max_iter);
                    if e < ebest && x.iter().all(|v| v.is_finite()) {
                        not_improved = 0;
                        ebest = e;
                        xbest = x.clone();
                        current = x;
                        e_current = e;
                    }
                    if c.nfev >= o.max_fun {
                        break 'outer;
                    }
                }
                if not_improved >= not_improved_max {
                    let (x, e) = c.local(&xmin, lower, upper, ls_max_iter);
                    let (x, e) = if e < emin { (x, e) } else { (xmin.clone(), emin) };
                    xmin = x.clone();
                    emin = e;
                    not_improved = 0;
                    not_improved_max = n;
                    if e < ebest {
                        ebest = e;
                        xbest = x.clone();
                        current = x;
                        e_current = e;
                    }
                    if c.nfev >= o.max_fun {
                        break 'outer;
                    }
                }
            }
            iteration += 1;
        }
    }
    Ok(GlobalResult { x: xbest, fun: ebest, nfev: c.nfev, iterations: iteration })
}
// ANCHOR_END: dual_annealing

// ------------------------------------------------------------------------------------- DIRECT

/// Options for [`direct`] (SciPy's defaults; `max_fun` defaults to `1000 n`).
#[derive(Clone, Copy, Debug)]
pub struct DirectOptions {
    /// The ε of the potential-optimality test.
    pub eps: f64,
    pub max_fun: Option<usize>,
    pub max_iterations: usize,
    /// Stop when the box holding the best point is smaller than this fraction of the domain.
    pub vol_tol: f64,
    /// Stop when half the longest side of that box (in the unit cube) is below this.
    pub len_tol: f64,
}

impl Default for DirectOptions {
    fn default() -> Self {
        DirectOptions { eps: 1e-4, max_fun: None, max_iterations: 1000, vol_tol: 1e-16, len_tol: 1e-6 }
    }
}

struct Box {
    centre: Vec<f64>,
    f: f64,
    /// Side `k` has length `3^{−levels[k]}` in the unit cube.
    levels: Vec<u32>,
}

impl Box {
    fn size(&self) -> f64 {
        3f64.powi(-(*self.levels.iter().min().expect("non-empty") as i32))
    }
}

// ANCHOR: direct
/// DIRECT, "DIviding RECTangles" (Jones, Perttunen and Stuckman, 1993), in Gablonsky and Kelley's
/// locally biased form DIRECT-L. The domain is scaled to the unit cube and covered by boxes, each
/// evaluated at its centre. Each iteration selects the *potentially optimal* boxes, those on the
/// lower-right convex hull of (size, value), one per size, whose value could be the lowest for some
/// Lipschitz constant `K` with a gain of at least `ε |f_min|`. Each selected box is trisected along
/// its longest sides, the side whose new centres give the lowest value first. No gradients, no
/// randomness: the search balances global exploration (large boxes) against local refinement.
pub fn direct(mut f: impl FnMut(&[f64]) -> f64, lower: &[f64], upper: &[f64], options: DirectOptions) -> Result<GlobalResult, ErrorsJSL> {
    let n = lower.len();
    if upper.len() != n || n == 0 || lower.iter().zip(upper).any(|(l, u)| u <= l || u.is_nan() || l.is_nan()) {
        return Err(ErrorsJSL::InvalidInputRange("need bounds with lower < upper"));
    }
    let o = options;
    let max_fun = o.max_fun.unwrap_or(1000 * n);
    let to_x = |c: &[f64]| -> Vec<f64> { c.iter().enumerate().map(|(i, v)| lower[i] + v * (upper[i] - lower[i])).collect() };
    let nfev = std::cell::Cell::new(0usize);
    let mut eval = |c: &[f64]| {
        nfev.set(nfev.get() + 1);
        f(&to_x(c))
    };
    let mut boxes = vec![Box { centre: vec![0.5; n], f: 0.0, levels: vec![0; n] }];
    boxes[0].f = eval(&boxes[0].centre);
    let mut best = 0usize;
    let mut iterations = 0;
    'outer: while iterations < o.max_iterations {
        iterations += 1;
        let fmin = boxes[best].f;
        // One candidate per size: the lowest value (DIRECT-L).
        let mut by_size: Vec<(f64, usize)> = Vec::new();
        for (i, b) in boxes.iter().enumerate() {
            let d = b.size();
            match by_size.iter_mut().find(|(s, _)| (*s - d).abs() <= 1e-12 * d) {
                Some(entry) => {
                    if b.f < boxes[entry.1].f {
                        entry.1 = i;
                    }
                }
                None => by_size.push((d, i)),
            }
        }
        by_size.sort_by(|a, b| a.0.total_cmp(&b.0));
        // From the lowest value (the largest such box) to the largest box, the lower convex hull.
        let start = (0..by_size.len()).min_by(|&a, &b| boxes[by_size[a].1].f.total_cmp(&boxes[by_size[b].1].f).then(by_size[b].0.total_cmp(&by_size[a].0))).expect("non-empty");
        let mut hull: Vec<usize> = Vec::new();
        for k in start..by_size.len() {
            let p = (by_size[k].0, boxes[by_size[k].1].f);
            while hull.len() >= 2 {
                let a = (by_size[hull[hull.len() - 2]].0, boxes[by_size[hull[hull.len() - 2]].1].f);
                let b = (by_size[hull[hull.len() - 1]].0, boxes[by_size[hull[hull.len() - 1]].1].f);
                // Remove b if it lies on or above the segment a-p.
                if (b.1 - a.1) * (p.0 - a.0) >= (p.1 - a.1) * (b.0 - a.0) {
                    hull.pop();
                } else {
                    break;
                }
            }
            if hull.len() == 1 && boxes[by_size[hull[0]].1].f <= p.1 && hull[0] != start {
                hull.pop();
            }
            hull.push(k);
        }
        let mut selected = Vec::new();
        for (h, &k) in hull.iter().enumerate() {
            let (d, fj) = (by_size[k].0, boxes[by_size[k].1].f);
            let ok = match hull.get(h + 1) {
                Some(&next) => {
                    let slope = (boxes[by_size[next].1].f - fj) / (by_size[next].0 - d);
                    fj - slope * d <= fmin - o.eps * fmin.abs()
                }
                None => true,
            };
            if ok {
                selected.push(by_size[k].1);
            }
        }
        for idx in selected {
            if nfev.get() >= max_fun {
                break 'outer;
            }
            let lmin = *boxes[idx].levels.iter().min().expect("non-empty");
            let dims: Vec<usize> = (0..n).filter(|&k| boxes[idx].levels[k] == lmin).collect();
            let delta = 3f64.powi(-(lmin as i32 + 1));
            // (best of the pair, dimension, centre +, value +, centre −, value −)
            type Trial = (f64, usize, Vec<f64>, f64, Vec<f64>, f64);
            let mut trials: Vec<Trial> = Vec::new();
            for &k in &dims {
                let mut cp = boxes[idx].centre.clone();
                cp[k] += delta;
                let fp = eval(&cp);
                let mut cm = boxes[idx].centre.clone();
                cm[k] -= delta;
                let fm = eval(&cm);
                trials.push((fp.min(fm), k, cp, fp, cm, fm));
            }
            trials.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let mut levels = boxes[idx].levels.clone();
            for (_, k, cp, fp, cm, fm) in trials {
                levels[k] += 1;
                boxes.push(Box { centre: cp, f: fp, levels: levels.clone() });
                boxes.push(Box { centre: cm, f: fm, levels: levels.clone() });
            }
            boxes[idx].levels = levels;
        }
        best = (0..boxes.len()).min_by(|&a, &b| boxes[a].f.total_cmp(&boxes[b].f)).expect("non-empty");
        let bb = &boxes[best];
        let volume: f64 = bb.levels.iter().map(|&l| 3f64.powi(-(l as i32))).product();
        if volume < o.vol_tol || 0.5 * bb.size() < o.len_tol || nfev.get() >= max_fun {
            break;
        }
    }
    let b = &boxes[best];
    Ok(GlobalResult { x: to_x(&b.centre), fun: b.f, nfev: nfev.get(), iterations })
}
// ANCHOR_END: direct

#[cfg(test)]
mod tests {
    use super::*;

    fn rastrigin(x: &[f64]) -> f64 {
        10.0 * x.len() as f64 + x.iter().map(|v| v * v - 10.0 * (2.0 * PI * v).cos()).sum::<f64>()
    }
    fn camel(x: &[f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        (4.0 - 2.1 * a * a + a.powi(4) / 3.0) * a * a + a * b + (-4.0 + 4.0 * b * b) * b * b
    }

    #[test]
    fn dual_annealing_finds_the_rastrigin_minimum() {
        let lb = vec![-5.12; 4];
        let ub = vec![5.12; 4];
        let r = dual_annealing(rastrigin, &lb, &ub, DualAnnealingOptions { seed: 3, ..Default::default() }).unwrap();
        assert!(r.fun < 1e-8, "{r:?}");
    }

    #[test]
    fn direct_and_basinhopping_find_the_six_hump_camel_minimum() {
        let r = direct(camel, &[-3.0, -2.0], &[3.0, 2.0], DirectOptions::default()).unwrap();
        assert!((r.fun + 1.031628453489877).abs() < 1e-4, "{r:?}");
        let r = basinhopping(camel, &[2.0, 1.5], BasinhoppingOptions { seed: 1, iterations: 50, ..Default::default() });
        assert!((r.fun + 1.031628453489877).abs() < 1e-8, "{r:?}");
    }
}
