//! Integrals with a known difficult weight, the cases of SciPy's `quad(weight=…)` (QUADPACK's
//! QAWO, QAWF, QAWS and QAWC): oscillatory `cos ωx` and `sin ωx` on finite and semi-infinite
//! intervals, algebraic-logarithmic endpoint singularities `(x − a)^α (b − x)^β [log]`, and Cauchy
//! principal values `⨍ f(x)/(x − c) dx`. Each uses a method suited to the weight rather than a
//! port of QUADPACK's modified Clenshaw-Curtis moments; the results are compared with SciPy in the
//! book.
//!
//! See the book entry [Vector integrands and weighted quadrature](https://jwkunz.github.io/lib_jsl/book/integration/vector-weighted.html#weighted-integrals).

use crate::quad_vec::{QuadVecOptions, quad_vec};
use crate::tanh_sinh::TanhSinhEngine;
use lib_jsl_core::ErrorsJSL;
use std::f64::consts::PI;

/// Which logarithmic factors an algebraic weight carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LogFactor {
    /// `(x − a)^α (b − x)^β` (SciPy's `'alg'`).
    None,
    /// Times `ln(x − a)` (`'alg-loga'`).
    Left,
    /// Times `ln(b − x)` (`'alg-logb'`).
    Right,
    /// Times `ln(x − a) ln(b − x)` (`'alg-log'`).
    Both,
}

/// The weight `w(x)` in `∫ f(x) w(x) dx`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Weight {
    /// `cos ωx`; `b` may be `+∞` (Fourier integral).
    Cos(f64),
    /// `sin ωx`; `b` may be `+∞`.
    Sin(f64),
    /// `(x − a)^α (b − x)^β`, possibly with logarithms, `α, β > −1`, on a finite interval.
    Algebraic { alpha: f64, beta: f64, log: LogFactor },
    /// `1/(x − c)` with `a < c < b`: the Cauchy principal value.
    Cauchy(f64),
}

/// Tolerances for [`quad_weighted`] (SciPy's `quad` defaults).
#[derive(Clone, Copy, Debug)]
pub struct WeightedOptions {
    pub epsabs: f64,
    pub epsrel: f64,
    /// The most half-periods summed for a Fourier integral on `[a, ∞)`.
    pub max_cycles: usize,
}

impl Default for WeightedOptions {
    fn default() -> Self {
        WeightedOptions { epsabs: 1.49e-8, epsrel: 1.49e-8, max_cycles: 500 }
    }
}

/// The result of [`quad_weighted`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct WeightedResult {
    pub value: f64,
    pub error: f64,
    pub nfev: usize,
    pub converged: bool,
}

fn scalar(f: &mut dyn FnMut(f64) -> f64, a: f64, b: f64, points: Vec<f64>, o: &WeightedOptions) -> Result<(f64, f64, usize, bool), ErrorsJSL> {
    let opts = QuadVecOptions { epsabs: o.epsabs.max(1e-200), epsrel: o.epsrel, points, ..Default::default() };
    let r = quad_vec(|x| vec![f(x)], a, b, &opts)?;
    Ok((r.value[0], r.error, r.nfev, r.converged))
}

// ANCHOR: wynn
/// Wynn's ε algorithm: the extrapolated limit of a sequence of partial sums, from its last
/// entries, with an error estimate from the change between successive extrapolations.
fn wynn_epsilon(sums: &[f64]) -> (f64, f64) {
    let n = sums.len();
    if n < 3 {
        let last = sums[n - 1];
        return (last, if n > 1 { (last - sums[n - 2]).abs() } else { f64::INFINITY });
    }
    // e[k][j]: the ε table, column k; even columns are estimates of the limit.
    let mut prev: Vec<f64> = vec![0.0; n + 1];
    let mut cur: Vec<f64> = sums.to_vec();
    let mut best = (sums[n - 1], (sums[n - 1] - sums[n - 2]).abs());
    let mut k = 0;
    while cur.len() > 1 {
        let next: Vec<f64> = (0..cur.len() - 1)
            .map(|j| {
                let d = cur[j + 1] - cur[j];
                if d == 0.0 { f64::INFINITY } else { prev[j + 1] + 1.0 / d }
            })
            .collect();
        k += 1;
        if k % 2 == 0 && next.len() >= 2 {
            let (a, b) = (next[next.len() - 1], next[next.len() - 2]);
            if a.is_finite() && b.is_finite() {
                let err = (a - b).abs();
                if err < best.1 {
                    best = (a, err);
                }
            }
        }
        prev = cur;
        cur = next;
    }
    best
}
// ANCHOR_END: wynn

// ANCHOR: quad_weighted
/// `∫_a^b f(x) w(x) dx` for the weights of SciPy's `quad(weight=…)`:
///
/// - `Cos(ω)`, `Sin(ω)` on `[a, b]`: adaptive Gauss-Kronrod on `f(x)w(x)` with the interval split
///   at every half-period `π/|ω|`; on `[a, ∞)`, the integrals over successive half-periods
///   form an alternating series whose limit is extrapolated by Wynn's ε algorithm (QAWF's idea).
/// - `Algebraic`: tanh-sinh quadrature of `f(x)w(x)`, which tolerates the integrable endpoint
///   singularities; the endpoint distances are formed exactly, so `(b − x)^β` loses nothing near `b`.
/// - `Cauchy(c)`: with `d = min(c − a, b − c)`, the principal value is
///   `∫₀^d (f(c + t) − f(c − t))/t dt` (a smooth integrand) plus the ordinary integral of
///   `f(x)/(x − c)` over the part of `[a, b]` outside `[c − d, c + d]`.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // the negations also reject NaN
pub fn quad_weighted(mut f: impl FnMut(f64) -> f64, a: f64, b: f64, weight: Weight, options: &WeightedOptions) -> Result<WeightedResult, ErrorsJSL> {
    if a.is_nan() || b.is_nan() || !a.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("quad_weighted: a must be finite"));
    }
    match weight {
        Weight::Cos(omega) | Weight::Sin(omega) => {
            let cosine = matches!(weight, Weight::Cos(_));
            let w = move |x: f64| if cosine { (omega * x).cos() } else { (omega * x).sin() };
            if b.is_finite() {
                if omega == 0.0 {
                    let (v, e, n, c) = scalar(&mut |x| f(x) * w(x), a, b, Vec::new(), options)?;
                    return Ok(WeightedResult { value: v, error: e, nfev: n, converged: c });
                }
                let half = PI / omega.abs();
                let (lo, hi) = (a.min(b), a.max(b));
                let pieces = ((hi - lo) / half).ceil() as usize;
                let points: Vec<f64> = (1..pieces).map(|k| lo + k as f64 * half).collect();
                let (v, e, n, c) = scalar(&mut |x| f(x) * w(x), a, b, points, options)?;
                return Ok(WeightedResult { value: v, error: e, nfev: n, converged: c });
            }
            if b != f64::INFINITY || omega == 0.0 {
                return Err(ErrorsJSL::InvalidInputRange("quad_weighted: a Fourier integral needs b = +∞ and ω ≠ 0"));
            }
            // Half-periods from a, aligned to the zeros of the weight after the first.
            let half = PI / omega.abs();
            let phase0 = if cosine { half / 2.0 } else { 0.0 };
            let first_zero = ((a - phase0) / half).floor() * half + phase0 + half;
            let mut edges = vec![a, first_zero];
            let mut sums = Vec::new();
            let mut total = 0.0;
            let mut nfev = 0;
            let mut err_sum = 0.0;
            let mut best = (0.0, f64::INFINITY);
            for k in 0..options.max_cycles {
                let (x0, x1) = (edges[k], edges[k + 1]);
                let (v, e, n, _) = scalar(&mut |x| f(x) * w(x), x0, x1, Vec::new(), options)?;
                total += v;
                err_sum += e;
                nfev += n;
                sums.push(total);
                edges.push(x1 + half);
                if sums.len() >= 4 {
                    let start = sums.len().saturating_sub(40);
                    best = wynn_epsilon(&sums[start..]);
                    let tol = options.epsabs.max(options.epsrel * best.0.abs());
                    if best.1 <= tol && sums.len() >= 8 {
                        return Ok(WeightedResult { value: best.0, error: best.1 + err_sum, nfev, converged: true });
                    }
                }
            }
            Ok(WeightedResult { value: best.0, error: best.1 + err_sum, nfev, converged: false })
        }
        Weight::Algebraic { alpha, beta, log } => {
            if !b.is_finite() || !(b > a) || !(alpha > -1.0) || !(beta > -1.0) {
                return Err(ErrorsJSL::InvalidInputRange("quad_weighted: algebraic weights need a < b finite and α, β > −1"));
            }
            let width = b - a;
            let mut engine = TanhSinhEngine::new(a, b);
            engine.tolerance = options.epsrel.max(1e-15);
            engine.max_levels = 12;
            let calls = std::cell::Cell::new(0usize);
            let fcell = std::cell::RefCell::new(&mut f);
            // `c` is x − a in the lower half and x − b in the upper half, exactly.
            let (value, error) = engine.integrate_with_complement(|x, c| {
                calls.set(calls.get() + 1);
                let (da, db) = if c >= 0.0 && x - a <= b - x { (c, width - c) } else { (width + c, -c) };
                if da <= 0.0 || db <= 0.0 {
                    return 0.0;
                }
                let mut wv = da.powf(alpha) * db.powf(beta);
                match log {
                    LogFactor::None => {}
                    LogFactor::Left => wv *= da.ln(),
                    LogFactor::Right => wv *= db.ln(),
                    LogFactor::Both => wv *= da.ln() * db.ln(),
                }
                (fcell.borrow_mut())(x) * wv
            })?;
            let tol = options.epsabs.max(options.epsrel * value.abs());
            Ok(WeightedResult { value, error, nfev: calls.get(), converged: error <= tol.max(1e-14 * value.abs()) })
        }
        Weight::Cauchy(c) => {
            if !b.is_finite() || !(a < c && c < b) {
                return Err(ErrorsJSL::InvalidInputRange("quad_weighted: a Cauchy weight needs a < c < b, finite"));
            }
            let d = (c - a).min(b - c);
            let (v1, e1, n1, ok1) = scalar(&mut |t| (f(c + t) - f(c - t)) / t, 0.0, d, Vec::new(), options)?;
            let (mut v, mut e, mut n, mut ok) = (v1, e1, n1, ok1);
            if c + d < b {
                let (v2, e2, n2, ok2) = scalar(&mut |x| f(x) / (x - c), c + d, b, Vec::new(), options)?;
                v += v2;
                e += e2;
                n += n2;
                ok &= ok2;
            }
            if c - d > a {
                let (v2, e2, n2, ok2) = scalar(&mut |x| f(x) / (x - c), a, c - d, Vec::new(), options)?;
                v += v2;
                e += e2;
                n += n2;
                ok &= ok2;
            }
            Ok(WeightedResult { value: v, error: e, nfev: n, converged: ok })
        }
    }
}
// ANCHOR_END: quad_weighted

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_forms() {
        let o = WeightedOptions::default();
        // ∫₀^π x cos(10x) dx = (cos 10π − 1)/100 = 0.
        let r = quad_weighted(|x| x, 0.0, PI, Weight::Cos(10.0), &o).unwrap();
        assert!(r.value.abs() < 1e-12, "{r:?}");
        // ∫₀^∞ e^{−x} cos x dx = 1/2, ∫₀^∞ sin x / x dx = π/2.
        let r = quad_weighted(|x| (-x).exp(), 0.0, f64::INFINITY, Weight::Cos(1.0), &o).unwrap();
        assert!((r.value - 0.5).abs() < 1e-9, "{r:?}");
        let r = quad_weighted(|x| if x == 0.0 { 1.0 } else { 1.0 / x }, 0.0, f64::INFINITY, Weight::Sin(1.0), &o).unwrap();
        assert!((r.value - PI / 2.0).abs() < 1e-7, "{r:?}");
        // ∫₀¹ x^{−1/2} (1 − x)^{−1/2} dx = π; ∫₀¹ ln x dx = −1.
        let r = quad_weighted(|_| 1.0, 0.0, 1.0, Weight::Algebraic { alpha: -0.5, beta: -0.5, log: LogFactor::None }, &o).unwrap();
        assert!((r.value - PI).abs() < 1e-10, "{r:?}");
        let r = quad_weighted(|_| 1.0, 0.0, 1.0, Weight::Algebraic { alpha: 0.0, beta: 0.0, log: LogFactor::Left }, &o).unwrap();
        assert!((r.value + 1.0).abs() < 1e-12, "{r:?}");
        // ⨍₋₁² 1/(x − 0) dx = ln 2.
        let r = quad_weighted(|_| 1.0, -1.0, 2.0, Weight::Cauchy(0.0), &o).unwrap();
        assert!((r.value - 2f64.ln()).abs() < 1e-12, "{r:?}");
    }
}
