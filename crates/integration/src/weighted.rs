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

/// `J_0(κ), …, J_L(κ)` for `κ ≥ 0`, by Miller's backward recurrence normalised with
/// `J_0 + 2 Σ J_{2k} = 1`.
fn bessel_j_all(kappa: f64, l: usize) -> Vec<f64> {
    let mut j = vec![0.0; l + 1];
    if kappa == 0.0 {
        j[0] = 1.0;
        return j;
    }
    let top = l.max(kappa as usize);
    let mut start = top + (160.0 * top as f64).sqrt() as usize + 20;
    start += start % 2;
    let (mut next, mut cur) = (0.0f64, 1e-300f64);
    let mut sum = 0.0;
    for n in (1..=start).rev() {
        let prev = 2.0 * n as f64 / kappa * cur - next;
        next = cur;
        cur = prev;
        // cur is now J_{n−1} (unnormalised).
        if n - 1 <= l {
            j[n - 1] = cur;
        }
        if (n - 1) % 2 == 0 && n > 1 {
            sum += 2.0 * cur;
        }
        if cur.abs() > 1e250 {
            cur *= 1e-250;
            next *= 1e-250;
            sum *= 1e-250;
            j.iter_mut().for_each(|v| *v *= 1e-250);
        }
    }
    let norm = sum + cur;
    j.iter_mut().for_each(|v| *v /= norm);
    j
}

// ANCHOR: fcc_moments
/// The modified moments `μ_k = ∫_{−1}^{1} T_k(t) e^{iκt} dt`, `k = 0..=n`, from the Jacobi-Anger
/// expansion `e^{iκt} = Σ ε_m i^m J_m(κ) T_m(t)` (`ε_0 = 1`, otherwise 2) and the closed form of
/// `∫ T_k T_m = 1/(1 − (k + m)²) + 1/(1 − (k − m)²)` for `k + m` even (zero otherwise). The cost
/// is `O(n κ)` operations and no evaluations of the integrand.
fn chebyshev_fourier_moments(kappa: f64, n: usize) -> Vec<(f64, f64)> {
    let k_abs = kappa.abs();
    let l = n + k_abs as usize + 40 + (10.0 * k_abs.cbrt()) as usize;
    let j = bessel_j_all(k_abs, l);
    (0..=n)
        .map(|k| {
            let (mut re, mut im) = (0.0, 0.0);
            for (m, &jm) in j.iter().enumerate() {
                if (k + m) % 2 == 1 {
                    continue;
                }
                let (s, d) = ((k + m) as f64, k as f64 - m as f64);
                let w = 1.0 / (1.0 - s * s) + 1.0 / (1.0 - d * d);
                let c = if m == 0 { 1.0 } else { 2.0 } * jm * w;
                // i^m: 1, i, −1, −i.
                match m % 4 {
                    0 => re += c,
                    1 => im += c,
                    2 => re -= c,
                    _ => im -= c,
                }
            }
            // e^{iκt} for κ < 0 is the conjugate of e^{i|κ|t}.
            (re, if kappa < 0.0 { -im } else { im })
        })
        .collect()
}
// ANCHOR_END: fcc_moments

// ANCHOR: filon
/// `∫_a^b f(x) e^{iωx} dx` by adaptive Filon-Clenshaw-Curtis quadrature: on each subinterval,
/// `f` alone is interpolated at 17, then 33, then 65 nested Chebyshev points, and the
/// interpolant is integrated against `e^{iωx}` exactly through the modified moments. The error
/// of a subinterval is estimated from the size of the last Chebyshev coefficients; the one with
/// the largest error is bisected until the total meets the tolerance. Because only `f` is
/// interpolated, the cost does not grow with `ω`. Returns `(re, im, error, nfev, converged)`.
fn filon_clenshaw_curtis(f: &mut dyn FnMut(f64) -> f64, a: f64, b: f64, omega: f64, o: &WeightedOptions) -> (f64, f64, f64, usize, bool) {
    let mut nfev = 0;
    let mut panel = |x0: f64, x1: f64, nfev: &mut usize| -> ((f64, f64), f64) {
        let (c, h) = (0.5 * (x0 + x1), 0.5 * (x1 - x0));
        let mut values: Vec<f64> = Vec::new();
        let mut last = ((0.0, 0.0), f64::INFINITY);
        for n in [16usize, 32, 64] {
            // Values at t_j = cos(jπ/n); the even-indexed ones are those of the previous level.
            let mut g = vec![0.0; n + 1];
            for (jj, gj) in g.iter_mut().enumerate() {
                if n > 16 && jj % 2 == 0 {
                    *gj = values[jj / 2];
                } else {
                    *gj = f(c + h * (jj as f64 * PI / n as f64).cos());
                    *nfev += 1;
                }
            }
            values = g;
            let coef: Vec<f64> = (0..=n)
                .map(|k| {
                    let s: f64 = (0..=n).map(|jj| {
                        let w = if jj == 0 || jj == n { 0.5 } else { 1.0 };
                        w * values[jj] * ((jj * k) as f64 * PI / n as f64).cos()
                    }).sum();
                    let ck = 2.0 / n as f64 * s;
                    if k == 0 || k == n { ck / 2.0 } else { ck }
                })
                .collect();
            let mu = chebyshev_fourier_moments(omega * h, n);
            let (mut re, mut im) = (0.0, 0.0);
            for (ck, (mr, mi)) in coef.iter().zip(&mu) {
                re += ck * mr;
                im += ck * mi;
            }
            // Multiply by h e^{iωc}.
            let (cs, sn) = ((omega * c).cos(), (omega * c).sin());
            let value = (h * (re * cs - im * sn), h * (re * sn + im * cs));
            let scale = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let tail = coef[n - 2..].iter().map(|v| v.abs()).sum::<f64>();
            let error = 2.0 * h.abs() * tail.max(50.0 * f64::EPSILON * scale);
            last = (value, error);
            let tol = o.epsabs.max(o.epsrel * value.0.hypot(value.1));
            if error <= tol {
                break;
            }
        }
        last
    };
    let mut pieces = vec![(a, b, panel(a, b, &mut nfev))];
    let limit = 200;
    loop {
        let (re, im): (f64, f64) = pieces.iter().fold((0.0, 0.0), |s, p| (s.0 + p.2.0.0, s.1 + p.2.0.1));
        let error: f64 = pieces.iter().map(|p| p.2.1).sum();
        let tol = o.epsabs.max(o.epsrel * re.hypot(im));
        if error <= tol || pieces.len() >= limit {
            return (re, im, error, nfev, error <= tol);
        }
        let worst = (0..pieces.len()).max_by(|&i, &j| pieces[i].2.1.total_cmp(&pieces[j].2.1)).expect("non-empty");
        let (x0, x1, _) = pieces.swap_remove(worst);
        let mid = 0.5 * (x0 + x1);
        pieces.push((x0, mid, panel(x0, mid, &mut nfev)));
        pieces.push((mid, x1, panel(mid, x1, &mut nfev)));
    }
}
// ANCHOR_END: filon

// ANCHOR: quad_weighted
/// `∫_a^b f(x) w(x) dx` for the weights of SciPy's `quad(weight=…)`:
///
/// - `Cos(ω)`, `Sin(ω)` on `[a, b]`: adaptive Filon-Clenshaw-Curtis quadrature, which
///   interpolates `f` alone and integrates the oscillation exactly, so its cost does not grow
///   with `ω`; on `[a, ∞)`, the integrals over successive half-periods
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
                let (re, im, error, nfev, converged) = filon_clenshaw_curtis(&mut f, a, b, omega, options);
                return Ok(WeightedResult { value: if cosine { re } else { im }, error, nfev, converged });
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

    #[test]
    fn filon_cost_does_not_grow_with_frequency() {
        // ∫₀^π e^x cos(ωx) dx = (e^π (−1)^m − 1) / (1 + ω²) for integer ω = m, and the sine
        // integral is −ω (e^π (−1)^m − 1) / (1 + ω²).
        // An absolute tolerance: the integral shrinks like 1/ω, while rounding errors stay at
        // ε max|f| (b − a), so a relative tolerance becomes unreachable at high frequency.
        let o = WeightedOptions { epsabs: 1e-12, epsrel: 0.0, ..Default::default() };
        let mut counts = Vec::new();
        for m in [3.0f64, 50.0, 2000.0, 100_000.0] {
            let sign = if m as i64 % 2 == 0 { 1.0 } else { -1.0 };
            let c = (std::f64::consts::PI.exp() * sign - 1.0) / (1.0 + m * m);
            let r = quad_weighted(f64::exp, 0.0, PI, Weight::Cos(m), &o).unwrap();
            assert!(r.converged && (r.value - c).abs() < 1e-13, "ω = {m}: {r:?} vs {c}");
            let s = quad_weighted(f64::exp, 0.0, PI, Weight::Sin(m), &o).unwrap();
            assert!((s.value + m * c).abs() < 1e-13, "ω = {m}: {s:?}");
            counts.push(r.nfev);
        }
        assert!(counts.iter().all(|&n| n == counts[0]), "{counts:?}");
        // The Bessel functions behind the moments: J₀(1), J₁(1), J₅(10).
        let j = bessel_j_all(1.0, 2);
        assert!((j[0] - 0.765_197_686_557_966_6).abs() < 1e-15 && (j[1] - 0.440_050_585_744_933_5).abs() < 1e-15);
        assert!((bessel_j_all(10.0, 5)[5] + 0.234_061_528_186_793_6).abs() < 1e-14);
    }
}
