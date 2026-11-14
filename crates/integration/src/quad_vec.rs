//! Adaptive integration of vector-valued functions over finite or infinite intervals, a port of
//! SciPy's `quad_vec`: every component shares one set of subintervals, chosen by a global
//! Gauss-Kronrod error estimate in the 2-norm (or the max norm).
//!
//! See the book entry [Vector integrands and weighted quadrature](https://jwkunz.github.io/lib_jsl/book/integration/vector-weighted.html).

#![allow(clippy::excessive_precision)] // Gauss-Kronrod nodes and weights as printed in SciPy.

use lib_jsl_core::ErrorsJSL;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// How the error of a vector of integrals is measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VecNorm {
    /// The largest absolute component (SciPy's `'max'`).
    Max,
    /// The Euclidean norm (SciPy's default, `'2'`).
    Euclidean,
}

/// The local rule applied to each subinterval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VecRule {
    /// 21-point Gauss-Kronrod (SciPy's default on finite intervals).
    GaussKronrod21,
    /// 15-point Gauss-Kronrod (SciPy's default after mapping an infinite interval).
    GaussKronrod15,
    /// The trapezoid rule with a Richardson error estimate, for non-smooth integrands.
    Trapezoid,
}

/// Options for [`quad_vec`] (SciPy's defaults).
#[derive(Clone, Debug)]
pub struct QuadVecOptions {
    pub epsabs: f64,
    pub epsrel: f64,
    pub norm: VecNorm,
    /// The rule; `None` chooses SciPy's default (21-point Gauss-Kronrod, or 15-point after mapping
    /// an infinite interval).
    pub rule: Option<VecRule>,
    /// The largest number of subintervals.
    pub limit: usize,
    /// Interior points (discontinuities, kinks) where the interval is split from the start.
    pub points: Vec<f64>,
}

impl Default for QuadVecOptions {
    fn default() -> Self {
        QuadVecOptions { epsabs: 1e-200, epsrel: 1e-8, norm: VecNorm::Euclidean, rule: None, limit: 10_000, points: Vec::new() }
    }
}

/// The result of [`quad_vec`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct QuadVecResult {
    pub value: Vec<f64>,
    /// The error estimate (in the chosen norm), including the rounding-error estimate.
    pub error: f64,
    pub nfev: usize,
    pub converged: bool,
    /// 0 converged, 1 subinterval limit reached, 2 limited by rounding error, 3 non-finite values.
    pub status: i32,
    /// The number of subintervals in the final partition.
    pub intervals: usize,
}

const GK21_X: [f64; 21] = [
    0.995657163025808080735527280689003, 0.973906528517171720077964012084452, 0.930157491355708226001207180059508,
    0.865063366688984510732096688423493, 0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
    0.562757134668604683339000099272694, 0.433395394129247190799265943165784, 0.294392862701460198131126603103866,
    0.148874338981631210884826001129720, 0.0, -0.148874338981631210884826001129720, -0.294392862701460198131126603103866,
    -0.433395394129247190799265943165784, -0.562757134668604683339000099272694, -0.679409568299024406234327365114874,
    -0.780817726586416897063717578345042, -0.865063366688984510732096688423493, -0.930157491355708226001207180059508,
    -0.973906528517171720077964012084452, -0.995657163025808080735527280689003,
];
const GK21_W: [f64; 10] = [
    0.066671344308688137593568809893332, 0.149451349150580593145776339657697, 0.219086362515982043995534934228163,
    0.269266719309996355091226921569469, 0.295524224714752870173892994651338, 0.295524224714752870173892994651338,
    0.269266719309996355091226921569469, 0.219086362515982043995534934228163, 0.149451349150580593145776339657697,
    0.066671344308688137593568809893332,
];
const GK21_V: [f64; 21] = [
    0.011694638867371874278064396062192, 0.032558162307964727478818972459390, 0.054755896574351996031381300244580,
    0.075039674810919952767043140916190, 0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
    0.123491976262065851077958109831074, 0.134709217311473325928054001771707, 0.142775938577060080797094273138717,
    0.147739104901338491374841515972068, 0.149445554002916905664936468389821, 0.147739104901338491374841515972068,
    0.142775938577060080797094273138717, 0.134709217311473325928054001771707, 0.123491976262065851077958109831074,
    0.109387158802297641899210590325805, 0.093125454583697605535065465083366, 0.075039674810919952767043140916190,
    0.054755896574351996031381300244580, 0.032558162307964727478818972459390, 0.011694638867371874278064396062192,
];
const GK15_X: [f64; 15] = [
    0.991455371120812639206854697526329, 0.949107912342758524526189684047851, 0.864864423359769072789712788640926,
    0.741531185599394439863864773280788, 0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
    0.207784955007898467600689403773245, 0.0, -0.207784955007898467600689403773245, -0.405845151377397166906606412076961,
    -0.586087235467691130294144838258730, -0.741531185599394439863864773280788, -0.864864423359769072789712788640926,
    -0.949107912342758524526189684047851, -0.991455371120812639206854697526329,
];
const GK15_W: [f64; 7] = [
    0.129484966168869693270611432679082, 0.279705391489276667901467771423780, 0.381830050505118944950369775488975,
    0.417959183673469387755102040816327, 0.381830050505118944950369775488975, 0.279705391489276667901467771423780,
    0.129484966168869693270611432679082,
];
const GK15_V: [f64; 15] = [
    0.022935322010529224963732008058970, 0.063092092629978553290700663189204, 0.104790010322250183839876322541518,
    0.140653259715525918745189590510238, 0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
    0.204432940075298892414161999234649, 0.209482141084727828012999174891714, 0.204432940075298892414161999234649,
    0.190350578064785409913256402421014, 0.169004726639267902826583426598550, 0.140653259715525918745189590510238,
    0.104790010322250183839876322541518, 0.063092092629978553290700663189204, 0.022935322010529224963732008058970,
];

fn norm_of(norm: VecNorm, v: &[f64]) -> f64 {
    match norm {
        VecNorm::Max => v.iter().fold(0.0f64, |m, x| m.max(x.abs())),
        VecNorm::Euclidean => v.iter().map(|x| x * x).sum::<f64>().sqrt(),
    }
}

/// One interval's estimate: the integral, its error, and its rounding-error estimate.
type Estimate = (Vec<f64>, f64, f64);

// ANCHOR: gk
/// Gauss-Kronrod on `[a, b]` (SciPy's `_quadrature_gk`): the Kronrod sum and the embedded Gauss
/// sum; the error `‖h (K − G)‖` is rescaled as in QUADPACK, `D min(1, (200 E / D)^{3/2})` with
/// `D = ‖h Σ vᵢ |fᵢ − K/2|‖`, and is not allowed below the rounding estimate `‖50 ε h Σ vᵢ|fᵢ|‖`.
fn gauss_kronrod(f: &mut dyn FnMut(f64) -> Vec<f64>, a: f64, b: f64, norm: VecNorm, x: &[f64], w: &[f64], v: &[f64]) -> Estimate {
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let fv: Vec<Vec<f64>> = x.iter().map(|&xi| f(c + h * xi)).collect();
    let m = fv[0].len();
    let mut s_k = vec![0.0; m];
    let mut s_k_abs = vec![0.0; m];
    for (fi, &vi) in fv.iter().zip(v) {
        for j in 0..m {
            s_k[j] += vi * fi[j];
            s_k_abs[j] += vi * fi[j].abs();
        }
    }
    let mut s_g = vec![0.0; m];
    for (i, &wi) in w.iter().enumerate() {
        for j in 0..m {
            s_g[j] += wi * fv[2 * i + 1][j];
        }
    }
    let mut s_k_dabs = vec![0.0; m];
    for (fi, &vi) in fv.iter().zip(v) {
        for j in 0..m {
            s_k_dabs[j] += vi * (fi[j] - s_k[j] / 2.0).abs();
        }
    }
    let mut err = norm_of(norm, &(0..m).map(|j| (s_k[j] - s_g[j]) * h).collect::<Vec<_>>());
    let dabs = norm_of(norm, &s_k_dabs.iter().map(|d| d * h).collect::<Vec<_>>());
    if dabs != 0.0 && err != 0.0 {
        err = dabs * (200.0 * err / dabs).powf(1.5).min(1.0);
    }
    let round_err = norm_of(norm, &s_k_abs.iter().map(|s| 50.0 * f64::EPSILON * h * s).collect::<Vec<_>>());
    if round_err > f64::MIN_POSITIVE {
        err = err.max(round_err);
    }
    (s_k.iter().map(|s| h * s).collect(), err, round_err)
}
// ANCHOR_END: gk

fn trapezoid(f: &mut dyn FnMut(f64) -> Vec<f64>, a: f64, b: f64, norm: VecNorm) -> Estimate {
    let x3 = 0.5 * (a + b);
    let (f1, f2, f3) = (f(a), f(b), f(x3));
    let m = f1.len();
    let s2: Vec<f64> = (0..m).map(|j| 0.25 * (b - a) * (f1[j] + 2.0 * f3[j] + f2[j])).collect();
    let round_err = 0.25 * (b - a).abs() * (norm_of(norm, &f1) + 2.0 * norm_of(norm, &f3) + norm_of(norm, &f2)) * 2e-16;
    let s1: Vec<f64> = (0..m).map(|j| 0.5 * (b - a) * (f1[j] + f2[j])).collect();
    let err = norm_of(norm, &(0..m).map(|j| s1[j] - s2[j]).collect::<Vec<_>>()) / 3.0;
    (s2, err, round_err)
}

fn apply(rule: VecRule, f: &mut dyn FnMut(f64) -> Vec<f64>, a: f64, b: f64, norm: VecNorm) -> (Estimate, usize) {
    match rule {
        VecRule::GaussKronrod21 => (gauss_kronrod(f, a, b, norm, &GK21_X, &GK21_W, &GK21_V), 21),
        VecRule::GaussKronrod15 => (gauss_kronrod(f, a, b, norm, &GK15_X, &GK15_W, &GK15_V), 15),
        VecRule::Trapezoid => (trapezoid(f, a, b, norm), 3),
    }
}

/// A subinterval in the priority queue, ordered as SciPy's heap of `(−error, a, b)`: the largest
/// error first, ties broken by the smaller `a`, then the smaller `b`.
#[derive(Debug, PartialEq)]
struct Item {
    err: f64,
    a: f64,
    b: f64,
}

impl Eq for Item {}

impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; "greater" pops first.
        self.err.total_cmp(&other.err).then(other.a.total_cmp(&self.a)).then(other.b.total_cmp(&self.b))
    }
}

impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn key(a: f64, b: f64) -> (u64, u64) {
    (a.to_bits(), b.to_bits())
}

// ANCHOR: quad_vec
/// `∫_a^b f(x) dx` for vector-valued `f` (SciPy's `quad_vec`), with infinite limits allowed. The
/// interval list starts from `[a, b]` split at `points`; each round pops the subintervals with the
/// largest errors (up to 128, while their errors exceed all but an eighth of the tolerance) and
/// bisects them, updating the global integral and error by the differences. It stops when the
/// global error is below `max(epsabs, epsrel ‖I‖)/8`, or when it falls below the accumulated
/// rounding error, or at `limit` subintervals. Infinite intervals are first mapped to `[0, 1]`
/// (`x = a + (1 − t)/t`) or `[−1, 1]` (`x = (1 − |t|)/t`).
pub fn quad_vec(mut f: impl FnMut(f64) -> Vec<f64>, a: f64, b: f64, options: &QuadVecOptions) -> Result<QuadVecResult, ErrorsJSL> {
    if a.is_nan() || b.is_nan() {
        return Err(ErrorsJSL::InvalidInputRange("quad_vec: the limits must not be NaN"));
    }
    let tmin = f64::MIN_POSITIVE.sqrt();
    // The number of components, from one evaluation at a finite point of the range (not counted
    // in nfev, as SciPy's evaluations of the mapped integrand are the only ones it counts).
    let probe_x = match (a.is_finite(), b.is_finite()) {
        (true, true) => 0.5 * (a + b),
        (true, false) => a + b.signum(),
        (false, true) => b + a.signum(),
        (false, false) => 0.0,
    };
    let dim = f(probe_x).len();
    if dim == 0 {
        return Err(ErrorsJSL::InvalidInputRange("quad_vec: the integrand returned an empty vector"));
    }
    if a.is_finite() && b.is_infinite() || b.is_finite() && a.is_infinite() {
        let (start, sgn, flip) = if a.is_finite() { (a, b.signum(), 1.0) } else { (b, a.signum(), -1.0) };
        let mut g = |t: f64| -> Vec<f64> {
            if t < tmin {
                return vec![0.0; dim];
            }
            let x = start + sgn * (1.0 - t) / t;
            f(x).into_iter().map(|v| sgn * (v / t) / t).collect()
        };
        let points: Vec<f64> = options.points.iter().map(|&p| { let z = sgn * (p - start) + 1.0; if z == 0.0 { f64::INFINITY } else { 1.0 / z } }).collect();
        let o = QuadVecOptions { rule: Some(options.rule.unwrap_or(VecRule::GaussKronrod15)), points, ..options.clone() };
        let mut r = finite(&mut g, 0.0, 1.0, &o)?;
        r.value.iter_mut().for_each(|v| *v *= flip);
        return Ok(r);
    }
    if a.is_infinite() && b.is_infinite() {
        let sgn = if b < a { -1.0 } else { 1.0 };
        let mut g = |t: f64| -> Vec<f64> {
            if t.abs() < tmin {
                return vec![0.0; dim];
            }
            let x = (1.0 - t.abs()) / t;
            f(x).into_iter().map(|v| (v / t) / t).collect()
        };
        let mut points = vec![0.0];
        points.extend(options.points.iter().map(|&p| p.signum() / (p.abs() + 1.0)));
        let o = QuadVecOptions { rule: Some(options.rule.unwrap_or(VecRule::GaussKronrod15)), points, ..options.clone() };
        let mut r = if a != b { finite(&mut g, -1.0, 1.0, &o)? } else { finite(&mut g, 1.0, 1.0, &o)? };
        r.value.iter_mut().for_each(|v| *v *= sgn);
        return Ok(r);
    }
    let o = QuadVecOptions { rule: Some(options.rule.unwrap_or(VecRule::GaussKronrod21)), ..options.clone() };
    finite(&mut f, a, b, &o)
}
// ANCHOR_END: quad_vec

/// `quad_vec` on a finite interval.
fn finite(eval: &mut dyn FnMut(f64) -> Vec<f64>, a: f64, b: f64, o: &QuadVecOptions) -> Result<QuadVecResult, ErrorsJSL> {
    let rule = o.rule.unwrap_or(VecRule::GaussKronrod21);
    let norm = o.norm;
    let mut intervals_init = Vec::new();
    let mut prev = a;
    let mut pts = o.points.clone();
    pts.sort_by(f64::total_cmp);
    for p in pts {
        if !(a < p && p < b) || p == prev {
            continue;
        }
        intervals_init.push((prev, p));
        prev = p;
    }
    intervals_init.push((prev, b));
    let mut cache: HashMap<(u64, u64), Vec<f64>> = HashMap::new();
    let mut heap = BinaryHeap::new();
    let mut nfev = 0;
    let mut global: Option<Vec<f64>> = None;
    let (mut global_error, mut rounding) = (0.0, 0.0);
    for (x1, x2) in intervals_init {
        let ((ig, err, rnd), n) = apply(rule, eval, x1, x2, norm);
        nfev += n;
        match &mut global {
            None => {
                global = Some(ig.clone());
                global_error = err;
                rounding = rnd;
            }
            Some(g) => {
                g.iter_mut().zip(&ig).for_each(|(a, b)| *a += b);
                global_error += err;
                rounding += rnd;
            }
        }
        cache.insert(key(x1, x2), ig);
        heap.push(Item { err, a: x1, b: x2 });
    }
    let mut global = global.expect("at least one interval");
    let mut status = 1;
    while !heap.is_empty() && heap.len() < o.limit {
        let tol = o.epsabs.max(o.epsrel * norm_of(norm, &global));
        let mut to_process = Vec::new();
        let mut err_sum = 0.0;
        for j in 0..128 {
            if heap.is_empty() {
                break;
            }
            if j > 0 && err_sum > global_error - tol / 8.0 {
                break;
            }
            let it = heap.pop().expect("non-empty");
            err_sum += it.err;
            to_process.push(it);
        }
        for it in to_process {
            let (x1, x2) = (it.a, it.b);
            let c = 0.5 * (x1 + x2);
            // As SciPy, the trapezoid rule caches values within one bisection, so the shared
            // midpoint is evaluated once; the Gauss-Kronrod nodes of the halves never coincide.
            let mut memo: HashMap<u64, Vec<f64>> = HashMap::new();
            let mut misses = 0usize;
            let mut cached = |x: f64| -> Vec<f64> {
                if rule != VecRule::Trapezoid {
                    misses += 1;
                    return eval(x);
                }
                memo.entry(x.to_bits()).or_insert_with(|| {
                    misses += 1;
                    eval(x)
                }).clone()
            };
            let ((s1, e1, r1), _) = apply(rule, &mut cached, x1, c, norm);
            let ((s2, e2, r2), _) = apply(rule, &mut cached, c, x2, norm);
            nfev += misses;
            let old = match cache.remove(&key(x1, x2)) {
                Some(v) => v,
                None => {
                    let ((v, _, _), n) = apply(rule, eval, x1, x2, norm);
                    nfev += n;
                    v
                }
            };
            for j in 0..global.len() {
                global[j] += s1[j] + s2[j] - old[j];
            }
            global_error += e1 + e2 - it.err;
            rounding += r1 + r2;
            cache.insert(key(x1, c), s1);
            cache.insert(key(c, x2), s2);
            heap.push(Item { err: e1, a: x1, b: c });
            heap.push(Item { err: e2, a: c, b: x2 });
        }
        if heap.len() >= 2 {
            let tol = o.epsabs.max(o.epsrel * norm_of(norm, &global));
            if global_error < tol / 8.0 {
                status = 0;
                break;
            }
            if global_error < rounding {
                status = 2;
                break;
            }
        }
        if !(global_error.is_finite() && rounding.is_finite()) {
            status = 3;
            break;
        }
    }
    Ok(QuadVecResult { value: global, error: global_error + rounding, nfev, converged: status == 0, status, intervals: heap.len() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_of_powers_and_infinite_ranges() {
        // ∫₀¹ (1, x, x², …, x⁹) dx = 1/(k + 1).
        let r = quad_vec(|x| (0..10).map(|k| x.powi(k)).collect(), 0.0, 1.0, &QuadVecOptions::default()).unwrap();
        assert!(r.converged);
        for (k, v) in r.value.iter().enumerate() {
            assert!((v - 1.0 / (k as f64 + 1.0)).abs() < 1e-14);
        }
        // ∫₀^∞ e^{−kx} dx = 1/k, and ∫_{−∞}^{∞} e^{−x²} = √π.
        let r = quad_vec(|x| vec![(-x).exp(), (-2.0 * x).exp()], 0.0, f64::INFINITY, &QuadVecOptions::default()).unwrap();
        assert!((r.value[0] - 1.0).abs() < 1e-10 && (r.value[1] - 0.5).abs() < 1e-10, "{r:?}");
        let r = quad_vec(|x| vec![(-x * x).exp()], f64::NEG_INFINITY, f64::INFINITY, &QuadVecOptions::default()).unwrap();
        assert!((r.value[0] - std::f64::consts::PI.sqrt()).abs() < 1e-10);
        let r = quad_vec(|x| vec![x.exp()], f64::NEG_INFINITY, 0.0, &QuadVecOptions::default()).unwrap();
        assert!((r.value[0] - 1.0).abs() < 1e-10);
    }
}
