//! Distribution functions of the multivariate normal and t distributions by Genz's
//! quasi-Monte Carlo method (SciPy's `multivariate_normal.cdf` and `multivariate_t.cdf`), the
//! exact bivariate normal distribution function (Drezner and Wesolowsky, as Genz's `bvnu`), and
//! Dunnett's many-to-one comparisons (SciPy's `dunnett`).
//!
//! See the book entry [Multivariate distribution functions and quasi-Monte Carlo sampling](https://jwkunz.github.io/lib_jsl/book/statistics/mvn-cdf.html).

// Lattice and Cholesky kernels read best as index loops.
#![allow(clippy::needless_range_loop)]

use crate::multivariate::{MultivariateNormal, MultivariateT};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_special_functions::{gammaincinv, normal_cdf, normal_quantile};
use std::f64::consts::PI;

fn phi(x: f64) -> f64 {
    if x == f64::INFINITY {
        1.0
    } else if x == f64::NEG_INFINITY {
        0.0
    } else {
        normal_cdf(x, 0.0, 1.0)
    }
}

fn phinv(p: f64) -> f64 {
    normal_quantile(p.clamp(1e-300, 1.0 - f64::EPSILON / 2.0), 0.0, 1.0)
}

// ------------------------------------------------------------------------- bivariate normal

/// Gauss-Legendre nodes and weights on `[−1, 1]` (Newton's method on `P_n`).
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                let (mut q0, mut q1) = (1.0, z);
                for k in 2..=n {
                    let q2 = ((2 * k - 1) as f64 * z * q1 - (k - 1) as f64 * q0) / k as f64;
                    q0 = q1;
                    q1 = q2;
                }
                let dq = n as f64 * (z * q1 - q0) / (z * z - 1.0);
                x[i] = z;
                x[n - 1 - i] = -z;
                w[i] = 2.0 / ((1.0 - z * z) * dq * dq);
                w[n - 1 - i] = w[i];
                break;
            }
        }
    }
    (x, w)
}

// ANCHOR: bvnu
/// `P(X > h, Y > k)` for standard bivariate normal `(X, Y)` with correlation `r` (Genz's `bvnu`,
/// after Drezner and Wesolowsky): for `|r| < 0.925` Plackett's integral over the correlation,
/// `∫₀^{asin r} exp(−(h² + k² − 2hk sin θ)/(2 cos²θ)) dθ / 2π + Φ(−h)Φ(−k)`, by 6-, 12- or
/// 20-point Gauss-Legendre rules; near `|r| = 1` an asymptotic expansion plus a correction
/// integral. Accurate to about 10⁻¹⁵.
pub fn bivariate_normal_upper(h: f64, k: f64, r: f64) -> f64 {
    if h == f64::INFINITY || k == f64::INFINITY {
        return 0.0;
    }
    if h == f64::NEG_INFINITY {
        return if k == f64::NEG_INFINITY { 1.0 } else { phi(-k) };
    }
    if k == f64::NEG_INFINITY {
        return phi(-h);
    }
    if r == 0.0 {
        return phi(-h) * phi(-k);
    }
    let tp = 2.0 * PI;
    let n = if r.abs() < 0.3 {
        6
    } else if r.abs() < 0.75 {
        12
    } else {
        20
    };
    let (xs, ws) = gauss_legendre(n);
    let (x, w): (Vec<f64>, Vec<f64>) = xs.iter().zip(&ws).filter(|(x, _)| **x < 0.0).map(|(a, b)| (*a, *b)).unzip();
    let mut k = k;
    let mut hk = h * k;
    let mut bvn = 0.0;
    if r.abs() < 0.925 {
        let hs = (h * h + k * k) / 2.0;
        let asr = r.asin();
        for i in 0..x.len() {
            let sn = (asr * (1.0 + x[i]) / 2.0).sin();
            bvn += w[i] * ((sn * hk - hs) / (1.0 - sn * sn)).exp();
            let sn = (asr * (1.0 - x[i]) / 2.0).sin();
            bvn += w[i] * ((sn * hk - hs) / (1.0 - sn * sn)).exp();
        }
        return bvn * asr / (2.0 * tp) + phi(-h) * phi(-k);
    }
    if r < 0.0 {
        k = -k;
        hk = -hk;
    }
    if r.abs() < 1.0 {
        let as_ = (1.0 - r) * (1.0 + r);
        let mut a = as_.sqrt();
        let bs = (h - k) * (h - k);
        let c = (4.0 - hk) / 8.0;
        let d = (12.0 - hk) / 16.0;
        bvn = a * (-(bs / as_ + hk) / 2.0).exp() * (1.0 - c * (bs - as_) * (1.0 - d * bs / 5.0) / 3.0 + c * d * as_ * as_ / 5.0);
        if hk > -160.0 {
            let b = bs.sqrt();
            bvn -= (-hk / 2.0).exp() * tp.sqrt() * phi(-b / a) * b * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
        }
        a /= 2.0;
        for i in 0..x.len() {
            for sgn in [-1.0, 1.0] {
                let xs = (a * (sgn * x[i] + 1.0)).powi(2);
                let rs = (1.0 - xs).sqrt();
                bvn += a * w[i] * ((-bs / (2.0 * xs) - hk / (1.0 + rs)).exp() / rs - (-(bs / xs + hk) / 2.0).exp() * (1.0 + c * xs * (1.0 + d * xs)));
            }
        }
        bvn = -bvn / tp;
    }
    if r > 0.0 {
        bvn += phi(-h.max(k));
    } else {
        bvn = -bvn;
        if k > h {
            if h < 0.0 {
                bvn += phi(k) - phi(h);
            } else {
                bvn += phi(-h) - phi(-k);
            }
        }
    }
    bvn.clamp(0.0, 1.0)
}
// ANCHOR_END: bvnu

/// `P(a < X < b)` for a bivariate normal with covariance `cov` (SciPy's `_bvn`).
fn bivariate_rectangle(a: &[f64], b: &[f64], cov: &[Vec<f64>]) -> f64 {
    let (s1, s2) = (cov[0][0].sqrt(), cov[1][1].sqrt());
    let r = cov[0][1] / (s1 * s2);
    let (xl, xu, yl, yu) = (a[0] / s1, b[0] / s1, a[1] / s2, b[1] / s2);
    let p = bivariate_normal_upper(xl, yl, r) - bivariate_normal_upper(xu, yl, r) - bivariate_normal_upper(xl, yu, r) + bivariate_normal_upper(xu, yu, r);
    p.clamp(0.0, 1.0)
}

// ------------------------------------------------------------------- Genz's QMC integration

fn primes_up_to(n: usize) -> Vec<usize> {
    if n < 2 {
        return Vec::new();
    }
    let mut sieve = vec![true; n + 1];
    sieve[0] = false;
    sieve[1] = false;
    let mut i = 2;
    while i * i <= n {
        if sieve[i] {
            for j in (i * i..=n).step_by(i) {
                sieve[j] = false;
            }
        }
        i += 1;
    }
    (0..=n).filter(|&k| sieve[k]).collect()
}

fn primitive_root(p: usize) -> usize {
    let pm = p - 1;
    let mut factors = Vec::new();
    let mut m = pm;
    let mut f = 2;
    while f * f <= m {
        if m.is_multiple_of(f) {
            factors.push(f);
            while m.is_multiple_of(f) {
                m /= f;
            }
        }
        f += 1;
    }
    if m > 1 {
        factors.push(m);
    }
    let powmod = |mut b: u128, mut e: u128, md: u128| {
        let mut r = 1u128;
        b %= md;
        while e > 0 {
            if e & 1 == 1 {
                r = r * b % md;
            }
            b = b * b % md;
            e >>= 1;
        }
        r
    };
    (2..p).find(|&g| factors.iter().all(|&q| powmod(g as u128, (pm / q) as u128, p as u128) != 1)).unwrap_or(1)
}

/// A rank-1 lattice generating vector by Nuyens and Cools's fast component-by-component
/// construction (SciPy's `_cbc_lattice`), for the largest prime number of points `≤ n`.
fn cbc_lattice(n_dim: usize, n_qmc: usize) -> (Vec<f64>, usize) {
    let p = *primes_up_to(n_qmc.max(2)).last().expect("a prime ≥ 2");
    let mut z = vec![1usize; n_dim.max(1)];
    if p < 5 || n_dim <= 1 {
        return (z.iter().map(|&v| v as f64 / p as f64).collect(), p);
    }
    let m = (p - 1) / 2;
    let g = primitive_root(p);
    let mut perm = vec![1usize; m];
    for j in 0..m - 1 {
        perm[j + 1] = g * perm[j] % p;
    }
    let perm: Vec<usize> = perm.into_iter().map(|v| v.min(p - v)).collect();
    let c: Vec<f64> = perm.iter().map(|&v| {
        let pn = v as f64 / p as f64;
        pn * pn - pn + 1.0 / 6.0
    }).collect();
    let mut q = vec![1.0f64; m];
    let mut w = 0usize;
    // Circular convolutions by FFTs of length m (SciPy's ifft(fft(c) · fft(q))).
    use lib_jsl_ffts::{BestFft, FftDirection, FftEngine1D, FftOrdering, FftScaleFactor};
    use num_complex::Complex;
    let mut fwd = BestFft::new();
    let mut inv = BestFft::new();
    let planned = fwd.plan(m, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).is_ok() && inv.plan(m, FftScaleFactor::N, FftDirection::Inverse, FftOrdering::Standard).is_ok();
    let fc: Vec<Complex<f64>> = if planned { fwd.execute(&c.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>()).unwrap_or_default() } else { Vec::new() };
    for s in 1..n_dim {
        // reordered = [c[..=w] reversed, c[w+1..] reversed]
        let mut reordered: Vec<f64> = c[..=w].iter().rev().copied().collect();
        reordered.extend(c[w + 1..].iter().rev());
        let gm = if s == 1 { 1.0 } else { 0.8f64.powi(s as i32 - 2) };
        for i in 0..m {
            q[i] *= 1.0 + gm * reordered[i];
        }
        let conv: Vec<f64> = if planned && fc.len() == m {
            let fq = fwd.execute(&q.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>()).expect("planned");
            let prod: Vec<Complex<f64>> = fc.iter().zip(&fq).map(|(a, b)| a * b).collect();
            inv.execute(&prod).expect("planned").iter().map(|v| v.re).collect()
        } else {
            (0..m).map(|t| (0..m).map(|i| c[i] * q[(t + m - i) % m]).sum()).collect()
        };
        w = conv.iter().enumerate().fold((f64::INFINITY, 0usize), |b, (t, &v)| if v < b.0 { (v, t) } else { b }).1;
        z[s] = perm[w];
    }
    (z.iter().map(|&v| v as f64 / p as f64).collect(), p)
}

/// SciPy's `_permuted_cholesky`: Genz's variable reordering (the most constrained variable
/// first) with the Cholesky factor of the correlation-scaled covariance, tolerating positive
/// semidefinite matrices.
fn permuted_cholesky(cov: &[Vec<f64>], low: &[f64], high: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let n = cov.len();
    let tol = 1e-10;
    let dc: Vec<f64> = (0..n).map(|i| { let v = cov[i][i].max(0.0).sqrt(); if v == 0.0 { 1.0 } else { v } }).collect();
    let mut lo: Vec<f64> = (0..n).map(|i| low[i] / dc[i]).collect();
    let mut hi: Vec<f64> = (0..n).map(|i| high[i] / dc[i]).collect();
    let mut cho: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| cov[i][j] / dc[i] / dc[j]).collect()).collect();
    let mut y = vec![0.0; n];
    let sqtp = (2.0 * PI).sqrt();
    for k in 0..n {
        let epk = (k + 1) as f64 * tol;
        let (mut im, mut ck, mut dem, mut lo_m, mut hi_m) = (k, 0.0, 1.0, 0.0, 0.0);
        for i in k..n {
            if cho[i][i] > tol {
                let ci = cho[i][i].sqrt();
                let s: f64 = if i > 0 { (0..k).map(|t| cho[i][t] * y[t]).sum() } else { 0.0 };
                let lo_i = (lo[i] - s) / ci;
                let hi_i = (hi[i] - s) / ci;
                let de = phi(hi_i) - phi(lo_i);
                if de <= dem {
                    ck = ci;
                    dem = de;
                    lo_m = lo_i;
                    hi_m = hi_i;
                    im = i;
                }
            }
        }
        if im > k {
            cho[im][im] = cho[k][k];
            for t in 0..k {
                let tmp = cho[im][t];
                cho[im][t] = cho[k][t];
                cho[k][t] = tmp;
            }
            for row in cho.iter_mut().take(n).skip(im + 1) {
                row.swap(im, k);
            }
            for t in k + 1..im {
                let tmp = cho[t][k];
                cho[t][k] = cho[im][t];
                cho[im][t] = tmp;
            }
            lo.swap(k, im);
            hi.swap(k, im);
        }
        if ck > epk {
            cho[k][k] = ck;
            for t in k + 1..n {
                cho[k][t] = 0.0;
            }
            for i in k + 1..n {
                cho[i][k] /= ck;
                for t in k + 1..=i {
                    cho[i][t] -= cho[i][k] * cho[t][k];
                }
            }
            if dem.abs() > tol {
                y[k] = ((-lo_m * lo_m / 2.0).exp() - (-hi_m * hi_m / 2.0).exp()) / (sqtp * dem);
            } else {
                y[k] = (lo_m + hi_m) / 2.0;
                if lo_m < -10.0 {
                    y[k] = hi_m;
                } else if hi_m > 10.0 {
                    y[k] = lo_m;
                }
            }
            for t in 0..=k {
                cho[k][t] /= ck;
            }
            lo[k] /= ck;
            hi[k] /= ck;
        } else {
            for t in k..n {
                cho[t][k] = 0.0;
            }
            y[k] = (lo[k] + hi[k]) / 2.0;
        }
    }
    (cho, lo, hi)
}

/// One randomised-lattice estimate: Genz's separation-of-variables integrand averaged over
/// `n_batches` random shifts of the lattice, with the tent periodisation. `chi` adds a leading
/// χ-variable for the t distribution with `ν` degrees of freedom.
fn qmc_estimate<U: UniformGenerator>(m: usize, cov: &[Vec<f64>], low: &[f64], high: &[f64], nu: Option<f64>, rng: &mut U) -> (f64, f64, usize) {
    let n_batches = 10;
    let n = cov.len();
    let sn = nu.map_or(1.0, |v| v.sqrt().max(1.0));
    let (lo_s, hi_s): (Vec<f64>, Vec<f64>) = (low.iter().map(|v| v / sn).collect(), high.iter().map(|v| v / sn).collect());
    let (cho, lo, hi) = permuted_cholesky(cov, &lo_s, &hi_s);
    let qdim = if nu.is_some() { n } else { n - 1 };
    let (q, nq) = cbc_lattice(qdim.max(1), (m / n_batches).max(1));
    let (mut prob, mut err_var) = (0.0, 0.0);
    let mut y = vec![0.0; n];
    for j in 0..n_batches {
        let shift: Vec<f64> = (0..qdim.max(1)).map(|_| rng.next_f64()).collect();
        let mut sum = 0.0;
        for k in 0..nq {
            let point = |d: usize| -> f64 {
                let z = q[d] * (k + 1) as f64 + shift[d];
                (2.0 * (z - z.floor()) - 1.0).abs()
            };
            // The scale of the limits: 1, or √(W/ν)·sn for the t distribution, W ~ χ²_ν.
            let (scale, off) = match nu {
                Some(v) => {
                    let u = point(0).clamp(1e-300, 1.0 - 1e-16);
                    let w = 2.0 * gammaincinv(v / 2.0, u);
                    ((w / v).sqrt() * sn, 1)
                }
                None => (1.0, 0),
            };
            let ct = cho[0][0];
            let mut c = phi(lo[0] * scale / ct);
            let d0 = phi(hi[0] * scale / ct);
            let mut dc = d0 - c;
            let mut pv = dc;
            for i in 1..n {
                let x = point(i - 1 + off);
                y[i - 1] = phinv(c + x * dc);
                let s: f64 = (0..i).map(|t| cho[i][t] * y[t]).sum();
                let ct = cho[i][i];
                if ct == 0.0 {
                    continue;
                }
                c = phi((lo[i] * scale - s) / ct);
                let d = phi((hi[i] * scale - s) / ct);
                dc = d - c;
                pv *= dc;
            }
            sum += pv;
        }
        let mean = sum / nq as f64;
        let d = (mean - prob) / (j + 1) as f64;
        prob += d;
        err_var = (j as f64 - 1.0) * err_var / (j + 1) as f64 + d * d;
    }
    (prob, 3.0 * err_var.sqrt(), nq * n_batches)
}

/// SciPy's `_qauto`: repeat the estimate with `√2` times more points until the error is below
/// `abseps` or `maxpts` points are used, weighting the estimates by their errors.
fn qauto<U: UniformGenerator>(cov: &[Vec<f64>], low: &[f64], high: &[f64], abseps: f64, maxpts: usize, rng: &mut U) -> (f64, f64) {
    let n = cov.len();
    if n == 1 {
        let s = cov[0][0].sqrt();
        return (phi(high[0] / s) - phi(low[0] / s), 1e-15);
    }
    if n == 2 {
        return (bivariate_rectangle(low, high, cov), 1e-15);
    }
    let mut mi = maxpts.min(n * 1000) as f64;
    let (mut prob, mut est_error, mut n_samples) = (0.0, 1.0, 0usize);
    while est_error > abseps && n_samples < maxpts {
        mi = (2f64.sqrt() * mi).round();
        let (pi, ei, ni) = qmc_estimate(mi as usize, cov, low, high, None, rng);
        n_samples += ni;
        let wt = 1.0 / (1.0 + (ei / est_error).powi(2));
        prob += wt * (pi - prob);
        est_error = wt.sqrt() * ei;
    }
    (prob, est_error)
}

/// Options for the multivariate distribution functions (SciPy's defaults).
#[derive(Clone, Copy, Debug)]
pub struct MvCdfOptions {
    /// The largest number of integrand evaluations; `None` for SciPy's `10⁶ d` (normal) or
    /// `1000 d` (t).
    pub maxpts: Option<usize>,
    /// The target absolute error of the normal distribution function (SciPy's `abseps`).
    pub abseps: f64,
}

impl Default for MvCdfOptions {
    fn default() -> Self {
        MvCdfOptions { maxpts: None, abseps: 1e-5 }
    }
}

/// Swaps reversed limits and returns the sign of the result (as SciPy does).
fn order_limits(lower: &[f64], upper: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
    let (mut a, mut b) = (lower.to_vec(), upper.to_vec());
    let mut sign = 1.0;
    for i in 0..a.len() {
        if b[i] < a[i] {
            std::mem::swap(&mut a[i], &mut b[i]);
            sign = -sign;
        }
    }
    (a, b, sign)
}

// ANCHOR: mvn_cdf
impl MultivariateNormal {
    /// `P(lower < X ≤ upper)` (SciPy's `multivariate_normal.cdf(upper, lower_limit=lower)`;
    /// `lower = None` for −∞), with the error estimate. One and two dimensions are exact; in
    /// more, Genz's method: reorder the variables so the most constrained comes first, factor
    /// `Σ = L Lᵀ`, and write the probability as an integral over the unit cube of a product of
    /// one-dimensional normal probabilities whose limits depend on the earlier variables, then
    /// integrate it with randomised rank-1 lattice rules until the error is below `abseps`.
    pub fn cdf_between<U: UniformGenerator>(&self, lower: Option<&[f64]>, upper: &[f64], options: MvCdfOptions, rng: &mut U) -> Result<(f64, f64), ErrorsJSL> {
        let d = self.dim();
        if upper.len() != d || lower.is_some_and(|l| l.len() != d) {
            return Err(ErrorsJSL::IncompatibleArraySizes((upper.len(), d)));
        }
        let low: Vec<f64> = match lower {
            Some(l) => l.iter().zip(&self.mean).map(|(a, m)| a - m).collect(),
            None => vec![f64::NEG_INFINITY; d],
        };
        let up: Vec<f64> = upper.iter().zip(&self.mean).map(|(a, m)| a - m).collect();
        let (a, b, sign) = order_limits(&low, &up);
        let (p, e) = qauto(&self.cov, &a, &b, options.abseps, options.maxpts.unwrap_or(1_000_000 * d), rng);
        Ok((sign * p, e))
    }

    /// `P(X ≤ x)` (SciPy's `multivariate_normal.cdf`).
    pub fn cdf<U: UniformGenerator>(&self, x: &[f64], rng: &mut U) -> Result<f64, ErrorsJSL> {
        Ok(self.cdf_between(None, x, MvCdfOptions::default(), rng)?.0)
    }
}
// ANCHOR_END: mvn_cdf

impl MultivariateT {
    /// `P(lower < X ≤ upper)` (SciPy's `multivariate_t.cdf`), by Genz's method with a leading
    /// χ variable: `X = μ + Z/√(W/ν)`, so the probability is the normal one with limits scaled
    /// by `√(W/ν)`, averaged over `W ~ χ²_ν` through its inverse distribution function. SciPy
    /// uses one lattice estimate of `1000 d` points.
    pub fn cdf_between<U: UniformGenerator>(&self, lower: Option<&[f64]>, upper: &[f64], options: MvCdfOptions, rng: &mut U) -> Result<(f64, f64), ErrorsJSL> {
        let d = self.loc.len();
        if upper.len() != d || lower.is_some_and(|l| l.len() != d) {
            return Err(ErrorsJSL::IncompatibleArraySizes((upper.len(), d)));
        }
        let low: Vec<f64> = match lower {
            Some(l) => l.iter().zip(&self.loc).map(|(a, m)| a - m).collect(),
            None => vec![f64::NEG_INFINITY; d],
        };
        let up: Vec<f64> = upper.iter().zip(&self.loc).map(|(a, m)| a - m).collect();
        let (a, b, sign) = order_limits(&low, &up);
        let (p, e, _) = qmc_estimate(options.maxpts.unwrap_or(1000 * d), &self.shape, &a, &b, Some(self.df), rng);
        Ok((sign * p, e))
    }

    /// `P(X ≤ x)` (SciPy's `multivariate_t.cdf`).
    pub fn cdf<U: UniformGenerator>(&self, x: &[f64], rng: &mut U) -> Result<f64, ErrorsJSL> {
        Ok(self.cdf_between(None, x, MvCdfOptions::default(), rng)?.0)
    }
}

// ------------------------------------------------------------------------------------ Dunnett

/// The result of [`dunnett`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DunnettResult {
    /// The t statistic of each treatment against the control.
    pub statistic: Vec<f64>,
    /// The adjusted p-value of each comparison.
    pub pvalue: Vec<f64>,
    /// The pooled standard deviation and its degrees of freedom.
    pub std: f64,
    pub df: f64,
}

/// The alternative hypothesis of [`dunnett`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DunnettAlternative {
    TwoSided,
    /// The treatment means exceed the control's.
    Greater,
    Less,
}

// ANCHOR: dunnett
/// Dunnett's test comparing each treatment group with a control (SciPy's `dunnett`):
/// `tᵢ = (x̄ᵢ − x̄₀)/(s √(1/nᵢ + 1/n₀))` with the pooled standard deviation `s` on
/// `N − k − 1` degrees of freedom. The statistics are jointly multivariate t with correlations
/// `ρᵢⱼ = 1/√((1 + n₀/nᵢ)(1 + n₀/nⱼ))`, and each p-value is the probability that the largest
/// (absolute) statistic exceeds `tᵢ`, from the multivariate t distribution function — so the
/// p-values carry its quasi-Monte Carlo error (about 10⁻³ with SciPy's 1000 points per
/// dimension).
pub fn dunnett<U: UniformGenerator>(samples: &[&[f64]], control: &[f64], alternative: DunnettAlternative, rng: &mut U) -> Result<DunnettResult, ErrorsJSL> {
    let k = samples.len();
    if k == 0 || control.len() < 2 || samples.iter().any(|s| s.is_empty()) {
        return Err(ErrorsJSL::InvalidInputRange("dunnett: need a control of two or more values and non-empty treatment groups"));
    }
    let n0 = control.len() as f64;
    let ns: Vec<f64> = samples.iter().map(|s| s.len() as f64).collect();
    let n_total = n0 + ns.iter().sum::<f64>();
    let df = n_total - k as f64 - 1.0;
    if df <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("dunnett: not enough observations for the degrees of freedom"));
    }
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let m0 = mean(control);
    let ms: Vec<f64> = samples.iter().map(|s| mean(s)).collect();
    let ss = |s: &[f64], m: f64| s.iter().map(|v| (v - m).powi(2)).sum::<f64>();
    let s2 = (ss(control, m0) + samples.iter().zip(&ms).map(|(s, &m)| ss(s, m)).sum::<f64>()) / df;
    let std = s2.sqrt();
    let statistic: Vec<f64> = (0..k).map(|i| (ms[i] - m0) / (1.0 / ns[i] + 1.0 / n0).sqrt() / std).collect();
    let r: Vec<f64> = ns.iter().map(|n| n0 / n + 1.0).collect();
    let rho: Vec<Vec<f64>> = (0..k).map(|i| (0..k).map(|j| if i == j { 1.0 } else { 1.0 / (r[i] * r[j]).sqrt() }).collect()).collect();
    let mvt = MultivariateT::new(vec![0.0; k], rho, df)?;
    let opts = MvCdfOptions::default();
    let mut pvalue = Vec::with_capacity(k);
    for &t in &statistic {
        let p = match alternative {
            DunnettAlternative::TwoSided => {
                let a = t.abs();
                1.0 - mvt.cdf_between(Some(&vec![-a; k]), &vec![a; k], opts, rng)?.0
            }
            DunnettAlternative::Greater => 1.0 - mvt.cdf_between(None, &vec![t; k], opts, rng)?.0,
            DunnettAlternative::Less => 1.0 - mvt.cdf_between(Some(&vec![t; k]), &vec![f64::INFINITY; k], opts, rng)?.0,
        };
        pvalue.push(p.clamp(0.0, 1.0));
    }
    Ok(DunnettResult { statistic, pvalue, std, df })
}
// ANCHOR_END: dunnett

#[cfg(test)]
mod tests {
    use super::*;
    use lib_jsl_random::uniform_generator::UniformRNG;

    #[test]
    fn bivariate_and_trivariate_against_known_values() {
        // P(X > 0, Y > 0) = 1/4 + asin(r)/(2π).
        for r in [-0.95f64, -0.5, 0.2, 0.8, 0.99] {
            let want = 0.25 + r.asin() / (2.0 * PI);
            assert!((bivariate_normal_upper(0.0, 0.0, r) - want).abs() < 1e-14, "{r}");
        }
        // An equicorrelated trivariate orthant: P(all < 0) = 1/8 + 3 asin(ρ)/(4π).
        let rho = 0.5;
        let cov = vec![vec![1.0, rho, rho], vec![rho, 1.0, rho], vec![rho, rho, 1.0]];
        let mvn = MultivariateNormal::new(vec![0.0; 3], cov.clone()).unwrap();
        let mut rng = UniformRNG::from_seed(7);
        let (p, e) = mvn.cdf_between(None, &[0.0; 3], MvCdfOptions::default(), &mut rng).unwrap();
        let want = 0.125 + 3.0 * rho.asin() / (4.0 * PI);
        assert!((p - want).abs() < 2e-5 && e < 1e-4, "{p} vs {want} (±{e})");
        // The t distribution's orthant probability is the same (elliptical symmetry).
        let t = MultivariateT::new(vec![0.0; 3], cov, 5.0).unwrap();
        let (pt, _) = t.cdf_between(None, &[0.0; 3], MvCdfOptions { maxpts: Some(100_000), ..Default::default() }, &mut rng).unwrap();
        assert!((pt - want).abs() < 2e-4, "{pt}");
    }
}
