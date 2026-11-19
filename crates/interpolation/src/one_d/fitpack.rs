//! Smoothing splines with automatically placed knots, after Dierckx's FITPACK (`curfit` and
//! `parcur`), following SciPy's `make_splrep`, `make_splprep` and `generate_knots`.
//!
//! Given data `(xᵢ, yᵢ)`, weights `wᵢ` and a smoothing factor `s`, the spline `g` of degree `k`
//! is the smoothest one — the sum of squared jumps of `g⁽ᵏ⁾` at its interior knots is least —
//! among those with `Σ (wᵢ (yᵢ − g(xᵢ)))² ≤ s`. The knots are found by adding them one interval
//! at a time where the least-squares spline fits worst, until the least-squares fit is within
//! `s`; then the weight `p` of the data against smoothness is found by rational interpolation so
//! that the residual equals `s`.
//!
//! See the book entry [Smoothing splines with automatic knots](https://jwkunz.github.io/lib_jsl/book/interpolation/fitpack.html).

// Guards such as `!(s >= 0.0)` also reject NaN; band indices read best as index loops.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::needless_range_loop)]

use super::b_spline::{BSpline, BoundaryCondition, make_interp_spline, nonzero_basis, not_a_knot_knots};
use lib_jsl_core::ErrorsJSL;

/// FITPACK's relative tolerance on `|fp − s|` and its iteration limit for `p`.
const TOL: f64 = 0.001;
const MAXIT: usize = 20;

/// Options for [`make_splrep`], [`make_splprep`] and [`generate_knots`] (SciPy's keyword
/// arguments; the defaults are SciPy's).
#[derive(Clone, Debug)]
pub struct SplrepOptions<'a> {
    /// Weights, one per point (default all one); the residuals are `wᵢ (yᵢ − g(xᵢ))`.
    pub w: Option<&'a [f64]>,
    /// The degree (default 3).
    pub k: usize,
    /// The smoothing factor `s ≥ 0` (default 0, interpolation).
    pub s: f64,
    /// The ends of the approximation interval (default the first and last `x`).
    pub xb: Option<f64>,
    pub xe: Option<f64>,
    /// Stop adding knots once there are at least this many (default no limit beyond the
    /// interpolating spline's `m + k + 1`).
    pub nest: Option<usize>,
}

impl Default for SplrepOptions<'_> {
    fn default() -> Self {
        SplrepOptions { w: None, k: 3, s: 0.0, xb: None, xe: None, nest: None }
    }
}

/// Data with `d` right-hand sides per point (one for a function, `d` for a curve), validated.
struct Data<'a> {
    x: &'a [f64],
    /// `y[i][j]`: point `i`, coordinate `j`.
    y: Vec<Vec<f64>>,
    w: Vec<f64>,
    k: usize,
    s: f64,
    xb: f64,
    xe: f64,
}

impl Data<'_> {
    fn m(&self) -> usize {
        self.x.len()
    }
}

/// An upper-triangular banded matrix (`r[i][j]` is entry `(i, i + j)`) and its right-hand sides,
/// built by Givens rotations as FITPACK does (`fpgivs`, `fprota`).
struct Banded {
    r: Vec<Vec<f64>>,
    z: Vec<Vec<f64>>,
}

impl Banded {
    fn new(nc: usize, bw: usize, d: usize) -> Self {
        Banded { r: vec![vec![0.0; bw]; nc], z: vec![vec![0.0; d]; nc] }
    }

    /// Rotates the row `h` (entries for columns `off, off + 1, …`) with right-hand side `rhs`
    /// into the triangle. The row is kept in a window of the bandwidth that slides one column
    /// per rotation, since each rotation can fill it in up to the band's edge.
    fn rotate_in(&mut self, h: Vec<f64>, off: usize, mut rhs: Vec<f64>) {
        let (nc, bw) = (self.r.len(), self.r[0].len());
        let mut win = vec![0.0; bw];
        win[..h.len().min(bw)].copy_from_slice(&h[..h.len().min(bw)]);
        let mut extra = h.get(bw..).map(|v| v.to_vec()).unwrap_or_default().into_iter();
        let mut i = off;
        while i < nc && win.iter().any(|&v| v != 0.0) {
            let piv = win[0];
            if piv != 0.0 {
                // fpgivs: the rotation that zeroes piv against the diagonal ww.
                let ww = self.r[i][0];
                let (a, b) = (piv.abs(), ww.abs());
                let dd = if a >= b { a * (1.0 + (b / a).powi(2)).sqrt() } else { b * (1.0 + (a / b).powi(2)).sqrt() };
                let (cos, sin) = (ww / dd, piv / dd);
                self.r[i][0] = dd;
                // fprota on the right-hand sides and the rest of the row.
                for (zv, yv) in self.z[i].iter_mut().zip(rhs.iter_mut()) {
                    let (s1, s2) = (*yv, *zv);
                    *zv = cos * s2 + sin * s1;
                    *yv = cos * s1 - sin * s2;
                }
                for l in 1..bw {
                    let (s1, s2) = (win[l], self.r[i][l]);
                    self.r[i][l] = cos * s2 + sin * s1;
                    win[l] = cos * s1 - sin * s2;
                }
            }
            win.rotate_left(1);
            win[bw - 1] = extra.next().unwrap_or(0.0);
            i += 1;
        }
    }

    /// Back substitution for the coefficients, one column per right-hand side (`fpback`).
    fn solve(&self) -> Vec<Vec<f64>> {
        let (nc, bw) = (self.r.len(), self.r[0].len());
        let d = self.z[0].len();
        let mut c = vec![vec![0.0; d]; nc];
        for i in (0..nc).rev() {
            for q in 0..d {
                let mut store = self.z[i][q];
                for l in 1..bw.min(nc - i) {
                    store -= c[i + l][q] * self.r[i][l];
                }
                c[i][q] = store / self.r[i][0];
            }
        }
        c
    }
}

/// The interval `l` (`t[l] ≤ x < t[l + 1]`, `k ≤ l ≤ n − k − 2`) of each point.
fn intervals(x: &[f64], t: &[f64], k: usize) -> Vec<usize> {
    let nk1 = t.len() - k - 1;
    let mut l = k;
    x.iter()
        .map(|&xi| {
            while xi >= t[l + 1] && l < nk1 - 1 {
                l += 1;
            }
            l
        })
        .collect()
}

/// The least-squares spline on knots `t`: its triangular factor, coefficients, per-point
/// residuals `Σⱼ (wᵢ (yᵢⱼ − g(xᵢ)ⱼ))²` and their sum.
struct Lsq {
    qr: Banded,
    c: Vec<Vec<f64>>,
    residuals: Vec<f64>,
    fp: f64,
}

fn residuals(data: &Data, t: &[f64], c: &[Vec<f64>]) -> (Vec<f64>, f64) {
    let k = data.k;
    let res: Vec<f64> = intervals(data.x, t, k)
        .into_iter()
        .enumerate()
        .map(|(i, l)| {
            let b = nonzero_basis(t, k, l, data.x[i]);
            (0..data.y[i].len())
                .map(|q| {
                    let g: f64 = b.iter().enumerate().map(|(r, v)| v * c[l - k + r][q]).sum();
                    (data.w[i] * (data.y[i][q] - g)).powi(2)
                })
                .sum()
        })
        .collect();
    let fp = res.iter().sum();
    (res, fp)
}

fn lsq(data: &Data, t: &[f64]) -> Lsq {
    let k = data.k;
    let nc = t.len() - k - 1;
    let mut qr = Banded::new(nc, k + 1, data.y[0].len());
    for (i, l) in intervals(data.x, t, k).into_iter().enumerate() {
        let wi = data.w[i];
        let h: Vec<f64> = nonzero_basis(t, k, l, data.x[i]).into_iter().map(|v| v * wi).collect();
        qr.rotate_in(h, l - k, data.y[i].iter().map(|v| v * wi).collect());
    }
    let c = qr.solve();
    let (residuals, fp) = residuals(data, t, &c);
    Lsq { qr, c, residuals, fp }
}

// ANCHOR: fpknot
/// FITPACK's `fpknot`: split the residuals into the knot intervals (a point on an interior knot
/// counts half to each side), pick the interval with the largest sum among those holding more
/// than one point, and insert a knot at the data point in its middle.
fn add_knot(x: &[f64], t: &[f64], k: usize, res: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
    let m = x.len();
    let bounds = &t[k..t.len() - k];
    let mut ix: Vec<usize> = bounds.iter().map(|&b| x.partition_point(|&v| v < b)).collect();
    let last = ix.len() - 1;
    ix[last] = m - 1;
    let mut parts: Vec<f64> = ix.windows(2).map(|w| res[w[0]..w[1]].iter().sum()).collect();
    for i in 1..last {
        let carry = res[ix[i]] / 2.0;
        parts[i - 1] += carry;
        parts[i] -= carry;
    }
    parts[last - 1] += res[m - 1];
    let mut best: Option<usize> = None;
    for i in 0..parts.len() {
        if ix[i + 1] - ix[i] > 1 && best.is_none_or(|b| parts[i] > parts[b]) {
            best = Some(i);
        }
    }
    let b = best.ok_or(ErrorsJSL::RuntimeError("generate_knots: no knot interval has room for another knot"))?;
    let knot = x[(ix[b] + ix[b + 1]).div_ceil(2)];
    let pos = t.partition_point(|&v| v < knot);
    let mut out = t.to_vec();
    out.insert(pos, knot);
    Ok(out)
}
// ANCHOR_END: fpknot

fn knot_sequence(data: &Data, nest: Option<usize>) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
    let (k, m, s) = (data.k, data.m(), data.s);
    let acc = s * TOL;
    let nest = match nest {
        None => (m + k + 1).max(2 * k + 3),
        Some(n) if n < 2 * (k + 1) => return Err(ErrorsJSL::InvalidInputRange("nest must be at least 2(k + 1)")),
        Some(n) => n,
    };
    let (nmin, nmax) = (2 * (k + 1), m + k + 1);
    let mut t: Vec<f64> = std::iter::repeat_n(data.xb, k + 1).chain(std::iter::repeat_n(data.xe, k + 1)).collect();
    let mut out = Vec::new();
    let (mut fp, mut nplus) = (0.0, 1usize);
    for _ in 0..m {
        out.push(t.clone());
        let fpold = fp;
        let fit = lsq(data, &t);
        fp = fit.fp;
        let fpms = fp - s;
        if fpms.abs() < acc || fpms < 0.0 {
            return Ok(out);
        }
        nplus = if t.len() == nmin {
            1
        } else {
            let delta = fpold - fp;
            let npl1 = if delta > acc { (nplus as f64 * fpms / delta) as usize } else { nplus * 2 };
            (nplus * 2).min(npl1.max(nplus / 2).max(1))
        };
        let mut res = fit.residuals;
        for j in 0..nplus {
            t = add_knot(data.x, &t, k, &res)?;
            if t.len() >= nmax {
                out.push(not_a_knot_knots(data.x, k));
                return Ok(out);
            }
            if t.len() >= nest {
                out.push(t);
                return Ok(out);
            }
            if j + 1 < nplus {
                res = lsq(data, &t).residuals;
            }
        }
    }
    Ok(out)
}

/// FITPACK's `fpdisc`: the jumps of the `k`-th derivatives of the B-splines at the interior
/// knots `t[k+1], …, t[n−k−2]`, row `j` holding `k + 2` entries from column `j`, scaled by
/// `(Δ/nrint)^k` with `Δ` the base interval and `nrint` its number of knot intervals.
fn discontinuity_jumps(t: &[f64], k: usize) -> Vec<Vec<f64>> {
    let n = t.len();
    let nrint = n - 2 * k - 1;
    let scale = ((t[n - k - 1] - t[k]) / nrint as f64).powi(k as i32);
    (0..nrint - 1)
        .map(|jj| {
            let j = jj + k + 1;
            (0..k + 2)
                .map(|ii| {
                    let i = jj + ii;
                    let prod: f64 = (0..k + 2).filter(|&q| i + q != j).map(|q| t[j] - t[i + q]).product();
                    (t[i + k + 1] - t[i]) / prod * scale
                })
                .collect()
        })
        .collect()
}

/// The smoothing spline for a given `p`: the data rows (already triangularised) with the jump
/// rows weighted by `1/p` rotated in; returns its coefficients and `fp − s`.
fn smoothing_for_p(data: &Data, t: &[f64], base: &Banded, jumps: &[Vec<f64>], p: f64) -> (Vec<Vec<f64>>, f64) {
    let k = data.k;
    let nc = base.r.len();
    let mut qr = Banded::new(nc, k + 2, base.z[0].len());
    for i in 0..nc {
        qr.r[i][..k + 1].copy_from_slice(&base.r[i]);
        qr.z[i].clone_from(&base.z[i]);
    }
    let d = base.z[0].len();
    for (j, row) in jumps.iter().enumerate() {
        qr.rotate_in(row.iter().map(|v| v / p).collect(), j, vec![0.0; d]);
    }
    let c = qr.solve();
    let fp = residuals(data, t, &c).1;
    (c, fp - data.s)
}

/// FITPACK's `fprati`: the zero of the rational function `(u p + v)/(p + w)` through three points
/// (`p₃` may be infinite).
fn fprati(p1: f64, f1: f64, p2: f64, f2: f64, p3: f64, f3: f64) -> f64 {
    let h1 = f1 * (f2 - f3);
    let h2 = f2 * (f3 - f1);
    let h3 = f3 * (f1 - f2);
    if p3.is_infinite() {
        return -(p2 * h1 + p1 * h2) / h3;
    }
    -(p1 * p2 * h3 + p2 * p3 * h1 + p1 * p3 * h2) / (p1 * h1 + p2 * h2 + p3 * h3)
}

/// The result of the smoothing fit: knots and coefficients (one column per coordinate), the
/// weighted sum of squared residuals, and whether `|fp − s| ≤ 0.001 s` was reached.
struct Fit {
    t: Vec<f64>,
    c: Vec<Vec<f64>>,
    fp: f64,
    converged: bool,
}

// ANCHOR: smoothing
fn fit_smoothing(data: &Data, t: Option<&[f64]>, nest: Option<usize>) -> Result<Fit, ErrorsJSL> {
    let (k, s) = (data.k, data.s);
    let acc = s * TOL;
    let t: Vec<f64> = match t {
        Some(t) => t.to_vec(),
        None => knot_sequence(data, nest)?.pop().expect("at least one knot vector"),
    };
    let base = lsq(data, &t);
    if t.len() == 2 * (k + 1) {
        return Ok(Fit { t, c: base.c, fp: base.fp, converged: true });
    }
    // f(p) = fp(p) − s decreases from the polynomial fit (p = 0) to the least-squares spline
    // (p = ∞); bracket the root and refine it by rational interpolation.
    let nc = t.len() - k - 1;
    let mut p = nc as f64 / base.qr.r.iter().map(|r| r[0]).sum::<f64>();
    let poly: Vec<f64> = std::iter::repeat_n(data.xb, k + 1).chain(std::iter::repeat_n(data.xe, k + 1)).collect();
    let (mut p1, mut f1) = (0.0, lsq(data, &poly).fp - s);
    let (mut p3, mut f3) = (f64::INFINITY, base.fp - s);
    let jumps = discontinuity_jumps(&t, k);
    let (con1, con4, con9) = (0.1, 0.04, 0.9);
    let (mut ich1, mut ich3) = (false, false);
    let mut last = (base.c.clone(), base.fp);
    for _ in 0..MAXIT {
        let (c, f2) = smoothing_for_p(data, &t, &base.qr, &jumps, p);
        let p2 = p;
        last = (c, f2 + s);
        if f2.abs() < acc {
            return Ok(Fit { t, c: last.0, fp: last.1, converged: true });
        }
        if !ich3 {
            if f2 - f3 <= acc {
                // p is too large.
                (p3, f3) = (p2, f2);
                p *= con4;
                if p <= p1 {
                    p = p1 * con9 + p2 * con1;
                }
                continue;
            }
            if f2 < 0.0 {
                ich3 = true;
            }
        }
        if !ich1 {
            if f1 - f2 <= acc {
                // p is too small.
                (p1, f1) = (p2, f2);
                p /= con4;
                if p3.is_finite() && p <= p3 {
                    p = p2 * con1 + p3 * con9;
                }
                continue;
            }
            if f2 > 0.0 {
                ich1 = true;
            }
        }
        if f1 <= f2 || f2 <= f3 {
            // f is not decreasing: s is too small for the arithmetic.
            break;
        }
        p = fprati(p1, f1, p2, f2, p3, f3);
        if f2 < 0.0 {
            (p3, f3) = (p2, f2);
        } else {
            (p1, f1) = (p2, f2);
        }
    }
    Ok(Fit { t, c: last.0, fp: last.1, converged: false })
}
// ANCHOR_END: smoothing

fn validate<'a>(x: &'a [f64], y: Vec<Vec<f64>>, options: &SplrepOptions) -> Result<Data<'a>, ErrorsJSL> {
    let m = x.len();
    let k = options.k;
    if y.len() != m {
        return Err(ErrorsJSL::IncompatibleArraySizes((y.len(), m)));
    }
    if k == 0 || m <= k {
        return Err(ErrorsJSL::InvalidInputRange("make_splrep: need degree k ≥ 1 and more than k points"));
    }
    if x.windows(2).any(|p| !(p[1] >= p[0])) || x.iter().any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("make_splrep: x must be finite and non-decreasing"));
    }
    if !(options.s >= 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("make_splrep: s must be non-negative"));
    }
    let w = match options.w {
        Some(w) if w.len() != m => return Err(ErrorsJSL::IncompatibleArraySizes((w.len(), m))),
        Some(w) if w.iter().any(|v| !(*v >= 0.0)) || w.iter().sum::<f64>() == 0.0 => {
            return Err(ErrorsJSL::InvalidInputRange("make_splrep: weights must be non-negative and not all zero"));
        }
        Some(w) => w.to_vec(),
        None => vec![1.0; m],
    };
    let xb = options.xb.unwrap_or(x[0]);
    let xe = options.xe.unwrap_or(x[m - 1]);
    if !(xb <= x[0] && xe >= x[m - 1] && xb < xe) {
        return Err(ErrorsJSL::InvalidInputRange("make_splrep: need xb ≤ x[0] and xe ≥ x[m − 1]"));
    }
    Ok(Data { x, y, w, k, s: options.s, xb, xe })
}

// ANCHOR: generate_knots
/// The successive knot vectors FITPACK tries (SciPy's `generate_knots`): from the `2(k + 1)` knots
/// of a polynomial, knots are added where the least-squares spline fits worst — `1, 2, 4, …` per
/// round, fewer as the residual nears `s` — until the least-squares residual is at most `s`, the
/// limit `nest` is reached, or the knots become those of the interpolating spline. The last
/// vector is the one [`make_splrep`] uses.
pub fn generate_knots(x: &[f64], y: &[f64], options: &SplrepOptions) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
    if options.s == 0.0 {
        if options.w.is_some() || options.nest.is_some() {
            return Err(ErrorsJSL::InvalidInputRange("generate_knots: s = 0 is interpolation, without weights or nest"));
        }
        return Ok(vec![not_a_knot_knots(x, options.k)]);
    }
    let data = validate(x, y.iter().map(|&v| vec![v]).collect(), options)?;
    knot_sequence(&data, options.nest)
}
// ANCHOR_END: generate_knots

/// The result of [`make_splrep`] and [`make_splprep`] beyond the spline itself.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SplrepInfo {
    /// The weighted sum of squared residuals `Σ (wᵢ (yᵢ − g(xᵢ)))²`.
    pub fp: f64,
    /// Whether `|fp − s| ≤ 0.001 s` was reached (or the spline is the least-squares spline or
    /// polynomial with `fp ≤ s`); otherwise SciPy warns and returns the last iterate.
    pub converged: bool,
}

// ANCHOR: make_splrep
/// The smoothing spline of degree `k` with automatically placed knots (SciPy's `make_splrep`,
/// FITPACK's `curfit`): of all splines with `Σ (wᵢ (yᵢ − g(xᵢ)))² ≤ s`, the one whose `k`-th
/// derivative jumps least at its interior knots. `s = 0` gives the interpolating spline with
/// not-a-knot ends; a large `s` gives the least-squares polynomial. With weights `1/σᵢ`, a good
/// `s` lies in `m ± √(2m)`. `t` fixes the knots instead of generating them.
pub fn make_splrep(x: &[f64], y: &[f64], t: Option<&[f64]>, options: &SplrepOptions) -> Result<(BSpline, SplrepInfo), ErrorsJSL> {
    if options.s == 0.0 {
        if t.is_some() || options.w.is_some() || options.nest.is_some() {
            return Err(ErrorsJSL::InvalidInputRange("make_splrep: s = 0 is interpolation, without t, weights or nest"));
        }
        let spl = make_interp_spline(x, y, options.k, &BoundaryCondition::NotAKnot)?;
        return Ok((spl, SplrepInfo { fp: 0.0, converged: true }));
    }
    if t.is_some() && options.nest.is_some() {
        return Err(ErrorsJSL::InvalidInputRange("make_splrep: give either t or nest"));
    }
    let data = validate(x, y.iter().map(|&v| vec![v]).collect(), options)?;
    let fit = fit_smoothing(&data, t, options.nest)?;
    let c = fit.c.iter().map(|r| r[0]).collect();
    Ok((BSpline::new(fit.t, c, data.k)?, SplrepInfo { fp: fit.fp, converged: fit.converged }))
}
// ANCHOR_END: make_splrep

/// A smoothing parametric spline curve through `points` (one row per point, one column per
/// coordinate) with shared, automatically placed knots (SciPy's `make_splprep`, FITPACK's
/// `parcur`). The parameter `u` defaults to the cumulative chord length scaled to `[0, 1]`; the
/// smoothing criterion sums the squared residuals over all coordinates. Returns `u` and one
/// spline per coordinate, all on the same knots.
pub fn make_splprep(points: &[Vec<f64>], u: Option<&[f64]>, t: Option<&[f64]>, options: &SplrepOptions) -> Result<(Vec<f64>, Vec<BSpline>, SplrepInfo), ErrorsJSL> {
    let m = points.len();
    if m < 2 || points.iter().any(|p| p.len() != points[0].len() || p.is_empty()) {
        return Err(ErrorsJSL::InvalidInputRange("make_splprep: need at least two points of equal, non-zero dimension"));
    }
    let u: Vec<f64> = match u {
        Some(u) => u.to_vec(),
        None => {
            let mut v = vec![0.0; m];
            for i in 1..m {
                v[i] = v[i - 1] + points[i].iter().zip(&points[i - 1]).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            }
            let total = v[m - 1];
            if total <= 0.0 {
                return Err(ErrorsJSL::InvalidInputRange("make_splprep: the points must not all coincide"));
            }
            v.iter().map(|a| a / total).collect()
        }
    };
    let d = points[0].len();
    if options.s == 0.0 {
        if t.is_some() || options.w.is_some() || options.nest.is_some() {
            return Err(ErrorsJSL::InvalidInputRange("make_splprep: s = 0 is interpolation, without t, weights or nest"));
        }
        let splines = (0..d)
            .map(|j| make_interp_spline(&u, &points.iter().map(|p| p[j]).collect::<Vec<_>>(), options.k, &BoundaryCondition::NotAKnot))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((u, splines, SplrepInfo { fp: 0.0, converged: true }));
    }
    if t.is_some() && options.nest.is_some() {
        return Err(ErrorsJSL::InvalidInputRange("make_splprep: give either t or nest"));
    }
    let data = validate(&u, points.to_vec(), options)?;
    let fit = fit_smoothing(&data, t, options.nest)?;
    let splines = (0..d).map(|j| BSpline::new(fit.t.clone(), fit.c.iter().map(|r| r[j]).collect(), data.k)).collect::<Result<Vec<_>, _>>()?;
    let info = SplrepInfo { fp: fit.fp, converged: fit.converged };
    Ok((u, splines, info))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noisy(m: usize) -> (Vec<f64>, Vec<f64>) {
        // A deterministic pseudo-random sequence.
        let mut state = 12345u64;
        let x: Vec<f64> = (0..m).map(|i| -3.0 + 6.0 * i as f64 / (m - 1) as f64).collect();
        let y = x
            .iter()
            .map(|&v| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                (-v * v).exp() + 0.2 * u
            })
            .collect();
        (x, y)
    }

    #[test]
    fn residual_meets_s_and_limits() {
        let (x, y) = noisy(60);
        for s in [0.05, 0.2, 0.5] {
            let (spl, info) = make_splrep(&x, &y, None, &SplrepOptions { s, ..Default::default() }).unwrap();
            assert!(info.converged, "s = {s}");
            let fp: f64 = x.iter().zip(&y).map(|(&a, &b)| (b - spl.evaluate(a)).powi(2)).sum();
            assert!((fp - s).abs() <= 1e-3 * s, "s = {s}: fp = {fp}");
            assert!((info.fp - fp).abs() < 1e-12);
        }
        // A huge s gives the least-squares cubic polynomial: no interior knots.
        let (spl, _) = make_splrep(&x, &y, None, &SplrepOptions { s: 1e6, ..Default::default() }).unwrap();
        assert_eq!(spl.t.len(), 8);
        // s = 0 interpolates.
        let (spl, _) = make_splrep(&x, &y, None, &SplrepOptions::default()).unwrap();
        assert!((spl.evaluate(x[17]) - y[17]).abs() < 1e-12);
    }

    #[test]
    fn knot_sequence_grows_and_ends_with_the_used_knots() {
        let (x, y) = noisy(50);
        let opts = SplrepOptions { s: 0.1, ..Default::default() };
        let seq = generate_knots(&x, &y, &opts).unwrap();
        assert_eq!(seq[0].len(), 8);
        assert!(seq.windows(2).all(|w| w[1].len() > w[0].len()));
        let (spl, _) = make_splrep(&x, &y, None, &opts).unwrap();
        assert_eq!(&spl.t, seq.last().unwrap());
        // Knots are data points.
        for v in &spl.t[4..spl.t.len() - 4] {
            assert!(x.contains(v));
        }
    }

    #[test]
    fn parametric_curve_is_smoothed_jointly() {
        let m = 40;
        let pts: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                let a = i as f64 * 6.0 / m as f64;
                vec![a.cos() + 0.01 * (7.0 * a).sin(), a.sin() + 0.01 * (5.0 * a).cos()]
            })
            .collect();
        let s = 0.002;
        let (u, sp, info) = make_splprep(&pts, None, None, &SplrepOptions { s, ..Default::default() }).unwrap();
        assert!(info.converged);
        let fp: f64 = (0..m).map(|i| (0..2).map(|j| (pts[i][j] - sp[j].evaluate(u[i])).powi(2)).sum::<f64>()).sum();
        assert!((fp - s).abs() <= 1e-3 * s, "{fp}");
        assert_eq!(sp[0].t, sp[1].t);
    }
}

#[cfg(test)]
mod banded_check {
    use super::Banded;
    #[test]
    fn normal_equations() {
        let nc = 6;
        let bw = 5;
        let rows: Vec<(Vec<f64>, usize)> = vec![
            (vec![1.0, 2.0, 0.5, 0.3], 0), (vec![0.2, 1.0, 0.7, 0.1], 1), (vec![0.3, 0.4, 1.0, 0.2], 2),
            (vec![0.9, 0.1, 0.2, 0.6], 2), (vec![0.5, 0.5, 0.5, 0.5], 2), (vec![0.1, 0.3, 0.8, 0.4], 2),
            (vec![1.0, -2.0, 3.0, -1.0, 0.5], 0), (vec![0.4, -1.0, 2.0, -3.0, 1.5], 1),
        ];
        let mut b = Banded::new(nc, bw, 1);
        let mut ata = vec![vec![0.0; nc]; nc];
        for (h, off) in &rows {
            for (a, &u) in h.iter().enumerate() {
                for (c, &v) in h.iter().enumerate() {
                    ata[off + a][off + c] += u * v;
                }
            }
            b.rotate_in(h.clone(), *off, vec![1.0]);
        }
        for i in 0..nc {
            for j in 0..nc {
                let mut s = 0.0;
                for q in 0..nc {
                    let rq = |r: usize, c: usize| if c >= r && c - r < bw { b.r[r][c - r] } else { 0.0 };
                    s += rq(q, i) * rq(q, j);
                }
                assert!((s - ata[i][j]).abs() < 1e-12, "{i} {j}: {s} vs {}", ata[i][j]);
            }
        }
    }
}
