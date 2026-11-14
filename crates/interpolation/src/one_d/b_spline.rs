//! B-splines of any degree, following `scipy.interpolate`: the [`BSpline`] type (evaluation,
//! derivatives, antiderivatives and integrals), interpolating splines with a choice of end
//! conditions ([`make_interp_spline`]), least-squares splines on given knots
//! ([`make_lsq_spline`]) and parametric interpolating curves ([`make_interp_curve`]).
//!
//! A spline of degree `k` with knots `t₀ ≤ t₁ ≤ … ≤ t_{n+k}` is a sum `s(x) = Σ cⱼ Bⱼ,ₖ(x)` of the
//! B-spline basis functions, each a piecewise polynomial of degree `k` that is non-zero only on
//! `[tⱼ, tⱼ₊ₖ₊₁)`. They are evaluated by the Cox-de Boor recurrence, and an interpolating spline
//! is found by solving a banded linear system for the coefficients.
//!
//! See the book entry [B-splines and smoothing splines](https://jwkunz.github.io/lib_jsl/book/interpolation/b-splines.html).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_linalg::{LeastSquaresSvd, Solve};
use ndarray::{Array1, Array2};

/// What a [`BSpline`] returns outside its base interval `[t_k, t_n]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extrapolate {
    /// Continue the first or last polynomial piece (SciPy's default, `extrapolate=True`).
    Yes,
    /// Return NaN.
    No,
    /// Wrap round periodically (`extrapolate='periodic'`).
    Periodic,
}

/// A spline `s(x) = Σ cⱼ Bⱼ,ₖ(x)` of degree `k` in the B-spline basis (SciPy's `BSpline`).
#[derive(Clone, Debug, PartialEq)]
pub struct BSpline {
    /// The knots, non-decreasing; `n + k + 1` of them for `n` coefficients.
    pub t: Vec<f64>,
    /// The coefficients.
    pub c: Vec<f64>,
    /// The degree.
    pub k: usize,
    /// Behaviour outside `[t[k], t[n]]`.
    pub extrapolate: Extrapolate,
}

// ANCHOR: basis
/// The `k + 1` B-splines of degree `k` that are non-zero on the knot interval
/// `t[l] ≤ x < t[l + 1]`, namely `B_{l−k}, …, B_l`, by the Cox-de Boor recurrence (de Boor's
/// `bsplvb`). Each degree is built from the previous one; the values are non-negative and sum
/// to one.
pub fn nonzero_basis(t: &[f64], k: usize, l: usize, x: f64) -> Vec<f64> {
    let mut b = vec![0.0; k + 1];
    let (mut left, mut right) = (vec![0.0; k + 1], vec![0.0; k + 1]);
    b[0] = 1.0;
    for j in 1..=k {
        left[j] = x - t[l + 1 - j];
        right[j] = t[l + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = b[r] / (right[r + 1] + left[j - r]);
            b[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        b[j] = saved;
    }
    b
}
// ANCHOR_END: basis

/// The `nu`-th derivatives of the `k + 1` B-splines that are non-zero on interval `l`, from the
/// basis of degree `k − nu` and the derivative formula
/// `B'ⱼ,ₖ = k (Bⱼ,ₖ₋₁ / (tⱼ₊ₖ − tⱼ) − Bⱼ₊₁,ₖ₋₁ / (tⱼ₊ₖ₊₁ − tⱼ₊₁))`.
pub fn nonzero_basis_derivatives(t: &[f64], k: usize, l: usize, x: f64, nu: usize) -> Vec<f64> {
    if nu > k {
        return vec![0.0; k + 1];
    }
    let mut v = nonzero_basis(t, k - nu, l, x);
    for d in k - nu + 1..=k {
        // v holds degree d − 1 values for indices l − (d − 1) ..= l.
        let old = |j: isize| -> f64 {
            let i = j - (l as isize - (d as isize - 1));
            if i < 0 || i as usize >= v.len() { 0.0 } else { v[i as usize] }
        };
        let mut next = vec![0.0; d + 1];
        for (i, slot) in next.iter_mut().enumerate() {
            let j = l as isize - d as isize + i as isize;
            let ju = j as usize;
            let a = t[ju + d] - t[ju];
            let b = t[ju + d + 1] - t[ju + 1];
            let ta = if a > 0.0 { old(j) / a } else { 0.0 };
            let tb = if b > 0.0 { old(j + 1) / b } else { 0.0 };
            *slot = d as f64 * (ta - tb);
        }
        v = next;
    }
    v
}

impl BSpline {
    /// A spline from knots, coefficients and degree; extrapolates by default.
    pub fn new(t: Vec<f64>, c: Vec<f64>, k: usize) -> Result<Self, ErrorsJSL> {
        if t.len() < 2 * k + 2 {
            return Err(ErrorsJSL::InvalidInputRange("a spline of degree k needs at least 2k + 2 knots"));
        }
        if t.windows(2).any(|w| w[1] < w[0] || !w[0].is_finite()) {
            return Err(ErrorsJSL::InvalidInputRange("knots must be finite and non-decreasing"));
        }
        let n = t.len() - k - 1;
        if c.len() != n {
            return Err(ErrorsJSL::IncompatibleArraySizes((c.len(), n)));
        }
        if t[k] >= t[n] {
            return Err(ErrorsJSL::InvalidInputRange("the base interval [t[k], t[n]] is empty"));
        }
        Ok(BSpline { t, c, k, extrapolate: Extrapolate::Yes })
    }

    /// The number of coefficients `n`.
    pub fn len(&self) -> usize {
        self.c.len()
    }

    /// `true` when the spline has no coefficients (never, for a valid spline).
    pub fn is_empty(&self) -> bool {
        self.c.is_empty()
    }

    /// The knot interval `l` with `t[l] ≤ x < t[l + 1]`, `k ≤ l < n`, after applying the
    /// extrapolation rule; `None` for NaN, or outside the base interval without extrapolation.
    fn interval(&self, x: f64) -> Option<(usize, f64)> {
        let (k, n) = (self.k, self.c.len());
        let (lo, hi) = (self.t[k], self.t[n]);
        let mut x = x;
        if x.is_nan() {
            return None;
        }
        match self.extrapolate {
            Extrapolate::No if x < lo || x > hi => return None,
            Extrapolate::Periodic => x = lo + (x - lo).rem_euclid(hi - lo),
            _ => {}
        }
        // Binary search on t[k..=n] for the last t[l] ≤ x, clamped to a non-empty interval.
        let mut l = match self.t[k..=n].partition_point(|&v| v <= x) {
            0 => k,
            p => (k + p - 1).min(n - 1),
        };
        while l > k && self.t[l] == self.t[l + 1] {
            l -= 1;
        }
        Some((l, x))
    }

    // ANCHOR: evaluate
    /// The value `s(x)` (SciPy's `BSpline.__call__`).
    pub fn evaluate(&self, x: f64) -> f64 {
        self.evaluate_derivative(x, 0)
    }

    /// The `nu`-th derivative `s⁽ⁿᵘ⁾(x)`, zero when `nu > k`.
    pub fn evaluate_derivative(&self, x: f64, nu: usize) -> f64 {
        let Some((l, x)) = self.interval(x) else { return f64::NAN };
        let b = if nu == 0 { nonzero_basis(&self.t, self.k, l, x) } else { nonzero_basis_derivatives(&self.t, self.k, l, x, nu) };
        b.iter().enumerate().map(|(r, v)| self.c[l - self.k + r] * v).sum()
    }
    // ANCHOR_END: evaluate

    /// The values at many points.
    pub fn evaluate_many(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.evaluate(v)).collect()
    }

    /// The derivative of order `nu` as a spline of degree `k − nu` on the knots with `nu` removed
    /// from each end (SciPy's `BSpline.derivative`):
    /// `c'ⱼ = k (cⱼ₊₁ − cⱼ) / (tⱼ₊ₖ₊₁ − tⱼ₊₁)`.
    pub fn derivative(&self, nu: usize) -> Result<BSpline, ErrorsJSL> {
        if nu > self.k {
            return Err(ErrorsJSL::InvalidInputRange("the order of the derivative exceeds the degree"));
        }
        let mut s = self.clone();
        for _ in 0..nu {
            let k = s.k;
            let n = s.c.len();
            let c: Vec<f64> = (0..n - 1)
                .map(|j| {
                    let d = s.t[j + k + 1] - s.t[j + 1];
                    if d > 0.0 { k as f64 * (s.c[j + 1] - s.c[j]) / d } else { 0.0 }
                })
                .collect();
            s = BSpline { t: s.t[1..s.t.len() - 1].to_vec(), c, k: k - 1, extrapolate: s.extrapolate };
        }
        Ok(s)
    }

    /// The antiderivative of order `nu`, a spline of degree `k + nu` that vanishes at `t[0]`
    /// (SciPy's `BSpline.antiderivative`): one end knot is added at each end and
    /// `Cⱼ₊₁ = Σᵢ≤ⱼ cᵢ (tᵢ₊ₖ₊₁ − tᵢ) / (k + 1)`.
    pub fn antiderivative(&self, nu: usize) -> BSpline {
        let mut s = self.clone();
        for _ in 0..nu {
            let k = s.k;
            let mut c = Vec::with_capacity(s.c.len() + 1);
            c.push(0.0);
            let mut acc = 0.0;
            for (j, &cj) in s.c.iter().enumerate() {
                acc += cj * (s.t[j + k + 1] - s.t[j]) / (k + 1) as f64;
                c.push(acc);
            }
            let mut t = Vec::with_capacity(s.t.len() + 2);
            t.push(s.t[0]);
            t.extend_from_slice(&s.t);
            t.push(*s.t.last().expect("non-empty"));
            s = BSpline { t, c, k: k + 1, extrapolate: if s.extrapolate == Extrapolate::Periodic { Extrapolate::Yes } else { s.extrapolate } };
        }
        s
    }

    /// `∫ₐᵇ s(x) dx` (SciPy's `BSpline.integrate`). With [`Extrapolate::No`] the integrand is zero
    /// outside the base interval; with [`Extrapolate::Periodic`] whole periods are counted.
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        let (k, n) = (self.k, self.c.len());
        let (lo, hi) = (self.t[k], self.t[n]);
        let anti = self.antiderivative(1);
        let f = |x: f64| BSpline { extrapolate: Extrapolate::Yes, ..anti.clone() }.evaluate(x);
        match self.extrapolate {
            Extrapolate::Yes => f(b) - f(a),
            Extrapolate::No => f(b.clamp(lo, hi)) - f(a.clamp(lo, hi)),
            Extrapolate::Periodic => {
                let period = hi - lo;
                let whole = f(hi) - f(lo);
                let reduce = |x: f64| {
                    let m = ((x - lo) / period).floor();
                    m * whole + f(lo + (x - lo) - m * period) - f(lo)
                };
                reduce(b) - reduce(a)
            }
        }
    }

    /// The B-spline design matrix `Mᵢⱼ = Bⱼ,ₖ(xᵢ)` for points in the base interval (SciPy's
    /// `BSpline.design_matrix`, as a dense array).
    pub fn design_matrix(x: &[f64], t: &[f64], k: usize) -> Result<Array2<f64>, ErrorsJSL> {
        let n = t.len().checked_sub(k + 1).ok_or(ErrorsJSL::InvalidInputRange("too few knots"))?;
        let probe = BSpline::new(t.to_vec(), vec![0.0; n], k)?;
        let mut m = Array2::zeros((x.len(), n));
        for (i, &xi) in x.iter().enumerate() {
            if xi < t[k] || xi > t[n] {
                return Err(ErrorsJSL::InvalidInputRange("design-matrix points must lie in [t[k], t[n]]"));
            }
            let (l, _) = probe.interval(xi).expect("finite point");
            for (r, v) in nonzero_basis(t, k, l, xi).into_iter().enumerate() {
                m[(i, l - k + r)] = v;
            }
        }
        Ok(m)
    }
}

/// End conditions for [`make_interp_spline`] (SciPy's `bc_type`).
#[derive(Clone, Debug, PartialEq)]
pub enum BoundaryCondition {
    /// No derivative conditions; the knots are chosen so that the first and last interior
    /// polynomial pieces join without a break ("not-a-knot", SciPy's default).
    NotAKnot,
    /// Zero second derivative at both ends (cubic splines only).
    Natural,
    /// Zero first derivative at both ends (cubic splines only).
    Clamped,
    /// Matching derivatives of orders `1..k` at both ends; requires `y[0] == y[n−1]`.
    Periodic,
    /// Given derivatives `(order, value)` at the left and right ends; there must be `k − 1` in
    /// total (for example `[(1, a)]` and `[(1, b)]` for a cubic spline with given slopes).
    Derivatives { left: Vec<(usize, f64)>, right: Vec<(usize, f64)> },
}

/// Knots for not-a-knot interpolation (SciPy's `_not_a_knot`): for odd `k` the data points
/// without the first and last `(k − 1)/2` interior ones; for even `k` the midpoints between data
/// points, likewise thinned; and `k + 1` copies of each end point.
fn not_a_knot_knots(x: &[f64], k: usize) -> Vec<f64> {
    let interior: Vec<f64> = if !k.is_multiple_of(2) { x.to_vec() } else { x.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect() };
    let k2 = if !k.is_multiple_of(2) { k.div_ceil(2) } else { k / 2 };
    let mut t = vec![x[0]; k + 1];
    if interior.len() > 2 * k2 {
        t.extend_from_slice(&interior[k2..interior.len() - k2]);
    }
    t.extend(std::iter::repeat_n(x[x.len() - 1], k + 1));
    t
}

/// Knots on a circle for periodic interpolation (SciPy's `_periodic_knots`).
fn periodic_knots(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    let mut xc = x.to_vec();
    if k.is_multiple_of(2) {
        let dx: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        for i in 1..n - 1 {
            xc[i] -= dx[i - 1] / 2.0;
        }
    }
    let dx: Vec<f64> = xc.windows(2).map(|w| w[1] - w[0]).collect();
    let mut t = vec![0.0; n + 2 * k];
    t[k..k + n].copy_from_slice(&xc);
    let len = t.len();
    for i in 0..k {
        t[k - i - 1] = t[k - i] - dx[n - 2 - (i % (n - 1))];
        t[len - k + i] = t[len - k + i - 1] + dx[i % (n - 1)];
    }
    t
}

/// Solves a banded system with partial pivoting. `rows[i]` holds the non-zero entries of row
/// `i` as `(first column, values)`; the band below the diagonal has width at most `kl`.
fn solve_banded(n: usize, rows: &[(usize, Vec<f64>)], rhs: &[f64]) -> Result<Vec<f64>, ErrorsJSL> {
    let kl = rows.iter().enumerate().map(|(i, (c0, _))| i.saturating_sub(*c0)).max().unwrap_or(0);
    let ku = rows.iter().enumerate().map(|(i, (c0, v))| (c0 + v.len()).saturating_sub(i + 1)).max().unwrap_or(0);
    // Row i stores columns i − kl ..= i + kl + ku (the upper band grows by kl through pivoting).
    let w = 2 * kl + ku + 1;
    let mut band = vec![0.0; n * w];
    let at = |i: usize, j: usize| i * w + (j + kl - i);
    for (i, (c0, vals)) in rows.iter().enumerate() {
        for (o, &v) in vals.iter().enumerate() {
            band[at(i, c0 + o)] = v;
        }
    }
    let mut b = rhs.to_vec();
    let scale = band.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
    for j in 0..n {
        let last = (j + kl).min(n - 1);
        let p = (j..=last).max_by(|&a, &c| band[at(a, j)].abs().total_cmp(&band[at(c, j)].abs())).expect("non-empty");
        if band[at(p, j)].abs() <= f64::EPSILON * scale * 1e-3 {
            return Err(ErrorsJSL::RuntimeError("the interpolation matrix is singular (are the knots compatible with the data?)"));
        }
        let cmax = (j + kl + ku).min(n - 1);
        if p != j {
            for c in j..=cmax {
                band.swap(at(p, c), at(j, c));
            }
            b.swap(p, j);
        }
        let piv = band[at(j, j)];
        for i in j + 1..=last {
            let f = band[at(i, j)] / piv;
            if f != 0.0 {
                for c in j..=cmax {
                    band[at(i, c)] -= f * band[at(j, c)];
                }
                b[i] -= f * b[j];
            }
        }
    }
    for i in (0..n).rev() {
        let cmax = (i + kl + ku).min(n - 1);
        let s: f64 = (i + 1..=cmax).map(|c| band[at(i, c)] * b[c]).sum();
        b[i] = (b[i] - s) / band[at(i, i)];
    }
    Ok(b)
}

fn check_data(x: &[f64], y: &[f64]) -> Result<(), ErrorsJSL> {
    if x.len() != y.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((x.len(), y.len())));
    }
    if x.windows(2).any(|w| w[1] <= w[0]) || x.iter().chain(y).any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("x must be strictly increasing and the data finite"));
    }
    Ok(())
}

// ANCHOR: make_interp_spline
/// The interpolating spline of degree `k` through `(x, y)` (SciPy's `make_interp_spline`).
///
/// For `k = 0` and `k = 1` the spline is piecewise constant or linear. Otherwise the knots are
/// chosen from the data (not-a-knot, or the data points with `k` copies of each end when
/// derivatives are given), and the `n` interpolation conditions `s(xᵢ) = yᵢ` together with the
/// end conditions form a banded linear system for the coefficients. Periodic splines solve a
/// cyclic system for `n − 1` free coefficients.
pub fn make_interp_spline(x: &[f64], y: &[f64], k: usize, bc: &BoundaryCondition) -> Result<BSpline, ErrorsJSL> {
    check_data(x, y)?;
    let n = x.len();
    if n < k + 1 {
        return Err(ErrorsJSL::InvalidInputRange("need at least k + 1 data points"));
    }
    let derivatives = match bc {
        BoundaryCondition::NotAKnot => None,
        BoundaryCondition::Natural => Some((vec![(2, 0.0)], vec![(2, 0.0)])),
        BoundaryCondition::Clamped => Some((vec![(1, 0.0)], vec![(1, 0.0)])),
        BoundaryCondition::Derivatives { left, right } => Some((left.clone(), right.clone())),
        BoundaryCondition::Periodic => return periodic_spline(x, y, k),
    };
    if k == 0 || k == 1 {
        if derivatives.is_some() {
            return Err(ErrorsJSL::InvalidInputRange("degrees 0 and 1 take no end conditions"));
        }
        let mut t = x.to_vec();
        if k == 0 {
            t.push(x[n - 1]);
        } else {
            t.insert(0, x[0]);
            t.push(x[n - 1]);
        }
        let mut s = BSpline::new(t, y.to_vec(), k)?;
        s.extrapolate = Extrapolate::Yes;
        return Ok(s);
    }
    let (left, right) = derivatives.unwrap_or_default();
    let t = if left.is_empty() && right.is_empty() {
        not_a_knot_knots(x, k)
    } else {
        let mut t = vec![x[0]; k];
        t.extend_from_slice(x);
        t.extend(std::iter::repeat_n(x[n - 1], k));
        t
    };
    let nt = t.len() - k - 1;
    if nt != n + left.len() + right.len() {
        return Err(ErrorsJSL::InvalidInputRange("the number of end conditions must be k − 1 (one at each end for a cubic)"));
    }
    if left.iter().chain(&right).any(|&(nu, _)| nu == 0 || nu > k) {
        return Err(ErrorsJSL::InvalidInputRange("end-condition derivative orders must lie in 1..=k"));
    }
    let probe = BSpline::new(t.clone(), vec![0.0; nt], k)?;
    let mut rows = Vec::with_capacity(nt);
    let mut rhs = Vec::with_capacity(nt);
    let (l0, _) = probe.interval(x[0]).expect("finite");
    for &(nu, v) in &left {
        rows.push((l0 - k, nonzero_basis_derivatives(&t, k, l0, x[0], nu)));
        rhs.push(v);
    }
    for (&xi, &yi) in x.iter().zip(y) {
        let (l, _) = probe.interval(xi).expect("finite");
        rows.push((l - k, nonzero_basis(&t, k, l, xi)));
        rhs.push(yi);
    }
    let (ln, _) = probe.interval(x[n - 1]).expect("finite");
    for &(nu, v) in &right {
        rows.push((ln - k, nonzero_basis_derivatives(&t, k, ln, x[n - 1], nu)));
        rhs.push(v);
    }
    let c = solve_banded(nt, &rows, &rhs)?;
    BSpline::new(t, c, k)
}
// ANCHOR_END: make_interp_spline

/// Periodic interpolation: `n − 1` free coefficients, the last `k` repeating the first, so that
/// every derivative matches across the ends.
fn periodic_spline(x: &[f64], y: &[f64], k: usize) -> Result<BSpline, ErrorsJSL> {
    let n = x.len();
    if (y[0] - y[n - 1]).abs() > 1e-12 * (1.0 + y[0].abs()) {
        return Err(ErrorsJSL::InvalidInputRange("periodic interpolation needs y[0] == y[n − 1]"));
    }
    if n < 2 {
        return Err(ErrorsJSL::InvalidInputRange("need at least two points"));
    }
    let t = periodic_knots(x, k);
    let nt = t.len() - k - 1; // = n + k − 1
    let free = n - 1;
    let probe = BSpline::new(t.clone(), vec![0.0; nt], k)?;
    let mut a = Array2::<f64>::zeros((free, free));
    for i in 0..free {
        let (l, _) = probe.interval(x[i]).expect("finite");
        for (r, v) in nonzero_basis(&t, k, l, x[i]).into_iter().enumerate() {
            a[(i, (l - k + r) % free)] += v;
        }
    }
    let sol = a.solve(&Array1::from(y[..free].to_vec())).map_err(|_| ErrorsJSL::RuntimeError("the periodic interpolation matrix is singular"))?;
    let c: Vec<f64> = (0..nt).map(|j| sol[j % free]).collect();
    let mut s = BSpline::new(t, c, k)?;
    s.extrapolate = Extrapolate::Periodic;
    Ok(s)
}

/// The least-squares spline of degree `k` on the knots `t`: minimises `Σ (wᵢ (yᵢ − s(xᵢ)))²`
/// (SciPy's `make_lsq_spline`; note that the weights multiply the residuals, not their squares). The data must satisfy the Schoenberg-Whitney conditions (each
/// basis function must be non-zero at some data point) for the solution to be unique.
pub fn make_lsq_spline(x: &[f64], y: &[f64], t: &[f64], k: usize, w: Option<&[f64]>) -> Result<BSpline, ErrorsJSL> {
    if x.len() != y.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((x.len(), y.len())));
    }
    if x.windows(2).any(|p| p[1] < p[0]) {
        return Err(ErrorsJSL::InvalidInputRange("x must be non-decreasing"));
    }
    let mut m = BSpline::design_matrix(x, t, k)?;
    let mut rhs = Array1::from(y.to_vec());
    if let Some(w) = w {
        if w.len() != x.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((w.len(), x.len())));
        }
        for (i, &wi) in w.iter().enumerate() {
            m.row_mut(i).mapv_inplace(|v| v * wi);
            rhs[i] *= wi;
        }
    }
    let ls = m.least_squares(&rhs).map_err(|_| ErrorsJSL::RuntimeError("least-squares spline failed"))?;
    if ls.rank < m.ncols() {
        return Err(ErrorsJSL::InvalidInputRange("the knots and data violate the Schoenberg-Whitney conditions"));
    }
    BSpline::new(t.to_vec(), ls.solution.to_vec(), k)
}

/// An interpolating parametric curve through the rows of `points` (an `n × d` array): the
/// parameter is the cumulative chord length normalised to `[0, 1]` (as `splprep` chooses it),
/// and each coordinate is interpolated with [`make_interp_spline`]. Returns the parameter values
/// and one spline per coordinate.
pub fn make_interp_curve(points: &Array2<f64>, k: usize, bc: &BoundaryCondition) -> Result<(Vec<f64>, Vec<BSpline>), ErrorsJSL> {
    let n = points.nrows();
    let mut u = vec![0.0; n];
    for i in 1..n {
        let d = points.row(i).iter().zip(points.row(i - 1).iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
        u[i] = u[i - 1] + d;
    }
    let total = u[n - 1];
    if total <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("the points must not all coincide"));
    }
    u.iter_mut().for_each(|v| *v /= total);
    let splines = (0..points.ncols()).map(|j| make_interp_spline(&u, &points.column(j).to_vec(), k, bc)).collect::<Result<Vec<_>, _>>()?;
    Ok((u, splines))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_is_a_partition_of_unity() {
        let t = [0.0, 0.0, 0.0, 0.0, 1.0, 2.5, 3.0, 3.0, 3.0, 3.0];
        for x in [0.0, 0.3, 1.7, 2.9] {
            let s: f64 = nonzero_basis(&t, 3, BSpline::new(t.to_vec(), vec![0.0; 6], 3).unwrap().interval(x).unwrap().0, x).iter().sum();
            assert!((s - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn interpolating_cubics_reproduce_cubics() {
        let x: Vec<f64> = (0..9).map(|i| (i as f64 * 0.7).powf(1.3)).collect();
        let f = |x: f64| 1.0 - 2.0 * x + 0.5 * x * x - 0.1 * x * x * x;
        let fp = |x: f64| -2.0 + x - 0.3 * x * x;
        let y: Vec<f64> = x.iter().map(|&v| f(v)).collect();
        let bcs = [
            BoundaryCondition::NotAKnot,
            BoundaryCondition::Derivatives { left: vec![(1, fp(x[0]))], right: vec![(1, fp(x[8]))] },
        ];
        for bc in bcs {
            let s = make_interp_spline(&x, &y, 3, &bc).unwrap();
            for z in [0.1, 1.3, 2.2, 4.0] {
                assert!((s.evaluate(z) - f(z)).abs() < 1e-12, "{bc:?} at {z}");
                assert!((s.evaluate_derivative(z, 1) - fp(z)).abs() < 1e-11);
            }
            let exact = |x: f64| x - x * x + x * x * x / 6.0 - 0.025 * x.powi(4);
            assert!((s.integrate(0.5, 3.0) - (exact(3.0) - exact(0.5))).abs() < 1e-12);
        }
    }

    #[test]
    fn periodic_spline_matches_derivatives_across_the_ends() {
        let n = 12;
        let x: Vec<f64> = (0..=n).map(|i| i as f64 * std::f64::consts::TAU / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin() + 0.3 * (2.0 * v).cos()).collect();
        let s = make_interp_spline(&x, &y, 3, &BoundaryCondition::Periodic).unwrap();
        let (a, b) = (x[0], x[n]);
        for nu in 0..3 {
            assert!((s.evaluate_derivative(a, nu) - s.evaluate_derivative(b - 1e-12, nu)).abs() < 1e-8, "order {nu}");
        }
        assert!((s.evaluate(a + 0.3) - s.evaluate(b + 0.3)).abs() < 1e-14);
    }
}
