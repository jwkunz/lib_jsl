//! Piecewise polynomials in the power basis, following SciPy's `PPoly`: evaluation, derivatives,
//! antiderivatives, definite integrals and real roots, with constructors from cubic Hermite data
//! (`CubicHermiteSpline`) and from B-splines (`PPoly.from_spline`).
//!
//! On interval `j`, `[x_j, x_{j+1})`, the polynomial is `Σ_i c[i][j] (x − x_j)^{k−i}`: row 0
//! holds the highest power, as in SciPy.
//!
//! See the book entry [Piecewise polynomials, Hermite data and scattered interpolation](https://jwkunz.github.io/lib_jsl/book/interpolation/piecewise-scattered.html).

use super::b_spline::BSpline;
use lib_jsl_core::ErrorsJSL;
use lib_jsl_linalg::EigVals;
use ndarray::Array2;

/// A piecewise polynomial of degree `k` on `m` intervals.
#[derive(Clone, Debug, PartialEq)]
pub struct PPoly {
    /// Coefficients, shape `(k + 1, m)`; row 0 multiplies the highest power.
    pub c: Array2<f64>,
    /// The `m + 1` breakpoints, increasing.
    pub x: Vec<f64>,
    /// Evaluate outside `[x₀, x_m]` by extending the end pieces (otherwise `NaN`).
    pub extrapolate: bool,
}

fn factorial_ratio(n: usize, nu: usize) -> f64 {
    // n (n − 1) … (n − nu + 1)
    (0..nu).map(|i| (n - i) as f64).product()
}

impl PPoly {
    /// A piecewise polynomial from coefficients (shape `(k + 1, m)`) and `m + 1` breakpoints.
    pub fn new(c: Array2<f64>, x: Vec<f64>, extrapolate: bool) -> Result<PPoly, ErrorsJSL> {
        if c.nrows() == 0 || c.ncols() + 1 != x.len() {
            return Err(ErrorsJSL::InvalidInputRange("need k + 1 ≥ 1 rows and one column per interval"));
        }
        if x.windows(2).any(|w| w[1] < w[0] || w[1].is_nan()) {
            return Err(ErrorsJSL::InvalidInputRange("breakpoints must be non-decreasing"));
        }
        Ok(PPoly { c, x, extrapolate })
    }

    /// The degree `k`.
    pub fn degree(&self) -> usize {
        self.c.nrows() - 1
    }

    /// The interval containing `v`: `j` with `x_j ≤ v < x_{j+1}`, the last interval at the right
    /// end, and the end intervals outside (when extrapolating).
    fn interval(&self, v: f64) -> Option<usize> {
        let m = self.c.ncols();
        if !self.extrapolate && (v < self.x[0] || v > self.x[m]) || v.is_nan() {
            return None;
        }
        Some(self.x.partition_point(|&b| b <= v).saturating_sub(1).min(m - 1))
    }

    /// The `nu`-th derivative of piece `j` at offset `s = v − x_j`, by Horner's rule.
    fn eval_piece(&self, j: usize, s: f64, nu: usize) -> f64 {
        let k = self.degree();
        if nu > k {
            return 0.0;
        }
        let mut acc = 0.0;
        for i in 0..=k - nu {
            acc = acc * s + self.c[(i, j)] * factorial_ratio(k - i, nu);
        }
        acc
    }

    // ANCHOR: evaluate
    /// The value at `v`.
    pub fn evaluate(&self, v: f64) -> f64 {
        self.evaluate_derivative(v, 0)
    }

    /// The `nu`-th derivative at `v`.
    pub fn evaluate_derivative(&self, v: f64, nu: usize) -> f64 {
        match self.interval(v) {
            Some(j) => self.eval_piece(j, v - self.x[j], nu),
            None => f64::NAN,
        }
    }
    // ANCHOR_END: evaluate

    /// The derivative of order `nu` as a piecewise polynomial of degree `k − nu` (degree 0 and
    /// value 0 if `nu > k`).
    pub fn derivative(&self, nu: usize) -> PPoly {
        let k = self.degree();
        let m = self.c.ncols();
        if nu > k {
            return PPoly { c: Array2::zeros((1, m)), x: self.x.clone(), extrapolate: self.extrapolate };
        }
        let c = Array2::from_shape_fn((k + 1 - nu, m), |(i, j)| self.c[(i, j)] * factorial_ratio(k - i, nu));
        PPoly { c, x: self.x.clone(), extrapolate: self.extrapolate }
    }

    // ANCHOR: antiderivative
    /// The antiderivative of order `nu`, zero at `x₀` and continuous with its first `nu − 1`
    /// derivatives (SciPy's `antiderivative`): the coefficients are divided by rising factorials,
    /// and the new low-order coefficients of each piece are set from the previous piece's values
    /// at the shared breakpoint.
    pub fn antiderivative(&self, nu: usize) -> PPoly {
        let k = self.degree();
        let m = self.c.ncols();
        let rows = k + 1 + nu;
        let mut c = Array2::zeros((rows, m));
        for i in 0..=k {
            let factor: f64 = (1..=nu).map(|r| (k - i + r) as f64).product();
            for j in 0..m {
                c[(i, j)] = self.c[(i, j)] / factor;
            }
        }
        let mut out = PPoly { c, x: self.x.clone(), extrapolate: self.extrapolate };
        for j in 1..m {
            let h = out.x[j] - out.x[j - 1];
            for dx in (0..nu).rev() {
                let value = out.eval_piece(j - 1, h, dx) / (1..=dx).map(|r| r as f64).product::<f64>();
                out.c[(rows - 1 - dx, j)] = value;
            }
        }
        out
    }
    // ANCHOR_END: antiderivative

    /// `∫_a^b p(x) dx` (with extrapolation beyond the breakpoints if enabled).
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        let anti = PPoly { extrapolate: true, ..self.antiderivative(1) };
        if !self.extrapolate {
            let (lo, hi) = (self.x[0], self.x[self.x.len() - 1]);
            let (ca, cb) = (a.clamp(lo, hi), b.clamp(lo, hi));
            return anti.evaluate(cb) - anti.evaluate(ca);
        }
        anti.evaluate(b) - anti.evaluate(a)
    }

    // ANCHOR: roots
    /// The real `x` with `p(x) = y` (SciPy's `solve`), increasing. Each piece's roots are the
    /// eigenvalues of the companion matrix of its polynomial, kept when real (to a small
    /// tolerance) and inside the piece, then polished by Newton steps. With `discontinuity`, a
    /// breakpoint where the polynomial jumps across `y` is reported too. Pieces that equal `y`
    /// identically are skipped.
    pub fn solve(&self, y: f64, discontinuity: bool) -> Vec<f64> {
        let k = self.degree();
        let m = self.c.ncols();
        let mut roots = Vec::new();
        for j in 0..m {
            let (lo, hi) = (self.x[j], self.x[j + 1]);
            if hi == lo {
                continue;
            }
            let h = hi - lo;
            let (left, right) = if self.extrapolate { (if j == 0 { f64::NEG_INFINITY } else { 0.0 }, if j == m - 1 { f64::INFINITY } else { h }) } else { (0.0, h) };
            let mut coef: Vec<f64> = (0..=k).map(|i| self.c[(i, j)]).collect();
            coef[k] -= y;
            // Drop leading zeros.
            let lead = coef.iter().position(|&v| v != 0.0);
            let Some(lead) = lead else { continue };
            let p = &coef[lead..];
            let deg = p.len() - 1;
            if deg == 0 {
                continue;
            }
            let candidates: Vec<f64> = if deg == 1 {
                vec![-p[1] / p[0]]
            } else {
                let comp = Array2::from_shape_fn((deg, deg), |(r, col)| if r == 0 { -p[col + 1] / p[0] } else if r == col + 1 { 1.0 } else { 0.0 });
                match comp.eigvals() {
                    Ok(ev) => {
                        let scale = ev.iter().fold(1.0f64, |a, z| a.max(z.norm()));
                        ev.iter().filter(|z| z.im.abs() <= 1e-7 * scale).map(|z| z.re).collect()
                    }
                    Err(_) => Vec::new(),
                }
            };
            let horner = |s: f64| p.iter().fold(0.0, |a, &c| a * s + c);
            let dhorner = |s: f64| p.iter().take(deg).enumerate().fold(0.0, |a, (i, &c)| a * s + c * (deg - i) as f64);
            for mut s in candidates {
                for _ in 0..3 {
                    let d = dhorner(s);
                    if d == 0.0 {
                        break;
                    }
                    let step = horner(s) / d;
                    s -= step;
                    if step.abs() <= 4.0 * f64::EPSILON * s.abs().max(h) {
                        break;
                    }
                }
                let tol = 1e-12 * h;
                if s >= left - tol && s < right + if j == m - 1 { tol } else { -tol.min(0.0) } {
                    let s = s.clamp(left.max(-f64::MAX), right.min(f64::MAX));
                    if s < h || j == m - 1 {
                        roots.push(lo + s);
                    }
                }
            }
            if discontinuity && j > 0 {
                let before = self.eval_piece(j - 1, lo - self.x[j - 1], 0) - y;
                let after = self.c[(k, j)] - y;
                if before * after < 0.0 {
                    roots.push(lo);
                }
            }
        }
        roots.sort_by(f64::total_cmp);
        roots.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (1.0 + b.abs()));
        roots
    }
    // ANCHOR_END: roots

    /// The real roots, `solve(0, true)` (SciPy's `roots`).
    pub fn roots(&self) -> Vec<f64> {
        self.solve(0.0, true)
    }

    // ANCHOR: cubic_hermite
    /// The cubic Hermite interpolant with values `y` and slopes `dydx` at the strictly increasing
    /// `x` (SciPy's `CubicHermiteSpline`). With `h` the interval length and `s` the secant slope,
    /// the piece is `y₀ + d₀ t + (3s − 2d₀ − d₁)/h · t² + (d₀ + d₁ − 2s)/h² · t³`.
    pub fn cubic_hermite(x: &[f64], y: &[f64], dydx: &[f64]) -> Result<PPoly, ErrorsJSL> {
        let n = x.len();
        if n < 2 || y.len() != n || dydx.len() != n {
            return Err(ErrorsJSL::InvalidInputRange("need at least two points and equal lengths"));
        }
        if x.windows(2).any(|w| w[1] <= w[0] || w[1].is_nan()) {
            return Err(ErrorsJSL::InvalidInputRange("x must be strictly increasing"));
        }
        let mut c = Array2::zeros((4, n - 1));
        for j in 0..n - 1 {
            let h = x[j + 1] - x[j];
            let s = (y[j + 1] - y[j]) / h;
            let (d0, d1) = (dydx[j], dydx[j + 1]);
            c[(0, j)] = (d0 + d1 - 2.0 * s) / (h * h);
            c[(1, j)] = (3.0 * s - 2.0 * d0 - d1) / h;
            c[(2, j)] = d0;
            c[(3, j)] = y[j];
        }
        Ok(PPoly { c, x: x.to_vec(), extrapolate: true })
    }
    // ANCHOR_END: cubic_hermite

    /// The same function as a B-spline, as pieces between its distinct knots in the base interval
    /// `[t_k, t_n]` (SciPy's `PPoly.from_spline`, which also keeps the empty intervals between
    /// repeated knots): each piece's coefficients are the spline's Taylor coefficients
    /// `s⁽ⁱ⁾(x_j)/i!` at the left end.
    pub fn from_spline(spline: &BSpline) -> PPoly {
        let k = spline.k;
        let n = spline.c.len();
        let mut x: Vec<f64> = spline.t[k..=n].to_vec();
        x.dedup();
        let m = x.len() - 1;
        let mut c = Array2::zeros((k + 1, m));
        let mut fact = 1.0;
        for d in 0..=k {
            if d > 0 {
                fact *= d as f64;
            }
            for j in 0..m {
                c[(k - d, j)] = spline.evaluate_derivative(x[j], d) / fact;
            }
        }
        PPoly { c, x, extrapolate: true }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::one_d::b_spline::{BoundaryCondition, make_interp_spline};

    #[test]
    fn hermite_reproduces_a_cubic() {
        let f = |v: f64| v * v * v - 2.0 * v + 1.0;
        let df = |v: f64| 3.0 * v * v - 2.0;
        let x = [-1.0, 0.0, 0.7, 2.0];
        let y = x.map(f);
        let d = x.map(df);
        let p = PPoly::cubic_hermite(&x, &y, &d).unwrap();
        for v in [-1.5, -0.3, 0.5, 1.9, 2.5] {
            assert!((p.evaluate(v) - f(v)).abs() < 1e-12);
            assert!((p.evaluate_derivative(v, 1) - df(v)).abs() < 1e-11);
            assert!((p.evaluate_derivative(v, 2) - 6.0 * v).abs() < 1e-10);
        }
        // ∫_{-1}^{2} f = [x⁴/4 − x² + x] = (4 − 4 + 2) − (1/4 − 1 − 1) = 3.75
        assert!((p.integrate(-1.0, 2.0) - 3.75).abs() < 1e-12);
        let a = p.antiderivative(2);
        assert!((a.evaluate_derivative(1.3, 2) - f(1.3)).abs() < 1e-11);
        // f has three real roots.
        let r = p.roots();
        assert_eq!(r.len(), 3, "{r:?}");
        for v in r {
            assert!(f(v).abs() < 1e-12);
        }
    }

    #[test]
    fn from_spline_matches_the_spline() {
        let x: Vec<f64> = (0..9).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|v| (v * 1.3).sin()).collect();
        let s = make_interp_spline(&x, &y, 3, &BoundaryCondition::NotAKnot).unwrap();
        let p = PPoly::from_spline(&s);
        for i in 0..40 {
            let v = -0.2 + i as f64 * 0.11;
            assert!((p.evaluate(v) - s.evaluate(v)).abs() < 1e-12);
            assert!((p.evaluate_derivative(v, 1) - s.evaluate_derivative(v, 1)).abs() < 1e-11);
        }
        assert!((p.integrate(0.3, 3.7) - s.integrate(0.3, 3.7)).abs() < 1e-12);
        let r = p.roots();
        // sin(1.3 x) = 0 at x = π/1.3 ≈ 2.4166 (and x = 0).
        assert!(r.iter().any(|v| (v - std::f64::consts::PI / 1.3).abs() < 1e-3), "{r:?}");
    }
}
