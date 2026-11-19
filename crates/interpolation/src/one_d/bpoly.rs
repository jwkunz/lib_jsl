//! Piecewise polynomials in the Bernstein basis, following SciPy's `BPoly`: evaluation by de
//! Casteljau's algorithm, derivatives, antiderivatives and integrals, conversion to and from the
//! power basis, and construction from derivatives at the breakpoints (`BPoly.from_derivatives`).
//!
//! On interval `j`, `[x_j, x_{j+1})`, with `s = (x − x_j)/(x_{j+1} − x_j)`, the polynomial is
//! `Σ_a c[a][j] b_{a,k}(s)` with the Bernstein polynomials `b_{a,k}(s) = C(k, a) sᵃ (1 − s)^{k−a}`.
//! The coefficients are the control points of a Bézier curve: the polynomial passes through
//! the first and last and stays in their convex hull.
//!
//! See the book entry [Smoothing splines with automatic knots](https://jwkunz.github.io/lib_jsl/book/interpolation/fitpack.html#the-bernstein-basis).

// Guards such as `!(b > a)` also reject NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::ppoly::PPoly;
use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

fn comb(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// A piecewise polynomial of degree `k` on `m` intervals in the Bernstein basis.
#[derive(Clone, Debug, PartialEq)]
pub struct BPoly {
    /// Coefficients, shape `(k + 1, m)`: `c[a][j]` multiplies `b_{a,k}` on interval `j`.
    pub c: Array2<f64>,
    /// The `m + 1` breakpoints, increasing.
    pub x: Vec<f64>,
    /// Evaluate outside `[x₀, x_m]` by extending the end pieces (otherwise `NaN`).
    pub extrapolate: bool,
}

impl BPoly {
    /// A piecewise polynomial from Bernstein coefficients (shape `(k + 1, m)`) and `m + 1`
    /// breakpoints.
    pub fn new(c: Array2<f64>, x: Vec<f64>, extrapolate: bool) -> Result<BPoly, ErrorsJSL> {
        if c.nrows() == 0 || c.ncols() + 1 != x.len() {
            return Err(ErrorsJSL::InvalidInputRange("need k + 1 ≥ 1 rows and one column per interval"));
        }
        if x.windows(2).any(|w| !(w[1] > w[0])) {
            return Err(ErrorsJSL::InvalidInputRange("breakpoints must be increasing"));
        }
        Ok(BPoly { c, x, extrapolate })
    }

    /// The degree `k`.
    pub fn degree(&self) -> usize {
        self.c.nrows() - 1
    }

    fn interval(&self, v: f64) -> Option<usize> {
        let m = self.c.ncols();
        if !self.extrapolate && (v < self.x[0] || v > self.x[m]) || v.is_nan() {
            return None;
        }
        Some(self.x.partition_point(|&b| b <= v).saturating_sub(1).min(m - 1))
    }

    // ANCHOR: evaluate
    /// The `nu`-th derivative at `v`: the coefficients are differenced `nu` times
    /// (`c'_a = k (c_{a+1} − c_a)/h`), then de Casteljau's algorithm evaluates the result by
    /// repeated convex combinations `c_a ← (1 − s) c_a + s c_{a+1}`.
    pub fn evaluate_derivative(&self, v: f64, nu: usize) -> f64 {
        let Some(j) = self.interval(v) else { return f64::NAN };
        let k = self.degree();
        if nu > k {
            return 0.0;
        }
        let h = self.x[j + 1] - self.x[j];
        let mut c: Vec<f64> = self.c.column(j).to_vec();
        for d in 0..nu {
            let deg = (k - d) as f64;
            c = c.windows(2).map(|w| deg * (w[1] - w[0]) / h).collect();
        }
        let s = (v - self.x[j]) / h;
        for r in (1..c.len()).rev() {
            for a in 0..r {
                c[a] = (1.0 - s) * c[a] + s * c[a + 1];
            }
        }
        c[0]
    }

    /// The value at `v`.
    pub fn evaluate(&self, v: f64) -> f64 {
        self.evaluate_derivative(v, 0)
    }
    // ANCHOR_END: evaluate

    /// The derivative of order `nu`, of degree `k − nu` (degree 0 and value 0 if `nu > k`).
    pub fn derivative(&self, nu: usize) -> BPoly {
        let (k, m) = (self.degree(), self.c.ncols());
        if nu > k {
            return BPoly { c: Array2::zeros((1, m)), x: self.x.clone(), extrapolate: self.extrapolate };
        }
        let mut c = self.c.clone();
        for d in 0..nu {
            let deg = (k - d) as f64;
            c = Array2::from_shape_fn((c.nrows() - 1, m), |(a, j)| deg * (c[(a + 1, j)] - c[(a, j)]) / (self.x[j + 1] - self.x[j]));
        }
        BPoly { c, x: self.x.clone(), extrapolate: self.extrapolate }
    }

    // ANCHOR: antiderivative
    /// The antiderivative of order `nu`, zero at `x₀` and continuous (SciPy's `antiderivative`).
    /// Once: `C_{a+1} = h (c₀ + … + c_a)/(k + 1)`, `C₀ = 0` on each interval; since only the last
    /// Bernstein polynomial is non-zero at the right end, adding the running sum of the last
    /// coefficients of the earlier intervals makes the pieces join up.
    pub fn antiderivative(&self, nu: usize) -> BPoly {
        let mut out = self.clone();
        for _ in 0..nu {
            let (k1, m) = (out.c.nrows(), out.c.ncols());
            let mut c = Array2::zeros((k1 + 1, m));
            for j in 0..m {
                let h = out.x[j + 1] - out.x[j];
                let mut acc = 0.0;
                for a in 0..k1 {
                    acc += out.c[(a, j)];
                    c[(a + 1, j)] = acc / k1 as f64 * h;
                }
            }
            // Each interval's own increase is its last coefficient (before any shift).
            let last: Vec<f64> = (0..m).map(|j| c[(k1, j)]).collect();
            let mut carry = 0.0;
            for j in 1..m {
                carry += last[j - 1];
                for a in 0..=k1 {
                    c[(a, j)] += carry;
                }
            }
            out = BPoly { c, x: out.x, extrapolate: out.extrapolate };
        }
        out
    }
    // ANCHOR_END: antiderivative

    /// `∫_a^b p(x) dx` (clamped to the breakpoints unless extrapolating).
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        let anti = BPoly { extrapolate: true, ..self.antiderivative(1) };
        let (lo, hi) = (self.x[0], self.x[self.x.len() - 1]);
        let (a, b) = if self.extrapolate { (a, b) } else { (a.clamp(lo, hi), b.clamp(lo, hi)) };
        anti.evaluate(b) - anti.evaluate(a)
    }

    /// The same polynomials in the Bernstein basis of degree `k + d` (degree elevation).
    pub fn raise_degree(&self, d: usize) -> BPoly {
        BPoly { c: raise_degree(&self.c, d), x: self.x.clone(), extrapolate: self.extrapolate }
    }

    /// From the power basis (SciPy's `BPoly.from_power_basis`):
    /// `c_a = Σ_{i ≤ a} C(a, i)/C(k, i) hⁱ pᵢ` with `pᵢ` the coefficient of `(x − x_j)ⁱ`.
    pub fn from_power_basis(pp: &PPoly) -> BPoly {
        let (k1, m) = (pp.c.nrows(), pp.c.ncols());
        let k = k1 - 1;
        let mut c = Array2::zeros((k1, m));
        for j in 0..m {
            let h = pp.x[j + 1] - pp.x[j];
            for a in 0..k1 {
                let factor = pp.c[(a, j)] / comb(k, k - a) * h.powi((k - a) as i32);
                for q in k - a..k1 {
                    c[(q, j)] += factor * comb(q, k - a);
                }
            }
        }
        BPoly { c, x: pp.x.clone(), extrapolate: pp.extrapolate }
    }

    /// To the power basis (SciPy's `PPoly.from_bernstein_basis`), by expanding
    /// `sᵃ (1 − s)^{k−a}` binomially.
    pub fn to_power_basis(&self) -> PPoly {
        let (k1, m) = (self.c.nrows(), self.c.ncols());
        let k = k1 - 1;
        let mut c = Array2::zeros((k1, m));
        for j in 0..m {
            let h = self.x[j + 1] - self.x[j];
            for a in 0..k1 {
                let factor = if a.is_multiple_of(2) { 1.0 } else { -1.0 } * comb(k, a) * self.c[(a, j)];
                for s in a..k1 {
                    let val = comb(k - a, s - a) * if s.is_multiple_of(2) { 1.0 } else { -1.0 };
                    c[(k - s, j)] += factor * val / h.powi(s as i32);
                }
            }
        }
        PPoly { c, x: self.x.clone(), extrapolate: self.extrapolate }
    }

    // ANCHOR: from_derivatives
    /// The piecewise polynomial matching given derivatives at the breakpoints (SciPy's
    /// `BPoly.from_derivatives`): `yi[i]` lists `f(xᵢ), f'(xᵢ), …`. On each interval the
    /// Bernstein coefficients follow from the end values one at a time, since the `q`-th
    /// derivative at the left end involves only the first `q + 1` coefficients (and at the right
    /// end only the last). With `orders`, interval `i` matches only enough derivatives for a
    /// polynomial of that degree, split between the ends. All pieces are raised to the degree of
    /// the interval with the most derivatives given, as in SciPy.
    pub fn from_derivatives(xi: &[f64], yi: &[Vec<f64>], orders: Option<&[usize]>) -> Result<BPoly, ErrorsJSL> {
        let m = xi.len().saturating_sub(1);
        if m == 0 || yi.len() != xi.len() || yi.iter().any(|y| y.is_empty()) {
            return Err(ErrorsJSL::InvalidInputRange("from_derivatives: need two or more breakpoints, each with at least a value"));
        }
        if xi.windows(2).any(|w| !(w[1] > w[0])) {
            return Err(ErrorsJSL::InvalidInputRange("from_derivatives: breakpoints must be increasing"));
        }
        if let Some(o) = orders
            && o.len() != 1
            && o.len() != m
        {
            return Err(ErrorsJSL::IncompatibleArraySizes((o.len(), m)));
        }
        if orders.is_some_and(|o| o.contains(&0)) {
            return Err(ErrorsJSL::InvalidInputRange("from_derivatives: orders must be positive"));
        }
        let mut pieces = Vec::with_capacity(m);
        for i in 0..m {
            let (la, lb) = (yi[i].len(), yi[i + 1].len());
            let (n1, n2) = match orders {
                None => (la, lb),
                Some(o) => {
                    let n = o[if o.len() == 1 { 0 } else { i }] + 1;
                    let n1 = (n / 2).min(la);
                    let n2 = (n - n1).min(lb);
                    let n1 = (n - n2).min(la);
                    if n1 + n2 != n {
                        return Err(ErrorsJSL::InvalidInputRange("from_derivatives: not enough derivatives for the requested order"));
                    }
                    (n1, n2)
                }
            };
            pieces.push(construct_from_derivatives(xi[i], xi[i + 1], &yi[i][..n1], &yi[i + 1][..n2]));
        }
        // SciPy's common size: the most derivatives given at the ends of any interval, or the
        // largest order if that is more.
        let given = (0..m).map(|i| yi[i].len() + yi[i + 1].len()).max().unwrap_or(1);
        let k1 = given.max(orders.and_then(|o| o.iter().copied().max()).unwrap_or(0));
        let mut c = Array2::zeros((k1, m));
        for (j, p) in pieces.iter().enumerate() {
            let col = Array2::from_shape_vec((p.len(), 1), p.clone()).expect("shape");
            let raised = raise_degree(&col, k1 - p.len());
            for a in 0..k1 {
                c[(a, j)] = raised[(a, 0)];
            }
        }
        Ok(BPoly { c, x: xi.to_vec(), extrapolate: true })
    }
    // ANCHOR_END: from_derivatives
}

/// The Bernstein coefficients of the polynomial of degree `na + nb − 1` on `[xa, xb]` with the
/// derivatives `ya` at `xa` and `yb` at `xb` (SciPy's `_construct_from_derivatives`).
fn construct_from_derivatives(xa: f64, xb: f64, ya: &[f64], yb: &[f64]) -> Vec<f64> {
    let (na, nb) = (ya.len(), yb.len());
    let n = na + nb;
    let h = xb - xa;
    let poch = |a: usize, q: usize| (0..q).map(|r| (a + r) as f64).product::<f64>();
    let sign = |p: usize| if p.is_multiple_of(2) { 1.0 } else { -1.0 };
    let mut c = vec![0.0; n];
    for q in 0..na {
        c[q] = ya[q] / poch(n - q, q) * h.powi(q as i32);
        for j in 0..q {
            c[q] -= sign(j + q) * comb(q, j) * c[j];
        }
    }
    for q in 0..nb {
        let idx = n - q - 1;
        c[idx] = yb[q] / poch(n - q, q) * sign(q) * h.powi(q as i32);
        for j in 0..q {
            c[idx] -= sign(j + 1) * comb(q, j + 1) * c[n - q + j];
        }
    }
    c
}

fn raise_degree(c: &Array2<f64>, d: usize) -> Array2<f64> {
    if d == 0 {
        return c.clone();
    }
    let (k1, m) = (c.nrows(), c.ncols());
    let k = k1 - 1;
    let mut out = Array2::zeros((k1 + d, m));
    for a in 0..k1 {
        for j in 0..m {
            let f = c[(a, j)] * comb(k, a);
            for q in 0..=d {
                out[(a + q, j)] += f * comb(d, q) / comb(k + d, a + q);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermite_data_round_trip_and_calculus() {
        // f = x³ − 2x on [0, 1, 3] from values and first derivatives.
        let f = |x: f64| x * x * x - 2.0 * x;
        let fp = |x: f64| 3.0 * x * x - 2.0;
        let xi = [0.0, 1.0, 3.0];
        let yi: Vec<Vec<f64>> = xi.iter().map(|&x| vec![f(x), fp(x)]).collect();
        let bp = BPoly::from_derivatives(&xi, &yi, None).unwrap();
        assert_eq!(bp.degree(), 3);
        for v in [0.0, 0.4, 1.0, 2.2, 3.0] {
            assert!((bp.evaluate(v) - f(v)).abs() < 1e-13, "{v}");
            assert!((bp.evaluate_derivative(v, 1) - fp(v)).abs() < 1e-12);
            assert!((bp.evaluate_derivative(v, 2) - 6.0 * v).abs() < 1e-11);
        }
        let exact = |x: f64| x.powi(4) / 4.0 - x * x;
        assert!((bp.integrate(0.3, 2.7) - (exact(2.7) - exact(0.3))).abs() < 1e-12);
        let anti = bp.antiderivative(2);
        assert!((anti.evaluate(2.0) - (2f64.powi(5) / 20.0 - 2f64.powi(3) / 3.0)).abs() < 1e-12);
        // Power basis and back.
        let pp = bp.to_power_basis();
        let back = BPoly::from_power_basis(&pp);
        assert!(back.c.iter().zip(bp.c.iter()).all(|(a, b)| (a - b).abs() < 1e-12));
        assert!((pp.evaluate(2.2) - f(2.2)).abs() < 1e-12);
        // Antiderivatives stay continuous across many intervals.
        let xi4 = [0.0, 0.5, 1.0, 2.0, 3.0];
        let yi4: Vec<Vec<f64>> = xi4.iter().map(|&x| vec![f(x), fp(x)]).collect();
        let b4 = BPoly::from_derivatives(&xi4, &yi4, None).unwrap();
        assert!((b4.integrate(0.0, 3.0) - exact(3.0)).abs() < 1e-12);
        assert!((b4.antiderivative(2).evaluate(2.5) - (2.5f64.powi(5) / 20.0 - 2.5f64.powi(3) / 3.0)).abs() < 1e-12);
        // Degree elevation leaves the values unchanged.
        let r = bp.raise_degree(2);
        assert!((r.evaluate(1.7) - f(1.7)).abs() < 1e-13);
    }
}
