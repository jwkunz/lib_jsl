//! Tensor-product interpolating splines on a rectangular grid (SciPy's `RectBivariateSpline`
//! with `s = 0`): `s(x, y) = Σᵢ Σⱼ cᵢⱼ Bᵢ,ₖₓ(x) Bⱼ,ₖᵧ(y)`, with values, partial derivatives and
//! integrals over rectangles.
//!
//! See the book entry [Piecewise polynomials, Hermite data and scattered interpolation](https://jwkunz.github.io/lib_jsl/book/interpolation/piecewise-scattered.html#tensor-product-splines).

use crate::one_d::b_spline::{BSpline, BoundaryCondition, make_interp_spline, nonzero_basis, nonzero_basis_derivatives};
use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

/// An interpolating spline of degrees `kx` and `ky` (odd: 1, 3 or 5) through `z[i][j] = f(xᵢ, yⱼ)`.
#[derive(Clone, Debug, PartialEq)]
pub struct RectBivariateSpline {
    tx: Vec<f64>,
    ty: Vec<f64>,
    kx: usize,
    ky: usize,
    /// Coefficients, shape `(nx, ny)`.
    c: Array2<f64>,
}

fn interval(t: &[f64], k: usize, n: usize, v: f64) -> usize {
    match t[k..=n].partition_point(|&a| a <= v) {
        0 => k,
        p => (k + p - 1).min(n - 1),
    }
}

impl RectBivariateSpline {
    // ANCHOR: rect_bivariate
    /// Fits the spline. With `s = 0` FITPACK places the interior knots at the data points except
    /// `(k + 1)/2` at each end, which for odd `k` is the not-a-knot spline; the separable system
    /// `Aₓ C Aᵧᵀ = Z` is solved one axis at a time: interpolate every column of `Z` along `x`,
    /// then every row of the result along `y`.
    pub fn new(x: &[f64], y: &[f64], z: &Array2<f64>, kx: usize, ky: usize) -> Result<RectBivariateSpline, ErrorsJSL> {
        if z.dim() != (x.len(), y.len()) {
            return Err(ErrorsJSL::InvalidInputRange("z must have shape (len(x), len(y))"));
        }
        for k in [kx, ky] {
            if k % 2 == 0 || k > 5 {
                return Err(ErrorsJSL::InvalidInputRange("degrees must be 1, 3 or 5 (FITPACK places even-degree knots differently)"));
            }
        }
        if x.len() <= kx || y.len() <= ky {
            return Err(ErrorsJSL::InvalidInputRange("need more than k points along each axis"));
        }
        let (nx, ny) = (x.len(), y.len());
        let mut along_x = Array2::zeros((nx, ny));
        let mut tx = Vec::new();
        for j in 0..ny {
            let col: Vec<f64> = z.column(j).to_vec();
            let s = make_interp_spline(x, &col, kx, &BoundaryCondition::NotAKnot)?;
            for i in 0..nx {
                along_x[(i, j)] = s.c[i];
            }
            tx = s.t;
        }
        let mut c = Array2::zeros((nx, ny));
        let mut ty = Vec::new();
        for i in 0..nx {
            let row: Vec<f64> = along_x.row(i).to_vec();
            let s = make_interp_spline(y, &row, ky, &BoundaryCondition::NotAKnot)?;
            for j in 0..ny {
                c[(i, j)] = s.c[j];
            }
            ty = s.t;
        }
        Ok(RectBivariateSpline { tx, ty, kx, ky, c })
    }
    // ANCHOR_END: rect_bivariate

    /// The value or partial derivative `∂^{dx+dy} s / ∂x^{dx} ∂y^{dy}` at `(x, y)`. Points outside
    /// the grid are clamped to its edge, as FITPACK's `bispev` does.
    pub fn evaluate(&self, x: f64, y: f64, dx: usize, dy: usize) -> f64 {
        let (nx, ny) = self.c.dim();
        let x = x.clamp(self.tx[self.kx], self.tx[nx]);
        let y = y.clamp(self.ty[self.ky], self.ty[ny]);
        let lx = interval(&self.tx, self.kx, nx, x);
        let ly = interval(&self.ty, self.ky, ny, y);
        let bx = if dx == 0 { nonzero_basis(&self.tx, self.kx, lx, x) } else { nonzero_basis_derivatives(&self.tx, self.kx, lx, x, dx) };
        let by = if dy == 0 { nonzero_basis(&self.ty, self.ky, ly, y) } else { nonzero_basis_derivatives(&self.ty, self.ky, ly, y, dy) };
        let mut sum = 0.0;
        for (a, wx) in bx.iter().enumerate() {
            let i = lx - self.kx + a;
            for (b, wy) in by.iter().enumerate() {
                sum += wx * wy * self.c[(i, ly - self.ky + b)];
            }
        }
        sum
    }

    /// `∫_{xa}^{xb} ∫_{ya}^{yb} s(x, y) dy dx` (SciPy's `integral`), as `Σ cᵢⱼ Iᵢ Jⱼ` with the
    /// integrals `Iᵢ`, `Jⱼ` of the one-dimensional basis functions.
    pub fn integral(&self, xa: f64, xb: f64, ya: f64, yb: f64) -> f64 {
        let (nx, ny) = self.c.dim();
        let basis_integrals = |t: &[f64], k: usize, n: usize, a: f64, b: f64| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let mut e = vec![0.0; n];
                    e[i] = 1.0;
                    BSpline::new(t.to_vec(), e, k).expect("valid knots").integrate(a, b)
                })
                .collect()
        };
        let ix = basis_integrals(&self.tx, self.kx, nx, xa, xb);
        let iy = basis_integrals(&self.ty, self.ky, ny, ya, yb);
        self.c.indexed_iter().map(|((i, j), c)| c * ix[i] * iy[j]).sum()
    }

    /// The knots along `x` and `y`.
    pub fn knots(&self) -> (&[f64], &[f64]) {
        (&self.tx, &self.ty)
    }

    /// The coefficients `cᵢⱼ`.
    pub fn coefficients(&self) -> &Array2<f64> {
        &self.c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_a_bicubic_polynomial() {
        let f = |x: f64, y: f64| x * x * y - 2.0 * x * y * y * y + 0.5 * x + 1.0;
        let x: Vec<f64> = (0..7).map(|i| i as f64 * 0.4).collect();
        let y: Vec<f64> = (0..6).map(|j| -1.0 + j as f64 * 0.5).collect();
        let z = Array2::from_shape_fn((7, 6), |(i, j)| f(x[i], y[j]));
        let s = RectBivariateSpline::new(&x, &y, &z, 3, 3).unwrap();
        for (a, b) in [(0.3, 0.2), (1.7, -0.9), (2.2, 1.4)] {
            assert!((s.evaluate(a, b, 0, 0) - f(a, b)).abs() < 1e-12);
            assert!((s.evaluate(a, b, 1, 0) - (2.0 * a * b - 2.0 * b.powi(3) + 0.5)).abs() < 1e-11);
            assert!((s.evaluate(a, b, 0, 1) - (a * a - 6.0 * a * b * b)).abs() < 1e-11);
        }
        // ∫₀^{2.4} ∫_{−1}^{1.5} f dy dx.
        let exact = {
            let fx = |x: f64| x * x * (1.5f64.powi(2) - 1.0) / 2.0 - 2.0 * x * (1.5f64.powi(4) - 1.0) / 4.0 + (0.5 * x + 1.0) * 2.5;
            // Integrate the cubic fx exactly by Simpson's rule (exact for cubics).
            let (a, b) = (0.0, 2.4);
            (b - a) / 6.0 * (fx(a) + 4.0 * fx(0.5 * (a + b)) + fx(b))
        };
        assert!((s.integral(0.0, 2.4, -1.0, 1.5) - exact).abs() < 1e-11);
    }
}
