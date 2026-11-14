//! Radial basis function interpolation of scattered data in any number of dimensions (SciPy's
//! `RBFInterpolator`): `s(x) = Σᵢ aᵢ φ(ε‖x − yᵢ‖) + Σⱼ bⱼ pⱼ(x)`, with a polynomial tail of the
//! degree the kernel needs, optional smoothing, and SciPy's scaling of the polynomial terms.
//!
//! See the book entry [Piecewise polynomials, Hermite data and scattered interpolation](https://jwkunz.github.io/lib_jsl/book/interpolation/piecewise-scattered.html#radial-basis-functions-in-any-dimension).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_linalg::Solve;
use ndarray::{Array1, Array2};

/// The radial kernels `φ(r)` of SciPy's `RBFInterpolator`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbfKernel {
    /// `−r` (needs degree ≥ 0).
    Linear,
    /// `r² ln r` (needs degree ≥ 1; SciPy's default).
    ThinPlateSpline,
    /// `r³` (needs degree ≥ 1).
    Cubic,
    /// `−r⁵` (needs degree ≥ 2).
    Quintic,
    /// `−√(1 + r²)` (needs degree ≥ 0).
    Multiquadric,
    /// `1/√(1 + r²)`.
    InverseMultiquadric,
    /// `1/(1 + r²)`.
    InverseQuadratic,
    /// `exp(−r²)`.
    Gaussian,
}

impl RbfKernel {
    pub fn eval(self, r: f64) -> f64 {
        match self {
            RbfKernel::Linear => -r,
            RbfKernel::ThinPlateSpline => if r == 0.0 { 0.0 } else { r * r * r.ln() },
            RbfKernel::Cubic => r * r * r,
            RbfKernel::Quintic => -r.powi(5),
            RbfKernel::Multiquadric => -(r * r + 1.0).sqrt(),
            RbfKernel::InverseMultiquadric => 1.0 / (r * r + 1.0).sqrt(),
            RbfKernel::InverseQuadratic => 1.0 / (r * r + 1.0),
            RbfKernel::Gaussian => (-r * r).exp(),
        }
    }

    /// The lowest polynomial degree that makes the system solvable (`−1`: none needed), from
    /// the order of conditional positive definiteness (Fasshauer, chapter 8).
    pub fn min_degree(self) -> i32 {
        match self {
            RbfKernel::Linear | RbfKernel::Multiquadric => 0,
            RbfKernel::ThinPlateSpline | RbfKernel::Cubic => 1,
            RbfKernel::Quintic => 2,
            _ => -1,
        }
    }

    /// Whether the interpolant is independent of `ε` (so `ε = 1` by default).
    pub fn scale_invariant(self) -> bool {
        matches!(self, RbfKernel::Linear | RbfKernel::ThinPlateSpline | RbfKernel::Cubic | RbfKernel::Quintic)
    }
}

/// An RBF interpolant.
#[derive(Clone, Debug)]
pub struct RbfInterpolator {
    y: Vec<Vec<f64>>,
    kernel: RbfKernel,
    epsilon: f64,
    powers: Vec<Vec<u32>>,
    shift: Vec<f64>,
    scale: Vec<f64>,
    coeffs: Array1<f64>,
}

/// The exponents of all monomials of total degree ≤ `degree` in `ndim` variables, in SciPy's
/// order (by degree, then by `combinations_with_replacement`).
pub fn monomial_powers(ndim: usize, degree: i32) -> Vec<Vec<u32>> {
    // Non-decreasing variable sequences of length `left`, starting at variable `start`.
    fn extend(ndim: usize, start: usize, left: usize, current: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
        if left == 0 {
            out.push(current.clone());
            return;
        }
        for v in start..ndim {
            current[v] += 1;
            extend(ndim, v, left - 1, current, out);
            current[v] -= 1;
        }
    }
    let mut out = Vec::new();
    for deg in 0..=degree {
        extend(ndim, 0, deg as usize, &mut vec![0; ndim], &mut out);
    }
    out
}

impl RbfInterpolator {
    // ANCHOR: rbf
    /// Fits the interpolant to values `d` at points `y` (rows). `epsilon` defaults to 1 for the
    /// scale-invariant kernels and is required otherwise; `degree` defaults to the kernel's
    /// minimum (at least 0); `smoothing` (≥ 0) relaxes interpolation to a least-squares fit. The
    /// system `[K + λI, P; Pᵀ, 0] [a; b] = [d; 0]` is solved by LU, with the polynomial built on
    /// coordinates shifted and scaled into `[−1, 1]`.
    pub fn new(y: &[Vec<f64>], d: &[f64], kernel: RbfKernel, epsilon: Option<f64>, degree: Option<i32>, smoothing: f64) -> Result<RbfInterpolator, ErrorsJSL> {
        let p = y.len();
        if p == 0 || d.len() != p {
            return Err(ErrorsJSL::InvalidInputRange("need one value per point"));
        }
        let ndim = y[0].len();
        if y.iter().any(|r| r.len() != ndim) {
            return Err(ErrorsJSL::InvalidInputRange("all points must have the same dimension"));
        }
        let epsilon = match epsilon {
            Some(e) => e,
            None if kernel.scale_invariant() => 1.0,
            None => return Err(ErrorsJSL::InvalidInputRange("this kernel needs a shape parameter epsilon")),
        };
        let degree = degree.unwrap_or(kernel.min_degree().max(0));
        let powers = monomial_powers(ndim, degree);
        let r = powers.len();
        let mut shift = vec![0.0; ndim];
        let mut scale = vec![1.0; ndim];
        for k in 0..ndim {
            let lo = y.iter().map(|v| v[k]).fold(f64::INFINITY, f64::min);
            let hi = y.iter().map(|v| v[k]).fold(f64::NEG_INFINITY, f64::max);
            shift[k] = (hi + lo) / 2.0;
            scale[k] = if hi == lo { 1.0 } else { (hi - lo) / 2.0 };
        }
        let mut lhs = Array2::zeros((p + r, p + r));
        for i in 0..p {
            for j in 0..p {
                let dist = y[i].iter().zip(&y[j]).map(|(a, b)| (a * epsilon - b * epsilon).powi(2)).sum::<f64>().sqrt();
                lhs[(i, j)] = kernel.eval(dist);
            }
            lhs[(i, i)] += smoothing;
            for (m, pw) in powers.iter().enumerate() {
                let v: f64 = (0..ndim).map(|k| ((y[i][k] - shift[k]) / scale[k]).powi(pw[k] as i32)).product();
                lhs[(i, p + m)] = v;
                lhs[(p + m, i)] = v;
            }
        }
        let mut rhs = Array1::zeros(p + r);
        for i in 0..p {
            rhs[i] = d[i];
        }
        let coeffs = lhs.solve(&rhs).map_err(|_| ErrorsJSL::RuntimeError("singular RBF system: too few points for the polynomial degree?"))?;
        Ok(RbfInterpolator { y: y.to_vec(), kernel, epsilon, powers, shift, scale, coeffs })
    }
    // ANCHOR_END: rbf

    /// The interpolant at `x`.
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        let p = self.y.len();
        let mut s = 0.0;
        for (i, yi) in self.y.iter().enumerate() {
            let dist = yi.iter().zip(x).map(|(a, b)| (b * self.epsilon - a * self.epsilon).powi(2)).sum::<f64>().sqrt();
            s += self.coeffs[i] * self.kernel.eval(dist);
        }
        for (m, pw) in self.powers.iter().enumerate() {
            let v: f64 = (0..x.len()).map(|k| ((x[k] - self.shift[k]) / self.scale[k]).powi(pw[k] as i32)).product();
            s += self.coeffs[p + m] * v;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monomials_in_scipy_order() {
        assert_eq!(monomial_powers(2, 2), vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1], vec![0, 2]]);
        assert_eq!(monomial_powers(3, 1).len(), 4);
        assert_eq!(monomial_powers(3, 3).len(), 20);
        assert!(monomial_powers(2, -1).is_empty());
    }

    #[test]
    fn interpolates_in_three_dimensions() {
        let pts: Vec<Vec<f64>> = (0..60).map(|i| {
            let t = i as f64;
            vec![(t * 0.37).sin(), (t * 0.71).cos(), (t * 0.13).sin() * 2.0]
        }).collect();
        let f = |v: &[f64]| v[0] * v[1] + v[2].sin();
        let d: Vec<f64> = pts.iter().map(|v| f(v)).collect();
        for kernel in [RbfKernel::ThinPlateSpline, RbfKernel::Cubic, RbfKernel::Quintic, RbfKernel::Linear] {
            let r = RbfInterpolator::new(&pts, &d, kernel, None, None, 0.0).unwrap();
            for (p, v) in pts.iter().zip(&d) {
                assert!((r.evaluate(p) - v).abs() < 1e-8, "{kernel:?}");
            }
        }
        let g = RbfInterpolator::new(&pts, &d, RbfKernel::Gaussian, Some(1.5), None, 0.0).unwrap();
        assert!((g.evaluate(&pts[3]) - d[3]).abs() < 1e-8);
        // Linear data are reproduced exactly by a kernel with a linear tail.
        let lin: Vec<f64> = pts.iter().map(|v| 1.0 + 2.0 * v[0] - v[2]).collect();
        let r = RbfInterpolator::new(&pts, &lin, RbfKernel::ThinPlateSpline, None, None, 0.0).unwrap();
        assert!((r.evaluate(&[0.1, 0.2, 0.3]) - (1.0 + 0.2 - 0.3)).abs() < 1e-10);
    }
}
