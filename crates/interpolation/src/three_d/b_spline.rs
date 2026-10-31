//! Interpolating tricubic B-spline on a uniform 3D grid.

use super::{Interpolation3D, validate_grid_3d};
use crate::two_d::uniform_axis;
use lib_jsl_core::*;
use ndarray::{Array3, Axis};

/// Cubic B-spline `s(u) = Σ c_k B₃(u - k)` through the samples, with natural end conditions
/// (zero second derivative, equivalent to the natural cubic spline along each axis). Coefficients
/// are found by solving a tridiagonal system along each axis; the surface is C² and each evaluation
/// touches a 4×4×4 coefficient block.
#[derive(Debug, Clone)]
pub struct BSpline3D {
    origin: [f64; 3],
    step: [f64; 3],
    /// Coefficients padded by one extra layer on each side (extrapolated linearly), shape (nx+2, ny+2, nz+2).
    coeffs: Array3<f64>,
}

/// Cubic B-spline coefficients with natural ends for samples `f`; returns `n + 2` values whose
/// index 0 and `n + 1` are the extrapolated `c_{-1}` and `c_n`.
fn coefficients_1d(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let mut c = vec![0.0; n + 2];
    c[1] = f[0];
    c[n] = f[n - 1];
    if n > 2 {
        // (c[i-1] + 4 c[i] + c[i+1]) / 6 = f[i] for interior i, c[0-node] and c[n-1 node] fixed.
        let m = n - 2;
        let (mut diag, mut rhs) = (vec![4.0; m], vec![0.0; m]);
        for i in 0..m {
            rhs[i] = 6.0 * f[i + 1];
        }
        rhs[0] -= f[0];
        rhs[m - 1] -= f[n - 1];
        for i in 1..m {
            let w = 1.0 / diag[i - 1];
            diag[i] -= w;
            rhs[i] -= w * rhs[i - 1];
        }
        let mut x = vec![0.0; m];
        x[m - 1] = rhs[m - 1] / diag[m - 1];
        for i in (0..m - 1).rev() {
            x[i] = (rhs[i] - x[i + 1]) / diag[i];
        }
        c[2..n].copy_from_slice(&x);
    }
    c[0] = 2.0 * c[1] - c[2.min(n)];
    c[n + 1] = 2.0 * c[n] - c[(n - 1).max(1)];
    c
}

fn basis(t: f64) -> [f64; 4] {
    // Weights of coefficients c[i-1..=i+2] for the position i + t, t in [0, 1).
    let (t2, t3) = (t * t, t * t * t);
    [(1.0 - t).powi(3) / 6.0, (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0, (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, t3 / 6.0]
}

impl BSpline3D {
    /// All three axes must be uniformly spaced with at least 3 points.
    pub fn new(x: VR1D, y: VR1D, z: VR1D, v: &Array3<f64>) -> Result<Self, ErrorsJSL> {
        validate_grid_3d([x, y, z], v, 3)?;
        let (ox, sx) = uniform_axis(x)?;
        let (oy, sy) = uniform_axis(y)?;
        let (oz, sz) = uniform_axis(z)?;
        let (nx, ny, nz) = v.dim();
        let mut c = v.clone();
        // Prefilter along each axis, growing it by the two padding layers.
        for (axis, n) in [(0usize, nx), (1, ny), (2, nz)] {
            let mut shape = c.raw_dim();
            shape[axis] = n + 2;
            let mut out = Array3::<f64>::zeros(shape);
            for (lane_in, mut lane_out) in c.lanes(Axis(axis)).into_iter().zip(out.lanes_mut(Axis(axis))) {
                let coef = coefficients_1d(&lane_in.to_vec());
                lane_out.iter_mut().zip(coef).for_each(|(o, k)| *o = k);
            }
            c = out;
        }
        Ok(BSpline3D { origin: [ox, oy, oz], step: [sx, sy, sz], coeffs: c })
    }
}

impl Interpolation3D for BSpline3D {
    fn interpolate_at(&self, x: f64, y: f64, z: f64) -> Result<f64, ErrorsJSL> {
        let (nx, ny, nz) = self.coeffs.dim();
        let n = [nx - 2, ny - 2, nz - 2];
        let p = [x, y, z];
        let mut base = [0usize; 3];
        let mut w = [[0.0; 4]; 3];
        for d in 0..3 {
            let u = (p[d] - self.origin[d]) / self.step[d];
            let cell = (u.floor().max(0.0) as usize).min(n[d] - 2);
            base[d] = cell; // padded index of c_{cell-1} is `cell`
            w[d] = basis(u - cell as f64);
        }
        let mut sum = 0.0;
        for (a, wa) in w[0].iter().enumerate() {
            for (b, wb) in w[1].iter().enumerate() {
                for (c, wc) in w[2].iter().enumerate() {
                    sum += wa * wb * wc * self.coeffs[[base[0] + a, base[1] + b, base[2] + c]];
                }
            }
        }
        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_nodes_and_reproduces_linear_functions() {
        let ax = R1D::from_iter((0..7).map(|k| 0.5 * k as f64));
        let ay = R1D::from_iter((0..6).map(|k| -1.0 + 0.4 * k as f64));
        let az = R1D::from_iter((0..5).map(|k| 2.0 + 0.25 * k as f64));
        let f = |a: f64, b: f64, c: f64| 1.0 + 2.0 * a - b + 0.5 * c;
        let v = Array3::from_shape_fn((7, 6, 5), |(i, j, k)| f(ax[i], ay[j], az[k]));
        let s = BSpline3D::new(ax.view(), ay.view(), az.view(), &v).unwrap();
        for &(a, b, c) in &[(0.0, -1.0, 2.0), (1.3, 0.2, 2.6), (3.0, 1.0, 3.0), (3.4, -0.7, 2.1)] {
            assert!((s.interpolate_at(a, b, c).unwrap() - f(a, b, c)).abs() < 1e-11, "{a},{b},{c}");
        }
        let g = Array3::from_shape_fn((7, 6, 5), |(i, j, k)| ((i * 3 + j * 5 + k * 7) % 4) as f64);
        let t = BSpline3D::new(ax.view(), ay.view(), az.view(), &g).unwrap();
        for i in 0..7 {
            for j in 0..6 {
                for k in 0..5 {
                    assert!((t.interpolate_at(ax[i], ay[j], az[k]).unwrap() - g[[i, j, k]]).abs() < 1e-11);
                }
            }
        }
    }

    #[test]
    fn smooth_function_accuracy_and_requirements() {
        let n = 21;
        let ax = R1D::from_iter((0..n).map(|k| k as f64 * 0.1));
        let f = |a: f64, b: f64, c: f64| (a + 0.2).sin() * (b * 1.1).cos() + c * c * 0.3;
        let v = Array3::from_shape_fn((n, n, n), |(i, j, k)| f(ax[i], ax[j], ax[k]));
        let s = BSpline3D::new(ax.view(), ax.view(), ax.view(), &v).unwrap();
        assert!((s.interpolate_at(0.83, 1.07, 0.55).unwrap() - f(0.83, 1.07, 0.55)).abs() < 2e-4);
        let bad = R1D::from_vec(vec![0.0, 1.0, 3.0]);
        assert!(BSpline3D::new(bad.view(), bad.view(), bad.view(), &Array3::zeros((3, 3, 3))).is_err());
    }
}
