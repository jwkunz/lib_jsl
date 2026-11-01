//! Radial basis function interpolation of scattered 2D data.
//!
//! See the book entry [Scattered data interpolation](https://jwkunz.github.io/lib_jsl/book/interpolation/scattered.html).

use super::Interpolation2D;
use lib_jsl_core::*;
use ndarray::{Array1, Array2};
use lib_jsl_linalg::Solve;

/// Radial kernel `φ(r)`; `epsilon` is the shape parameter where one is used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RadialKernel {
    /// `exp(-(εr)²)`
    Gaussian(f64),
    /// `sqrt(1 + (εr)²)`
    Multiquadric(f64),
    /// `1 / sqrt(1 + (εr)²)`
    InverseMultiquadric(f64),
    /// `r³`
    Cubic,
    /// `r² ln r`
    ThinPlateSpline,
}

impl RadialKernel {
    fn value(self, r: f64) -> f64 {
        match self {
            RadialKernel::Gaussian(e) => (-(e * r).powi(2)).exp(),
            RadialKernel::Multiquadric(e) => (1.0 + (e * r).powi(2)).sqrt(),
            RadialKernel::InverseMultiquadric(e) => 1.0 / (1.0 + (e * r).powi(2)).sqrt(),
            RadialKernel::Cubic => r * r * r,
            RadialKernel::ThinPlateSpline => if r == 0.0 { 0.0 } else { r * r * r.ln() },
        }
    }
}

/// `s(p) = Σ w_i φ(|p - p_i|) + a + b x + c y`, with the weights and the affine term found from
/// the interpolation conditions `s(p_i) = f_i` and the side conditions `Σ w_i {1, x_i, y_i} = 0`.
/// The affine term makes the cubic and thin-plate kernels well posed (conditionally positive
/// definite) and lets the interpolant reproduce planes exactly with any kernel. The system is
/// dense, so cost is O(n³) to build and O(n) per evaluation.
#[derive(Debug, Clone)]
pub struct RadialBasisInterpolator {
    kernel: RadialKernel,
    centers: Vec<[f64; 2]>,
    weights: Vec<f64>,
    affine: [f64; 3],
}

impl RadialBasisInterpolator {
    /// Needs at least three non-collinear points with distinct locations.
    pub fn new(x: VR1D, y: VR1D, f: VR1D, kernel: RadialKernel) -> Result<Self, ErrorsJSL> {
        let n = x.len();
        if y.len() != n || f.len() != n || n < 3 {
            return Err(ErrorsJSL::IncompatibleArraySizes((n, f.len())));
        }
        let centers: Vec<[f64; 2]> = (0..n).map(|i| [x[i], y[i]]).collect();
        // ANCHOR: rbf_system
        let m = n + 3;
        let mut a = Array2::<f64>::zeros((m, m));
        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = kernel.value(((centers[i][0] - centers[j][0]).powi(2) + (centers[i][1] - centers[j][1]).powi(2)).sqrt());
            }
            let row = [1.0, centers[i][0], centers[i][1]];
            for (q, v) in row.into_iter().enumerate() {
                a[[i, n + q]] = v;
                a[[n + q, i]] = v;
            }
        }
        let mut rhs = Array1::<f64>::zeros(m);
        rhs.slice_mut(ndarray::s![..n]).assign(&f);
        let sol = a.solve(&rhs).map_err(|_| ErrorsJSL::RuntimeError("singular RBF system (duplicate or collinear points?)"))?;
        // ANCHOR_END: rbf_system
        Ok(RadialBasisInterpolator {
            kernel,
            centers,
            weights: sol.slice(ndarray::s![..n]).to_vec(),
            affine: [sol[n], sol[n + 1], sol[n + 2]],
        })
    }
}

impl Interpolation2D for RadialBasisInterpolator {
    fn interpolate_at(&self, x: f64, y: f64) -> Result<f64, ErrorsJSL> {
        let rbf: f64 = self
            .centers
            .iter()
            .zip(&self.weights)
            .map(|(c, w)| w * self.kernel.value(((x - c[0]).powi(2) + (y - c[1]).powi(2)).sqrt()))
            .sum();
        Ok(rbf + self.affine[0] + self.affine[1] * x + self.affine[2] * y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scattered() -> (R1D, R1D) {
        // A deterministic quasi-random scatter in the unit square.
        let n = 60;
        let x = R1D::from_iter((0..n).map(|k| (k as f64 * 0.618_033_988_75).fract()));
        let y = R1D::from_iter((0..n).map(|k| (k as f64 * 0.754_877_666_25).fract()));
        (x, y)
    }

    #[test]
    fn interpolates_data_and_reproduces_planes_with_every_kernel() {
        let (x, y) = scattered();
        let plane = |a: f64, b: f64| 1.5 - 2.0 * a + 0.5 * b;
        let f = R1D::from_iter(x.iter().zip(y.iter()).map(|(&a, &b)| plane(a, b)));
        for k in [
            RadialKernel::Gaussian(2.0),
            RadialKernel::Multiquadric(2.0),
            RadialKernel::InverseMultiquadric(2.0),
            RadialKernel::Cubic,
            RadialKernel::ThinPlateSpline,
        ] {
            let r = RadialBasisInterpolator::new(x.view(), y.view(), f.view(), k).unwrap();
            for i in [0, 10, 33] {
                assert!((r.interpolate_at(x[i], y[i]).unwrap() - f[i]).abs() < 1e-7, "{k:?}");
            }
            assert!((r.interpolate_at(0.37, 0.61).unwrap() - plane(0.37, 0.61)).abs() < 1e-6, "{k:?}");
        }
    }

    #[test]
    fn approximates_a_smooth_surface() {
        let (x, y) = scattered();
        let g = |a: f64, b: f64| (2.0 * a).sin() * (1.0 + b);
        let f = R1D::from_iter(x.iter().zip(y.iter()).map(|(&a, &b)| g(a, b)));
        let r = RadialBasisInterpolator::new(x.view(), y.view(), f.view(), RadialKernel::ThinPlateSpline).unwrap();
        assert!((r.interpolate_at(0.5, 0.5).unwrap() - g(0.5, 0.5)).abs() < 2e-2);
    }

    #[test]
    fn duplicate_points_are_rejected() {
        let x = R1D::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y = R1D::from_vec(vec![0.0, 0.0, 1.0, 0.0]);
        let f = R1D::from_vec(vec![1.0, 1.0, 2.0, 0.0]);
        assert!(RadialBasisInterpolator::new(x.view(), y.view(), f.view(), RadialKernel::Cubic).is_err());
    }
}
