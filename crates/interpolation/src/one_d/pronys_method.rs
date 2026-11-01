//! Prony's method: fit uniformly sampled data with a sum of complex exponentials.
//!
//! See the book entry [Trigonometric interpolation and Prony's method](https://jwkunz.github.io/lib_jsl/book/interpolation/trigonometric.html#pronys-method). With noisy data, fit more exponentials than the
//! signal contains: the extra ones absorb the noise as strongly damped components.

use lib_jsl_core::*;
use ndarray::{Array1, Array2};
use lib_jsl_linalg::{Eig, LeastSquaresSvd};
use num_complex::Complex;

/// `y(x) = Re Σ_i c_i z_i^((x - x0)/dx)` fitted to samples at `x0 + k dx`. Each `z_i` is a
/// discrete-time pole (`|z|` is the per-sample decay, `arg z` the per-sample phase advance), which
/// makes the model both an interpolator and an extrapolator for damped oscillations.
#[derive(Debug, Clone)]
pub struct PronyModel {
    x0: f64,
    dx: f64,
    poles: Vec<Complex<f64>>,
    amplitudes: Vec<Complex<f64>>,
}

impl PronyModel {
    /// Fits `order` exponentials to `y` (needs `y.len() >= 2 * order`; extra samples are used in a
    /// least-squares sense, which averages down noise).
    pub fn new(y: VR1D, x0: f64, dx: f64, order: usize) -> Result<Self, ErrorsJSL> {
        let n = y.len();
        if order == 0 || n < 2 * order || dx <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("need order > 0, dx > 0 and at least 2*order samples"));
        }
        // ANCHOR: prony_steps
        // Linear prediction: y[k+p] = -Σ_{j=1..p} a_j y[k+p-j] for k = 0..n-p.
        let rows = n - order;
        let a = Array2::from_shape_fn((rows, order), |(k, j)| y[k + order - 1 - j]);
        let b = Array1::from_shape_fn(rows, |k| -y[k + order]);
        let coef = a
            .least_squares(&b)
            .map_err(|_| ErrorsJSL::RuntimeError("prediction least squares failed"))?
            .solution;
        // Poles are the roots of z^p + a_1 z^(p-1) + ... + a_p: eigenvalues of the companion matrix.
        let mut companion = Array2::<f64>::zeros((order, order));
        for j in 0..order {
            companion[[0, j]] = -coef[j];
        }
        for i in 1..order {
            companion[[i, i - 1]] = 1.0;
        }
        let (eigenvalues, _) = companion.eig().map_err(|_| ErrorsJSL::RuntimeError("eigenvalue solve failed"))?;
        let poles: Vec<Complex<f64>> = eigenvalues.to_vec();
        // Amplitudes: least squares on the Vandermonde system y[k] = Σ c_i z_i^k.
        let v = Array2::from_shape_fn((n, order), |(k, i)| poles[i].powi(k as i32));
        let yc = Array1::from_shape_fn(n, |k| Complex::new(y[k], 0.0));
        let amplitudes = v
            .least_squares(&yc)
            .map_err(|_| ErrorsJSL::RuntimeError("amplitude least squares failed"))?
            .solution
            .to_vec();
        // ANCHOR_END: prony_steps
        Ok(PronyModel { x0, dx, poles, amplitudes })
    }

    /// The fitted discrete-time poles `z_i`.
    pub fn poles(&self) -> &[Complex<f64>] {
        &self.poles
    }

    /// The complex amplitudes `c_i`.
    pub fn amplitudes(&self) -> &[Complex<f64>] {
        &self.amplitudes
    }

    /// Continuous-time exponents `s_i = ln(z_i) / dx`, so `y(x) = Re Σ c_i exp(s_i (x - x0))`.
    pub fn exponents(&self) -> Vec<Complex<f64>> {
        self.poles.iter().map(|z| z.ln() / self.dx).collect()
    }

    /// Evaluates the model (the real part of the exponential sum) at `x`.
    pub fn evaluate(&self, x: f64) -> f64 {
        let t = (x - self.x0) / self.dx;
        self.poles
            .iter()
            .zip(&self.amplitudes)
            .map(|(z, c)| (c * Complex::from_polar(z.norm().powf(t), z.arg() * t)).re)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_damped_sinusoid_parameters() {
        let (dx, alpha, omega) = (0.05, -0.8, 7.0);
        let f = |x: f64| 2.0 * (alpha * x).exp() * (omega * x + 0.4).cos();
        let y = R1D::from_iter((0..40).map(|k| f(k as f64 * dx)));
        let m = PronyModel::new(y.view(), 0.0, dx, 2).unwrap();
        // The two poles are conjugates at exp((alpha ± i omega) dx).
        let want = Complex::new(alpha, omega) * dx;
        let want = want.exp();
        assert!(m.poles().iter().any(|z| (z - want).norm() < 1e-9));
        assert!(m.poles().iter().any(|z| (z - want.conj()).norm() < 1e-9));
        // Off-grid interpolation and extrapolation beyond the data.
        for &x in &[0.0123, 0.777, 1.9, 3.3] {
            assert!((m.evaluate(x) - f(x)).abs() < 1e-8, "x={x}");
        }
    }

    #[test]
    fn two_real_exponentials_and_overdetermined_noise() {
        let f = |x: f64| 3.0 * (-x).exp() + 1.5 * (-4.0 * x).exp();
        let y = R1D::from_iter((0..30).map(|k| f(k as f64 * 0.1)));
        let m = PronyModel::new(y.view(), 0.0, 0.1, 2).unwrap();
        let mut s: Vec<f64> = m.exponents().iter().map(|e| e.re).collect();
        s.sort_by(f64::total_cmp);
        assert!((s[0] + 4.0).abs() < 1e-7 && (s[1] + 1.0).abs() < 1e-7, "{s:?}");
        assert!((m.evaluate(0.55) - f(0.55)).abs() < 1e-8);
    }

    #[test]
    fn rejects_too_few_samples() {
        let y = R1D::zeros(3);
        assert!(PronyModel::new(y.view(), 0.0, 1.0, 2).is_err());
    }
}
