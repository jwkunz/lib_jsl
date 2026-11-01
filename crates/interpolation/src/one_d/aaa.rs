//! AAA rational approximation (Nakatsukasa, Sète and Trefethen, 2018) for real data.

use lib_jsl_core::*;
use ndarray::Array2;
use lib_jsl_linalg::SVD;

/// A barycentric rational approximant `r(x) = Σ w_k f_k / (x - z_k) / Σ w_k / (x - z_k)` chosen
/// greedily from the sample points to reach a target accuracy.
#[derive(Debug, Clone)]
pub struct AaaApproximation {
    support_points: Vec<f64>,
    support_values: Vec<f64>,
    weights: Vec<f64>,
}

impl AaaApproximation {
    /// Fits samples `f(z)` (distinct `z`) with relative tolerance `tol` (e.g. `1e-13`) and at most
    /// `max_terms` support points.
    pub fn new(z: VR1D, f: VR1D, tol: f64, max_terms: usize) -> Result<Self, ErrorsJSL> {
        let m = z.len();
        if m != f.len() || m < 2 {
            return Err(ErrorsJSL::IncompatibleArraySizes((m, f.len())));
        }
        let f_norm = f.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        let mut remaining: Vec<usize> = (0..m).collect();
        let mut support: Vec<usize> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mean = f.sum() / m as f64;
        let mut approx = vec![mean; m];
        for _ in 0..max_terms.min(m) {
            // Greedy: add the point with the worst current error.
            let (pos, &worst) = remaining
                .iter()
                .enumerate()
                .max_by(|a, b| (f[*a.1] - approx[*a.1]).abs().total_cmp(&(f[*b.1] - approx[*b.1]).abs()))
                .ok_or(ErrorsJSL::RuntimeError("no points left"))?;
            if (f[worst] - approx[worst]).abs() <= tol * f_norm {
                break;
            }
            support.push(worst);
            remaining.remove(pos);
            let k = support.len();
            if remaining.len() < k {
                // Not enough rows left for a null-space estimate; keep the previous fit.
                support.pop();
                break;
            }
            // Loewner matrix on the remaining points.
            let a = Array2::from_shape_fn((remaining.len(), k), |(i, j)| {
                let (r, s) = (remaining[i], support[j]);
                (f[r] - f[s]) / (z[r] - z[s])
            });
            let (_, _, vt) = a.svd(false, true).map_err(|_| ErrorsJSL::RuntimeError("SVD failed"))?;
            let vt = vt.ok_or(ErrorsJSL::RuntimeError("SVD returned no right vectors"))?;
            weights = vt.row(vt.nrows() - 1).to_vec();
            // Fit residual at the remaining points.
            let sz: Vec<f64> = support.iter().map(|&s| z[s]).collect();
            let sf: Vec<f64> = support.iter().map(|&s| f[s]).collect();
            for &r in &remaining {
                approx[r] = eval(&sz, &sf, &weights, z[r]);
            }
            for &s in &support {
                approx[s] = f[s];
            }
        }
        if support.is_empty() {
            // Constant data: the degree-0 fit.
            return Ok(AaaApproximation { support_points: vec![z[0]], support_values: vec![f[0]], weights: vec![1.0] });
        }
        Ok(AaaApproximation {
            support_points: support.iter().map(|&s| z[s]).collect(),
            support_values: support.iter().map(|&s| f[s]).collect(),
            weights,
        })
    }

    /// Number of support points (the degree of the rational function is one less).
    pub fn terms(&self) -> usize {
        self.support_points.len()
    }

    /// Evaluates the approximant at `x`.
    pub fn evaluate(&self, x: f64) -> f64 {
        eval(&self.support_points, &self.support_values, &self.weights, x)
    }
}

fn eval(z: &[f64], f: &[f64], w: &[f64], x: f64) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for k in 0..z.len() {
        let d = x - z[k];
        if d == 0.0 {
            return f[k];
        }
        let t = w[k] / d;
        num += t * f[k];
        den += t;
    }
    num / den
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_rational_function_exactly() {
        let g = |x: f64| (x * x + 1.0) / (x * x - 2.0 * x + 5.0); // poles off the real line
        let z = R1D::from_iter((0..200).map(|k| -3.0 + 6.0 * k as f64 / 199.0));
        let f = z.mapv(g);
        let a = AaaApproximation::new(z.view(), f.view(), 1e-13, 30).unwrap();
        assert!(a.terms() <= 6, "terms {}", a.terms());
        for k in 0..50 {
            let x = -2.9 + k as f64 * 0.117;
            assert!((a.evaluate(x) - g(x)).abs() < 1e-9, "x={x}");
        }
    }

    #[test]
    fn approximates_a_nonrational_function_and_extrapolates_gracefully() {
        let z = R1D::from_iter((0..400).map(|k| -1.0 + 2.0 * k as f64 / 399.0));
        let f = z.mapv(|x| (3.0 * x).exp() * x.cos());
        let a = AaaApproximation::new(z.view(), f.view(), 1e-12, 40).unwrap();
        for k in 0..30 {
            let x = -0.98 + k as f64 * 0.0657;
            let want = (3.0 * x).exp() * x.cos();
            assert!((a.evaluate(x) - want).abs() < 1e-8 * want.abs().max(1.0), "x={x}");
        }
    }

    #[test]
    fn constant_data_and_bad_input() {
        let z = R1D::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let f = R1D::from_vec(vec![2.0; 4]);
        let a = AaaApproximation::new(z.view(), f.view(), 1e-13, 10).unwrap();
        assert_eq!(a.evaluate(1.7), 2.0);
        assert!(AaaApproximation::new(z.view(), R1D::zeros(3).view(), 1e-13, 5).is_err());
    }
}
