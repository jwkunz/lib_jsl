//! Quasi-Monte Carlo samples from a multivariate normal distribution (SciPy's
//! `stats.qmc.MultivariateNormalQMC`).
//!
//! See the book entry [Multivariate distribution functions and quasi-Monte Carlo sampling](https://jwkunz.github.io/lib_jsl/book/statistics/mvn-cdf.html#quasi-monte-carlo-normal-samples).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_random::quasi_random::SobolSequence;
use lib_jsl_special_functions::normal_quantile;

/// A generator of low-discrepancy multivariate normal samples: Sobol points mapped to standard
/// normals, then correlated by a square root of the covariance.
#[derive(Clone, Debug)]
pub struct MultivariateNormalQmc {
    mean: Vec<f64>,
    /// `d × d`: `x = μ + R z`.
    root: Vec<Vec<f64>>,
    inv_transform: bool,
    engine: SobolSequence,
}

// ANCHOR: mvn_qmc
impl MultivariateNormalQmc {
    /// Samples from `N(mean, cov)` driven by `engine`, a Sobol sequence of dimension `d` (or
    /// `2⌈d/2⌉` with `inv_transform = false`), plain or scrambled. With `inv_transform`, each
    /// coordinate is `Φ⁻¹(½ + (1 − 10⁻¹⁰)(u − ½))` (SciPy's default); otherwise pairs of
    /// coordinates go through the Box-Muller transform. The covariance's square root is its
    /// Cholesky factor, or `V √Λ` from its eigendecomposition when it is only semidefinite.
    pub fn new(mean: Vec<f64>, cov: &[Vec<f64>], inv_transform: bool, engine: SobolSequence) -> Result<Self, ErrorsJSL> {
        let d = mean.len();
        if d == 0 || cov.len() != d || cov.iter().any(|r| r.len() != d) {
            return Err(ErrorsJSL::InvalidInputRange("MultivariateNormalQmc: the mean and covariance must have one dimension"));
        }
        let need = if inv_transform { d } else { 2 * d.div_ceil(2) };
        if engine.dimension() != need {
            return Err(ErrorsJSL::InvalidInputRange("MultivariateNormalQmc: the engine's dimension must be d (or 2⌈d/2⌉ without the inverse transform)"));
        }
        let root = match cholesky(cov) {
            Some(l) => l,
            None => {
                let (lam, v) = crate::multivariate::symmetric_eigen_pub(cov);
                if lam.iter().any(|&l| l < -1e-8) {
                    return Err(ErrorsJSL::InvalidInputRange("MultivariateNormalQmc: the covariance is not positive semidefinite"));
                }
                (0..d).map(|i| (0..d).map(|j| v[i][j] * lam[j].max(0.0).sqrt()).collect()).collect()
            }
        };
        Ok(MultivariateNormalQmc { mean, root, inv_transform, engine })
    }

    /// The next `n` samples.
    pub fn random(&mut self, n: usize) -> Vec<Vec<f64>> {
        let d = self.mean.len();
        (0..n)
            .map(|_| {
                let u = self.engine.next_point();
                let z: Vec<f64> = if self.inv_transform {
                    u.iter().map(|&v| normal_quantile(0.5 + (1.0 - 1e-10) * (v - 0.5), 0.0, 1.0)).collect()
                } else {
                    let mut z = Vec::with_capacity(u.len());
                    for p in u.chunks(2) {
                        let r = (-2.0 * p[0].ln()).sqrt();
                        let t = 2.0 * std::f64::consts::PI * p[1];
                        z.push(r * t.cos());
                        z.push(r * t.sin());
                    }
                    z.truncate(d);
                    z
                };
                (0..d).map(|i| self.mean[i] + (0..d).map(|k| self.root[i][k] * z[k]).sum::<f64>()).collect()
            })
            .collect()
    }
}
// ANCHOR_END: mvn_qmc

fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for j in 0..n {
        let d = a[j][j] - (0..j).map(|k| l[j][k] * l[j][k]).sum::<f64>();
        if d <= 0.0 || !d.is_finite() {
            return None;
        }
        l[j][j] = d.sqrt();
        for i in j + 1..n {
            l[i][j] = (a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>()) / l[j][j];
        }
    }
    Some(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_converge_quickly() {
        let cov = vec![vec![2.0, 0.6], vec![0.6, 1.0]];
        let mut g = MultivariateNormalQmc::new(vec![1.0, -1.0], &cov, true, SobolSequence::new(2).unwrap()).unwrap();
        let s = g.random(4096);
        let n = s.len() as f64;
        let m0 = s.iter().map(|p| p[0]).sum::<f64>() / n;
        let c01 = s.iter().map(|p| (p[0] - 1.0) * (p[1] + 1.0)).sum::<f64>() / n;
        // (The first, unscrambled Sobol point is 0, which maps to −6.5: a small bias, as in SciPy.)
        assert!((m0 - 1.0).abs() < 5e-3 && (c01 - 0.6).abs() < 3e-2, "{m0} {c01}");
        // Singular covariance: the samples lie on the line x₁ = x₀.
        let mut g = MultivariateNormalQmc::new(vec![0.0, 0.0], &[vec![1.0, 1.0], vec![1.0, 1.0]], false, SobolSequence::new(2).unwrap()).unwrap();
        g.random(1);
        assert!(g.random(10).iter().all(|p| (p[0] - p[1]).abs() < 1e-12));
    }
}
