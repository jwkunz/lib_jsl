use crate::{distributions::gaussian::gaussian_distribution_box_muller, uniform_generator::UniformGenerator};
use lib_jsl_core::ErrorsJSL;

/// The multivariate normal distribution `N(μ, Σ)`, sampled as `μ + L z` with the Cholesky
/// factor `Σ = L Lᵀ` computed once and `z` standard normal.
#[derive(Debug, Clone)]
pub struct MultivariateNormal {
    mean: Vec<f64>,
    cholesky: Vec<Vec<f64>>,
}

impl MultivariateNormal {
    /// Fails unless `covariance` is a symmetric positive-definite `n × n` matrix matching `mean`.
    pub fn new(mean: Vec<f64>, covariance: &[Vec<f64>]) -> Result<Self, ErrorsJSL> {
        let n = mean.len();
        if n == 0 || covariance.len() != n || covariance.iter().any(|r| r.len() != n) {
            return Err(ErrorsJSL::IncompatibleArraySizes((covariance.len(), n)));
        }
        for i in 0..n {
            for j in 0..i {
                if (covariance[i][j] - covariance[j][i]).abs() > 1e-12 * (covariance[i][i].abs() + covariance[j][j].abs()) {
                    return Err(ErrorsJSL::InvalidInputRange("The covariance must be symmetric."));
                }
            }
        }
        let mut l = vec![vec![0.0; n]; n];
        for j in 0..n {
            let d = covariance[j][j] - (0..j).map(|k| l[j][k] * l[j][k]).sum::<f64>();
            if d.is_nan() || d <= 0.0 {
                return Err(ErrorsJSL::InvalidInputRange("The covariance must be positive definite."));
            }
            l[j][j] = d.sqrt();
            for i in j + 1..n {
                l[i][j] = (covariance[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>()) / l[j][j];
            }
        }
        Ok(MultivariateNormal { mean, cholesky: l })
    }

    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// The lower-triangular Cholesky factor of the covariance.
    pub fn cholesky_factor(&self) -> &[Vec<f64>] {
        &self.cholesky
    }

    pub fn sample<U: UniformGenerator>(&self, uniform_generator: &mut U) -> Vec<f64> {
        let n = self.mean.len();
        let mut z = Vec::with_capacity(n + 1);
        while z.len() < n {
            let (a, b) = gaussian_distribution_box_muller(uniform_generator, 0.0, 1.0);
            z.push(a);
            z.push(b);
        }
        (0..n).map(|i| self.mean[i] + (0..=i).map(|k| self.cholesky[i][k] * z[k]).sum::<f64>()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uniform_generator::UniformRNG;

    #[test]
    fn sample_mean_and_covariance_match() {
        let cov = vec![vec![4.0, 1.2, -0.6], vec![1.2, 1.0, 0.3], vec![-0.6, 0.3, 0.5]];
        let mvn = MultivariateNormal::new(vec![1.0, -2.0, 0.5], &cov).unwrap();
        let mut rng = UniformRNG::from_seed(21);
        let m = 200_000;
        let samples: Vec<Vec<f64>> = (0..m).map(|_| mvn.sample(&mut rng)).collect();
        let mean: Vec<f64> = (0..3).map(|i| samples.iter().map(|s| s[i]).sum::<f64>() / m as f64).collect();
        for i in 0..3 {
            assert!((mean[i] - [1.0, -2.0, 0.5][i]).abs() < 0.02);
            for j in 0..3 {
                let c = samples.iter().map(|s| (s[i] - mean[i]) * (s[j] - mean[j])).sum::<f64>() / (m - 1) as f64;
                assert!((c - cov[i][j]).abs() < 0.03, "({i},{j}): {c}");
            }
        }
    }

    #[test]
    fn rejects_invalid_covariances() {
        assert!(MultivariateNormal::new(vec![0.0; 2], &[vec![1.0, 2.0], vec![2.0, 1.0]]).is_err()); // indefinite
        assert!(MultivariateNormal::new(vec![0.0; 2], &[vec![1.0, 0.5], vec![0.4, 1.0]]).is_err()); // asymmetric
        assert!(MultivariateNormal::new(vec![0.0; 3], &[vec![1.0]]).is_err());
    }
}
