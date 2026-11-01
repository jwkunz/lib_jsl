//! Kernel density estimation with a Gaussian kernel.

use crate::descriptive::{interquartile_range, standard_deviation};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_random::distributions::gaussian_distribution_box_muller;
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_special_functions::normal_cdf;
use std::f64::consts::PI;

/// How to choose the kernel bandwidth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bandwidth {
    /// Scott's rule `σ n^{−1/5}` (SciPy's default factor).
    Scott,
    /// Silverman's rule of thumb `0.9 min(σ, IQR/1.34) n^{−1/5}`, robust to outliers and
    /// heavy tails.
    Silverman,
    /// A given bandwidth.
    Fixed(f64),
}

/// A Gaussian kernel density estimate `f̂(x) = (1/(n h)) Σ φ((x − xᵢ)/h)`.
#[derive(Debug, Clone)]
pub struct GaussianKde {
    data: Vec<f64>,
    bandwidth: f64,
}

impl GaussianKde {
    pub fn new(data: &[f64], bandwidth: Bandwidth) -> Result<Self, ErrorsJSL> {
        if data.len() < 2 || data.iter().any(|v| !v.is_finite()) {
            return Err(ErrorsJSL::InvalidInputRange("Need at least two finite observations."));
        }
        let n = data.len() as f64;
        let h = match bandwidth {
            Bandwidth::Scott => standard_deviation(data, 1)? * n.powf(-0.2),
            Bandwidth::Silverman => {
                let sigma = standard_deviation(data, 1)?;
                let spread = interquartile_range(data)? / 1.34;
                0.9 * if spread > 0.0 { sigma.min(spread) } else { sigma } * n.powf(-0.2)
            }
            Bandwidth::Fixed(h) => h,
        };
        if !(h > 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("The bandwidth must be positive (is the sample constant?)."));
        }
        Ok(GaussianKde { data: data.to_vec(), bandwidth: h })
    }

    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// The estimated density at `x`.
    pub fn pdf(&self, x: f64) -> f64 {
        let h = self.bandwidth;
        self.data.iter().map(|&xi| (-0.5 * ((x - xi) / h).powi(2)).exp()).sum::<f64>() / (self.data.len() as f64 * h * (2.0 * PI).sqrt())
    }

    /// The estimated cumulative distribution at `x`.
    pub fn cdf(&self, x: f64) -> f64 {
        self.data.iter().map(|&xi| normal_cdf(x, xi, self.bandwidth)).sum::<f64>() / self.data.len() as f64
    }

    /// Draws `count` samples from the estimate (a random observation plus kernel noise).
    pub fn resample<U: UniformGenerator>(&self, count: usize, rng: &mut U) -> Vec<f64> {
        (0..count)
            .map(|_| {
                let i = ((rng.next_f64() * self.data.len() as f64) as usize).min(self.data.len() - 1);
                gaussian_distribution_box_muller(rng, self.data[i], self.bandwidth).0
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lib_jsl_random::uniform_generator::UniformRNG;
    use lib_jsl_special_functions::normal_pdf;

    fn normal_sample(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = UniformRNG::from_seed(seed);
        (0..n).map(|_| gaussian_distribution_box_muller(&mut rng, 0.0, 1.0).0).collect()
    }

    #[test]
    fn recovers_a_normal_density() {
        let data = normal_sample(5000, 1);
        let kde = GaussianKde::new(&data, Bandwidth::Scott).unwrap();
        assert!((kde.bandwidth() - 5000f64.powf(-0.2)).abs() < 0.01);
        for &x in &[-2.0, -1.0, 0.0, 0.5, 1.5] {
            assert!((kde.pdf(x) - normal_pdf(x, 0.0, 1.0)).abs() < 0.02, "x={x}");
        }
        // The estimate is a density: it integrates to one, and its CDF runs from 0 to 1.
        let h = 0.01;
        let integral: f64 = (-800..800).map(|i| kde.pdf(i as f64 * h) * h).sum();
        assert!((integral - 1.0).abs() < 1e-4);
        assert!(kde.cdf(-10.0) < 1e-12 && (kde.cdf(10.0) - 1.0).abs() < 1e-12 && (kde.cdf(0.0) - 0.5).abs() < 0.03);
    }

    #[test]
    fn silverman_resists_outliers_and_resampling_matches_moments() {
        let mut data = normal_sample(500, 2);
        data.extend([200.0, -150.0]);
        let scott = GaussianKde::new(&data, Bandwidth::Scott).unwrap().bandwidth();
        let silverman = GaussianKde::new(&data, Bandwidth::Silverman).unwrap().bandwidth();
        assert!(silverman < scott / 5.0, "{silverman} vs {scott}");
        let kde = GaussianKde::new(&normal_sample(2000, 3), Bandwidth::Fixed(0.3)).unwrap();
        let draws = kde.resample(50_000, &mut UniformRNG::from_seed(4));
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let var = draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / draws.len() as f64;
        // Resampled variance is the data variance plus h².
        assert!(mean.abs() < 0.05 && (var - (1.0 + 0.09)).abs() < 0.1, "{mean} {var}");
        assert!(GaussianKde::new(&[1.0, 1.0, 1.0], Bandwidth::Scott).is_err() && GaussianKde::new(&[1.0], Bandwidth::Fixed(1.0)).is_err());
    }
}
