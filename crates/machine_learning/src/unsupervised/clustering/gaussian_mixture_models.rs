//! Gaussian mixture models fitted by expectation-maximisation.

use crate::common::{Clusterer, argmax, check_features};
use crate::unsupervised::clustering::k_means::KMeans;
use lib_jsl_core::{ErrorsJSL, R1D, R2D, VR2D};
use ndarray::Array2;
use lib_jsl_linalg::{Cholesky, Diag, SolveTriangular, UPLO};

/// The shape of each component's covariance matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovarianceType {
    /// A full matrix per component: arbitrary ellipsoids with any orientation.
    Full,
    /// Axis-aligned ellipsoids (independent features) per component: fewer parameters.
    Diagonal,
}

/// A mixture `p(x) = Σ_k w_k N(x; μ_k, Σ_k)` fitted by EM: the E-step computes each component's
/// responsibility for each point, the M-step re-estimates weights, means and covariances from the
/// responsibility-weighted data, and the log-likelihood never decreases. Initialised from a k-means
/// run. `reg_covar` is added to every covariance diagonal so components cannot collapse onto single
/// points. Unlike k-means this is a *soft* clustering with elliptical clusters, and gives a
/// probability model: density, membership probabilities, and the BIC/AIC for choosing `k`.
#[derive(Debug, Clone)]
pub struct GaussianMixture {
    k: usize,
    covariance: CovarianceType,
    max_iterations: usize,
    tolerance: f64,
    reg_covar: f64,
    seed: u64,
    weights: Vec<f64>,
    means: R2D,
    covariances: Vec<R2D>,
    chol: Vec<R2D>,
    log_likelihood: f64,
    iterations: usize,
    samples: usize,
}

impl GaussianMixture {
    pub fn new(k: usize, covariance: CovarianceType, seed: u64) -> Self {
        GaussianMixture {
            k,
            covariance,
            max_iterations: 200,
            tolerance: 1e-6,
            reg_covar: 1e-6,
            seed,
            weights: Vec::new(),
            means: R2D::zeros((0, 0)),
            covariances: Vec::new(),
            chol: Vec::new(),
            log_likelihood: f64::NAN,
            iterations: 0,
            samples: 0,
        }
    }

    pub fn with_limits(mut self, max_iterations: usize, tolerance: f64) -> Self {
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn means(&self) -> &R2D {
        &self.means
    }

    pub fn covariances(&self) -> &[R2D] {
        &self.covariances
    }

    /// The mean log-likelihood per sample on the training data.
    pub fn mean_log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    fn parameter_count(&self) -> usize {
        let p = self.means.ncols();
        let cov = match self.covariance {
            CovarianceType::Full => p * (p + 1) / 2,
            CovarianceType::Diagonal => p,
        };
        self.k * (p + cov) + self.k - 1
    }

    /// Bayesian information criterion on the training data (lower is better).
    pub fn bic(&self) -> f64 {
        -2.0 * self.log_likelihood * self.samples as f64 + self.parameter_count() as f64 * (self.samples as f64).ln()
    }

    /// Akaike information criterion on the training data (lower is better).
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood * self.samples as f64 + 2.0 * self.parameter_count() as f64
    }

    fn component_log_pdf(&self, c: usize, x: &[f64]) -> f64 {
        let p = x.len();
        let diff = R1D::from_iter((0..p).map(|j| x[j] - self.means[[c, j]]));
        let l = &self.chol[c];
        let z = l.solve_triangular(UPLO::Lower, Diag::NonUnit, &diff).expect("triangular solve");
        let log_det: f64 = (0..p).map(|j| l[[j, j]].ln()).sum::<f64>() * 2.0;
        -0.5 * (p as f64 * std::f64::consts::TAU.ln() + log_det + z.dot(&z))
    }

    fn log_joint(&self, x: &[f64]) -> Vec<f64> {
        (0..self.k).map(|c| self.weights[c].max(1e-300).ln() + self.component_log_pdf(c, x)).collect()
    }

    /// Membership probabilities, `(rows, components)`.
    pub fn predict_proba(&self, x: VR2D) -> Result<R2D, ErrorsJSL> {
        self.check(x)?;
        let mut out = R2D::zeros((x.nrows(), self.k));
        for (i, row) in x.rows().into_iter().enumerate() {
            let lj = self.log_joint(&row.to_vec());
            let m = lj.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = lj.iter().map(|v| (v - m).exp()).sum();
            for c in 0..self.k {
                out[[i, c]] = (lj[c] - m).exp() / total;
            }
        }
        Ok(out)
    }

    /// The most probable component of each row.
    pub fn predict(&self, x: VR2D) -> Result<Vec<usize>, ErrorsJSL> {
        Ok(self.predict_proba(x)?.rows().into_iter().map(|r| argmax(&r.to_vec()).unwrap_or(0)).collect())
    }

    /// The log of the mixture density at each row.
    pub fn score_samples(&self, x: VR2D) -> Result<Vec<f64>, ErrorsJSL> {
        self.check(x)?;
        Ok(x.rows()
            .into_iter()
            .map(|r| {
                let lj = self.log_joint(&r.to_vec());
                let m = lj.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                m + lj.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
            })
            .collect())
    }

    fn check(&self, x: VR2D) -> Result<(), ErrorsJSL> {
        if self.weights.is_empty() {
            return Err(ErrorsJSL::Misconfiguration("The model has not been fitted."));
        }
        check_features(x)?;
        if x.ncols() != self.means.ncols() {
            return Err(ErrorsJSL::IncompatibleArraySizes((x.ncols(), self.means.ncols())));
        }
        Ok(())
    }

    /// M-step from responsibilities `r` (`n × k`).
    fn maximise(&mut self, x: VR2D, r: &R2D) -> Result<(), ErrorsJSL> {
        let (n, p) = x.dim();
        self.weights = (0..self.k).map(|c| r.column(c).sum() / n as f64).collect();
        self.means = R2D::zeros((self.k, p));
        self.covariances = Vec::with_capacity(self.k);
        self.chol = Vec::with_capacity(self.k);
        for c in 0..self.k {
            let nk = r.column(c).sum().max(1e-12);
            for j in 0..p {
                self.means[[c, j]] = (0..n).map(|i| r[[i, c]] * x[[i, j]]).sum::<f64>() / nk;
            }
            let mut cov = Array2::<f64>::zeros((p, p));
            for i in 0..n {
                let d: Vec<f64> = (0..p).map(|j| x[[i, j]] - self.means[[c, j]]).collect();
                for a in 0..p {
                    match self.covariance {
                        CovarianceType::Full => (0..p).for_each(|b| cov[[a, b]] += r[[i, c]] * d[a] * d[b]),
                        CovarianceType::Diagonal => cov[[a, a]] += r[[i, c]] * d[a] * d[a],
                    }
                }
            }
            cov /= nk;
            for a in 0..p {
                cov[[a, a]] += self.reg_covar;
            }
            let l = cov.cholesky(UPLO::Lower).map_err(|_| ErrorsJSL::RuntimeError("A component covariance is not positive definite."))?;
            self.covariances.push(cov);
            self.chol.push(l);
        }
        Ok(())
    }
}

impl Clusterer for GaussianMixture {
    fn fit_predict(&mut self, x: VR2D) -> Result<Vec<Option<usize>>, ErrorsJSL> {
        check_features(x)?;
        let n = x.nrows();
        if self.k == 0 || self.k > n {
            return Err(ErrorsJSL::InvalidInputRange("k must be between 1 and the number of samples."));
        }
        // Initial responsibilities from k-means (hard assignments).
        let init = KMeans::new(self.k, self.seed).fit_predict(x)?;
        let mut resp = R2D::zeros((n, self.k));
        for (i, l) in init.iter().enumerate() {
            resp[[i, l.expect("k-means labels every point")]] = 1.0;
        }
        self.samples = n;
        self.maximise(x, &resp)?;
        let mut previous = f64::NEG_INFINITY;
        self.iterations = 0;
        for it in 1..=self.max_iterations {
            self.iterations = it;
            // E-step.
            let mut total = 0.0;
            for (i, row) in x.rows().into_iter().enumerate() {
                let lj = self.log_joint(&row.to_vec());
                let m = lj.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let s: f64 = lj.iter().map(|v| (v - m).exp()).sum();
                total += m + s.ln();
                for c in 0..self.k {
                    resp[[i, c]] = (lj[c] - m).exp() / s;
                }
            }
            self.log_likelihood = total / n as f64;
            if (self.log_likelihood - previous).abs() < self.tolerance {
                break;
            }
            previous = self.log_likelihood;
            self.maximise(x, &resp)?;
            if it == self.max_iterations {
                // Stopped at the limit after an M-step: report the likelihood of the final parameters.
                self.log_likelihood = self.score_samples(x)?.iter().sum::<f64>() / n as f64;
            }
        }
        Ok(self.predict(x)?.into_iter().map(Some).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::make_blobs;
    use crate::metrics::adjusted_rand_index;
    use ndarray::{Axis, concatenate};

    fn truth(l: &[usize]) -> Vec<Option<usize>> {
        l.iter().map(|&c| Some(c)).collect()
    }

    #[test]
    fn reported_likelihood_matches_the_final_parameters_even_at_the_iteration_limit() {
        let (x, _) = make_blobs(60, &[vec![0.0, 0.0], vec![3.0, 1.0]], 1.0, 2);
        for limit in [1, 2, 5] {
            let mut g = GaussianMixture::new(2, CovarianceType::Full, 1).with_limits(limit, 0.0);
            g.fit_predict(x.view()).unwrap();
            let mean = g.score_samples(x.view()).unwrap().iter().sum::<f64>() / x.nrows() as f64;
            assert!((g.mean_log_likelihood() - mean).abs() < 1e-12, "limit {limit}");
        }
    }

    #[test]
    fn recovers_blob_parameters() {
        let centres = vec![vec![0.0, 0.0], vec![6.0, 1.0], vec![2.0, 7.0]];
        let (x, l) = make_blobs(150, &centres, 0.7, 4);
        let mut g = GaussianMixture::new(3, CovarianceType::Full, 1);
        let got = g.fit_predict(x.view()).unwrap();
        assert!((adjusted_rand_index(&got, &truth(&l)).unwrap() - 1.0).abs() < 1e-9);
        for w in g.weights() {
            assert!((w - 1.0 / 3.0).abs() < 0.02);
        }
        // Every true centre has a fitted mean within a small distance, and variance ≈ 0.49.
        for c in &centres {
            let best = (0..3).map(|k| ((g.means()[[k, 0]] - c[0]).powi(2) + (g.means()[[k, 1]] - c[1]).powi(2)).sqrt()).fold(f64::INFINITY, f64::min);
            assert!(best < 0.2, "{best}");
        }
        // The clusters barely overlap, so each fitted covariance is the sample covariance (divisor n)
        // of the points generated from the matching centre.
        for k in 0..3 {
            let c = (0..3).min_by(|&a, &b| {
                let d = |c: usize| (g.means()[[k, 0]] - centres[c][0]).powi(2) + (g.means()[[k, 1]] - centres[c][1]).powi(2);
                d(a).total_cmp(&d(b))
            }).unwrap();
            let idx: Vec<usize> = (0..x.nrows()).filter(|&i| l[i] == c).collect();
            let m: Vec<f64> = (0..2).map(|j| idx.iter().map(|&i| x[[i, j]]).sum::<f64>() / idx.len() as f64).collect();
            for (a, b) in [(0, 0), (0, 1), (1, 1)] {
                let sample = idx.iter().map(|&i| (x[[i, a]] - m[a]) * (x[[i, b]] - m[b])).sum::<f64>() / idx.len() as f64;
                assert!((g.covariances()[k][[a, b]] - sample).abs() < 1e-3, "{} vs {sample}", g.covariances()[k][[a, b]]);
            }
        }
        assert!(g.iterations() >= 1 && g.mean_log_likelihood().is_finite());
    }

    #[test]
    fn fits_elongated_clusters_that_k_means_splits_wrongly_and_bic_picks_k() {
        // Two long, thin, parallel Gaussians: full covariances capture them, k-means cuts across.
        let (mut a, _) = make_blobs(200, &[vec![0.0, 0.0]], 1.0, 1);
        let (mut b, _) = make_blobs(200, &[vec![0.0, 0.0]], 1.0, 2);
        a.column_mut(0).mapv_inplace(|v| v * 4.0);
        b.column_mut(0).mapv_inplace(|v| v * 4.0);
        b.column_mut(1).mapv_inplace(|v| v + 4.0);
        let x = concatenate(Axis(0), &[a.view(), b.view()]).unwrap();
        let t: Vec<Option<usize>> = (0..400).map(|i| Some(usize::from(i >= 200))).collect();
        let mut gm = GaussianMixture::new(2, CovarianceType::Full, 3);
        let gl = gm.fit_predict(x.view()).unwrap();
        let km = KMeans::new(2, 3).fit_predict(x.view()).unwrap();
        // The best possible labelling is the optimal rule for the true model, y > 2; the mixture
        // should match it (the classes overlap, so neither is perfect).
        let optimal: Vec<Option<usize>> = (0..400).map(|i| Some(usize::from(x[[i, 1]] > 2.0))).collect();
        assert!(adjusted_rand_index(&gl, &t).unwrap() >= adjusted_rand_index(&optimal, &t).unwrap() - 0.02);
        assert!(adjusted_rand_index(&gl, &t).unwrap() > adjusted_rand_index(&km, &t).unwrap());
        // BIC is minimised at the true number of components on well-separated blobs.
        let (bx, _) = make_blobs(100, &[vec![0.0, 0.0], vec![8.0, 0.0], vec![0.0, 8.0]], 0.8, 6);
        let bic = |k: usize| {
            let mut g = GaussianMixture::new(k, CovarianceType::Full, 1);
            g.fit_predict(bx.view()).unwrap();
            g.bic()
        };
        assert!(bic(3) < bic(2) && bic(3) < bic(5), "{} {} {}", bic(2), bic(3), bic(5));
    }

    #[test]
    fn soft_assignments_density_and_diagonal_covariance() {
        let (x, _) = make_blobs(80, &[vec![0.0, 0.0], vec![5.0, 0.0]], 0.6, 8);
        let mut g = GaussianMixture::new(2, CovarianceType::Diagonal, 2);
        g.fit_predict(x.view()).unwrap();
        let p = g.predict_proba(x.view()).unwrap();
        assert!(p.rows().into_iter().all(|r| (r.sum() - 1.0).abs() < 1e-12));
        // A point midway between the blobs is uncertain; one at a centre is confident.
        let mid = g.predict_proba(ndarray::arr2(&[[2.5, 0.0], [0.0, 0.0]]).view()).unwrap();
        assert!(mid[[0, 0]].max(mid[[0, 1]]) < 0.9 && mid[[1, 0]].max(mid[[1, 1]]) > 0.99);
        let dens = g.score_samples(ndarray::arr2(&[[0.0, 0.0], [30.0, 30.0]]).view()).unwrap();
        assert!(dens[0] > dens[1]);
        assert!(g.covariances().iter().all(|c| c[[0, 1]] == 0.0));
        let mut full = GaussianMixture::new(2, CovarianceType::Full, 2);
        full.fit_predict(x.view()).unwrap();
        assert!(g.aic() < full.aic() + 20.0); // fewer parameters, comparable fit
    }

    #[test]
    fn errors() {
        let (x, _) = make_blobs(5, &[vec![0.0, 0.0]], 1.0, 1);
        assert!(GaussianMixture::new(0, CovarianceType::Full, 1).fit_predict(x.view()).is_err());
        assert!(GaussianMixture::new(6, CovarianceType::Full, 1).fit_predict(x.view()).is_err());
        assert!(GaussianMixture::new(2, CovarianceType::Full, 1).predict(x.view()).is_err());
        // Identical points: the regulariser keeps the covariance positive definite.
        let same = R2D::ones((20, 2));
        assert!(GaussianMixture::new(1, CovarianceType::Full, 1).fit_predict(same.view()).is_ok());
    }
}
