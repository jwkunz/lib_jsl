//! Complete scenes, their array covariance, and covariance estimation from snapshots.

use super::distributed::BrightnessScene;
use super::point_source::PointSource;
use crate::arrays::SensorArray;
use crate::conventions::C64;
use crate::linalg::{check_hermitian, hermitian_eigen};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_core::constants::physics::K_B;
use lib_jsl_random::distributions::gaussian::gaussian_distribution_box_muller;
use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;
use ndarray::Array2;

/// Everything the array sees over a bandwidth: point sources, distributed brightness scenes and
/// the receivers' own noise (an independent noise temperature per channel).
#[derive(Debug, Clone)]
pub struct Scene {
    pub bandwidth: f64,
    pub point_sources: Vec<PointSource>,
    pub distributed: Vec<BrightnessScene>,
    pub receiver_temperature: f64,
}

impl Scene {
    pub fn new(bandwidth: f64) -> Result<Self, ErrorsJSL> {
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("The bandwidth must be positive."));
        }
        Ok(Scene { bandwidth, point_sources: Vec::new(), distributed: Vec::new(), receiver_temperature: 0.0 })
    }

    pub fn with_point_source(mut self, source: PointSource) -> Self {
        self.point_sources.push(source);
        self
    }

    pub fn with_distributed(mut self, scene: BrightnessScene) -> Self {
        self.distributed.push(scene);
        self
    }

    pub fn with_receiver_temperature(mut self, temperature: f64) -> Self {
        self.receiver_temperature = temperature;
        self
    }

    /// `k_B T_rx B I`, the receivers' uncorrelated noise.
    pub fn receiver_noise_covariance(&self, array: &SensorArray) -> Array2<C64> {
        Array2::from_diag_elem(array.len(), C64::new(K_B * self.receiver_temperature * self.bandwidth, 0.0))
    }

    /// The covariance of the point sources only.
    pub fn point_source_covariance(&self, array: &SensorArray) -> Result<Array2<C64>, ErrorsJSL> {
        let mut r = Array2::<C64>::zeros((array.len(), array.len()));
        for s in &self.point_sources {
            r += &s.covariance(array)?;
        }
        Ok(r)
    }

    /// The covariance of the distributed scenes only.
    pub fn distributed_covariance(&self, array: &SensorArray) -> Result<Array2<C64>, ErrorsJSL> {
        let mut r = Array2::<C64>::zeros((array.len(), array.len()));
        for s in &self.distributed {
            r += &s.covariance(array, self.bandwidth)?;
        }
        Ok(r)
    }

    /// The total covariance: point sources, distributed scenes and receiver noise.
    pub fn covariance(&self, array: &SensorArray) -> Result<Array2<C64>, ErrorsJSL> {
        Ok(self.point_source_covariance(array)? + self.distributed_covariance(array)? + self.receiver_noise_covariance(array))
    }
}

/// The sample covariance `X Xᴴ / K` of snapshots stored as the columns of an `N × K` matrix.
pub fn sample_covariance(snapshots: &Array2<C64>) -> Result<Array2<C64>, ErrorsJSL> {
    let k = snapshots.ncols();
    if k == 0 || snapshots.nrows() == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Need at least one snapshot of at least one channel."));
    }
    Ok(snapshots.dot(&snapshots.t().mapv(|v| v.conj())) / C64::new(k as f64, 0.0))
}

/// `count` independent circularly-symmetric complex Gaussian snapshots with covariance `R`
/// (`N × count`), generated as `V √Λ z` from the eigendecomposition of `R` (so singular `R` is
/// fine). Deterministic for a given `seed`.
pub fn synthesize_snapshots(r: &Array2<C64>, count: usize, seed: u64) -> Result<Array2<C64>, ErrorsJSL> {
    let n = r.nrows();
    check_hermitian(r, n)?;
    let (vals, vecs) = hermitian_eigen(r)?;
    let scale: Vec<f64> = vals.iter().map(|&v| v.max(0.0).sqrt()).collect();
    let mut rng = Xoshiro256PlusPlus::from_seed(seed);
    let mut out = Array2::<C64>::zeros((n, count));
    for c in 0..count {
        let z: Vec<C64> = (0..n)
            .map(|_| loop {
                let (a, b) = gaussian_distribution_box_muller(&mut rng, 0.0, std::f64::consts::FRAC_1_SQRT_2);
                if a.is_finite() && b.is_finite() {
                    break C64::new(a, b);
                }
            })
            .collect();
        for i in 0..n {
            out[[i, c]] = (0..n).map(|k| vecs[[i, k]] * scale[k] * z[k]).sum();
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::antenna_models::Isotropic;
    use crate::conventions::TimeConvention;
    use crate::coordinates::transforms::RigidTransform;
    use crate::fields::domain::Domain;
    use crate::signal_models::{Coherency, SkyPolarization};
    use lib_jsl_core::constants::em::C0;
    use std::sync::Arc;

    fn array() -> SensorArray {
        let f = 1e9;
        SensorArray::from_antenna(&SensorArray::linear_positions(4, 0.5 * C0 / f, [1.0, 0.0, 0.0]), Arc::new(Isotropic::theta_polarized()), &RigidTransform::identity(), f, TimeConvention::PlusJ, 1.0).unwrap()
    }

    #[test]
    fn scene_covariance_is_the_sum_of_its_parts_and_hermitian() {
        let a = array();
        let sky = BrightnessScene::from_fn(Arc::new(Domain::far_field_gauss(20, 40).unwrap()), 1e9, SkyPolarization::Unpolarized, |t, _| if t > 1.6 { 290.0 } else { 10.0 }).unwrap();
        let scene = Scene::new(1e6).unwrap().with_point_source(PointSource::new(0.8, 0.3, 1e-15, Coherency::theta()).unwrap()).with_distributed(sky).with_receiver_temperature(50.0);
        let r = scene.covariance(&a).unwrap();
        let parts = scene.point_source_covariance(&a).unwrap() + scene.distributed_covariance(&a).unwrap() + scene.receiver_noise_covariance(&a);
        assert!(r.iter().zip(parts.iter()).all(|(x, y)| (x - y).norm() < 1e-30));
        assert!(check_hermitian(&r, 4).is_ok());
        // Positive definite thanks to the receiver noise.
        let (vals, _) = hermitian_eigen(&r).unwrap();
        assert!(vals.iter().all(|&v| v > 0.0));
        assert!(Scene::new(0.0).is_err());
    }

    #[test]
    fn snapshots_reproduce_the_covariance() {
        let a = array();
        let scene = Scene::new(1e6).unwrap().with_point_source(PointSource::new(1.1, 0.2, 1e-14, Coherency::theta()).unwrap()).with_receiver_temperature(100.0);
        let r = scene.covariance(&a).unwrap();
        let x = synthesize_snapshots(&r, 20_000, 7).unwrap();
        let est = sample_covariance(&x).unwrap();
        let scale = r[[0, 0]].norm();
        let worst = r.iter().zip(est.iter()).map(|(p, q)| (p - q).norm()).fold(0.0, f64::max);
        assert!(worst < 0.05 * scale, "{}", worst / scale);
        assert_eq!(synthesize_snapshots(&r, 3, 7).unwrap(), synthesize_snapshots(&r, 3, 7).unwrap());
        assert!(sample_covariance(&Array2::zeros((2, 0))).is_err());
        assert!(synthesize_snapshots(&Array2::from_shape_fn((2, 2), |(i, j)| C64::new(0.0, (i as f64) - (j as f64) + 1.0)), 1, 1).is_err());
    }
}
