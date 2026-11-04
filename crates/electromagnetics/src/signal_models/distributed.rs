//! Distributed sources: brightness- or noise-temperature distributions over the sky.
//!
//! See the book entry [Arrays, scenes and beamformers](https://jwkunz.github.io/lib_jsl/book/em-arrays/beamforming.html).

use super::Coherency;
use crate::arrays::SensorArray;
use crate::conventions::C64;
use crate::fields::domain::Domain;
use crate::fields::field_container::{FieldQuantity, HarmonicField};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_core::constants::physics::K_B;
use ndarray::Array2;
use std::f64::consts::PI;
use std::sync::Arc;

/// How the distributed emission is polarized.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SkyPolarization {
    /// Unpolarized emission (thermal noise): each polarization carries the brightness temperature.
    Unpolarized,
    /// Partially or fully polarized emission with unit-trace coherency `J`; the total brightness
    /// `2T` is shared between the polarizations according to `J` (so `J = ½ I` is unpolarized).
    Polarized(Coherency),
}

/// A brightness-temperature map `T(Ω)` (K) on a far-field domain: the sky, the ground, a hot
/// spot, any spatially distributed thermal scene. The map's domain also sets the quadrature used
/// to integrate it.
#[derive(Debug, Clone)]
pub struct BrightnessScene {
    temperature: Arc<HarmonicField>,
    polarization: SkyPolarization,
}

impl BrightnessScene {
    /// A scene from a scalar temperature field on a far-field domain (the real parts are used).
    pub fn new(temperature: Arc<HarmonicField>, polarization: SkyPolarization) -> Result<Self, ErrorsJSL> {
        if !temperature.domain().is_far_field() || !temperature.is_scalar() || temperature.quantity() != FieldQuantity::Temperature {
            return Err(ErrorsJSL::InvalidInputRange("A brightness scene needs a scalar temperature map on the far-field sphere."));
        }
        if temperature.values().iter().any(|v| !(v.re.is_finite() && v.re >= 0.0)) {
            return Err(ErrorsJSL::InvalidInputRange("Brightness temperatures must be finite and non-negative."));
        }
        Ok(BrightnessScene { temperature, polarization })
    }

    /// A map built from a function `T(θ, φ)` on `domain`, at the array's frequency (the value is
    /// only a tag for the container; temperatures are frequency-independent here).
    pub fn from_fn(domain: Arc<Domain>, frequency: f64, polarization: SkyPolarization, t: impl Fn(f64, f64) -> f64) -> Result<Self, ErrorsJSL> {
        let map = HarmonicField::scalar_from_fn(domain, FieldQuantity::Temperature, frequency, crate::conventions::TimeConvention::PlusJ, |p| C64::new(t(p.coords[1], p.coords[2]), 0.0))?;
        Self::new(Arc::new(map), polarization)
    }

    pub fn temperature_map(&self) -> &Arc<HarmonicField> {
        &self.temperature
    }

    /// The covariance `R = k_B B / (4π) ∫ T(Ω) A(Ω) M A(Ω)ᴴ dΩ` (W) over bandwidth `B`, with
    /// `M = I` for unpolarized emission and `2J` for polarized. A single co-polarized element
    /// therefore sees `k_B B T_A` with the antenna temperature `T_A = (1/4π) ∫ G T dΩ`.
    pub fn covariance(&self, array: &SensorArray, bandwidth: f64) -> Result<Array2<C64>, ErrorsJSL> {
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("The bandwidth must be positive."));
        }
        let m: [[C64; 2]; 2] = match self.polarization {
            SkyPolarization::Unpolarized => [[C64::new(1.0, 0.0), C64::new(0.0, 0.0)], [C64::new(0.0, 0.0), C64::new(1.0, 0.0)]],
            SkyPolarization::Polarized(j) => j.0.map(|row| row.map(|v| v * 2.0)),
        };
        let n = array.len();
        let domain = self.temperature.domain();
        let mut r = Array2::<C64>::zeros((n, n));
        for i in 0..domain.len() {
            let t = self.temperature.scalar(i).re;
            let w = domain.weights()[i] * t;
            if w == 0.0 {
                continue;
            }
            let p = domain.position_flat(i);
            let a = array.response_matrix(p.coords[1], p.coords[2])?;
            // A M Aᴴ, accumulated.
            for row in 0..n {
                let am = [a[[row, 0]] * m[0][0] + a[[row, 1]] * m[1][0], a[[row, 0]] * m[0][1] + a[[row, 1]] * m[1][1]];
                for col in 0..n {
                    r[[row, col]] += (am[0] * a[[col, 0]].conj() + am[1] * a[[col, 1]].conj()) * w;
                }
            }
        }
        Ok(r.mapv(|v| v * (K_B * bandwidth / (4.0 * PI))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::antenna_models::{HalfWaveDipole, Isotropic};
    use crate::conventions::TimeConvention;
    use crate::coordinates::transforms::RigidTransform;
    use crate::signal_models::PointSource;
    use lib_jsl_core::constants::em::C0;

    #[test]
    fn uniform_sky_gives_k_t_b_per_element_regardless_of_the_pattern() {
        let f = 1e9;
        let b = 1e6;
        let sky = BrightnessScene::from_fn(Arc::new(Domain::far_field_gauss(40, 80).unwrap()), f, SkyPolarization::Unpolarized, |_, _| 290.0).unwrap();
        for antenna in [Arc::new(Isotropic::theta_polarized()) as Arc<dyn crate::antenna_models::Antenna + Send + Sync>, Arc::new(HalfWaveDipole::new(f))] {
            let array = SensorArray::from_antenna(&[[0.0; 3]], antenna, &RigidTransform::identity(), f, TimeConvention::PlusJ, 1.0).unwrap();
            let r = sky.covariance(&array, b).unwrap();
            // ∫ G dΩ = 4π for a lossless antenna, so T_A = T.
            assert!((r[[0, 0]].re - K_B * b * 290.0).abs() < 2e-3 * K_B * b * 290.0, "{}", r[[0, 0]].re / (K_B * b));
        }
    }

    #[test]
    fn a_small_hot_spot_matches_the_equivalent_point_source() {
        // Rayleigh-Jeans: a patch of brightness T over solid angle ΔΩ has flux S = 2 k_B T ΔΩ B / λ²
        // (both polarizations); the distributed and point-source models must agree.
        let f = 1e9;
        let lambda = C0 / f;
        let b = 1e6;
        let d = Arc::new(Domain::far_field_uniform(181, 360).unwrap());
        let (t0, p0) = (d.position(&[60, 45]).coords[1], d.position(&[60, 45]).coords[2]);
        let spot = BrightnessScene::from_fn(d.clone(), f, SkyPolarization::Unpolarized, |t, p| if (t - t0).abs() < 1e-9 && (p - p0).abs() < 1e-9 { 1000.0 } else { 0.0 }).unwrap();
        let omega = d.weights()[d.flatten(&[60, 45])];
        let array = SensorArray::from_antenna(&SensorArray::linear_positions(3, lambda / 2.0, [1.0, 0.0, 0.0]), Arc::new(Isotropic::theta_polarized()), &RigidTransform::identity(), f, TimeConvention::PlusJ, 1.0).unwrap();
        let r_spot = spot.covariance(&array, b).unwrap();
        let point = PointSource::new(t0, p0, 2.0 * K_B * 1000.0 * omega * b / (lambda * lambda), Coherency::unpolarized()).unwrap();
        let r_point = point.covariance(&array).unwrap();
        assert!(r_spot.iter().zip(r_point.iter()).all(|(a, c)| (a - c).norm() < 1e-9 * r_point[[0, 0]].norm()));
        // Validation.
        let scalar_e = HarmonicField::zeros(d.clone(), FieldQuantity::Generic, crate::fields::field_container::Rank::Scalar, f, TimeConvention::PlusJ).unwrap();
        assert!(BrightnessScene::new(Arc::new(scalar_e), SkyPolarization::Unpolarized).is_err());
        assert!(BrightnessScene::from_fn(d.clone(), f, SkyPolarization::Unpolarized, |_, _| -1.0).is_err());
        assert!(spot.covariance(&array, 0.0).is_err());
        let polarized = BrightnessScene::from_fn(d, f, SkyPolarization::Polarized(Coherency::phi()), |_, _| 10.0).unwrap();
        assert!(polarized.covariance(&array, b).unwrap().iter().all(|v| v.norm() < 1e-25)); // θ-pol elements see nothing
    }
}
