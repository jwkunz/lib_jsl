//! Discrete far-field point sources.
//!
//! See the book entry [Arrays, scenes and beamformers](https://jwkunz.github.io/lib_jsl/book/em-arrays/beamforming.html).

use super::Coherency;
use crate::arrays::SensorArray;
use crate::conventions::C64;
use crate::linalg::hermitian_transpose;
use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;
use std::f64::consts::PI;

/// A distant source in direction `(θ, φ)` delivering power flux density `S` (W/m², integrated
/// over the scene bandwidth, both polarizations) with coherency `J` (unit trace).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointSource {
    pub theta: f64,
    pub phi: f64,
    pub flux_density: f64,
    pub polarization: Coherency,
}

impl PointSource {
    pub fn new(theta: f64, phi: f64, flux_density: f64, polarization: Coherency) -> Result<Self, ErrorsJSL> {
        if !flux_density.is_finite() || flux_density < 0.0 || !theta.is_finite() || !phi.is_finite() {
            return Err(ErrorsJSL::InvalidInputRange("Need a finite direction and a non-negative flux density."));
        }
        Ok(PointSource { theta, phi, flux_density, polarization })
    }

    /// The source's contribution `R = S λ²/(4π) · A(Ω) J A(Ω)ᴴ` to the array covariance (W):
    /// a co-polarized element of gain `G` receives the available power `S · λ² G / (4π)`.
    pub fn covariance(&self, array: &SensorArray) -> Result<Array2<C64>, ErrorsJSL> {
        let a = array.response_matrix(self.theta, self.phi)?;
        let j = Array2::from_shape_fn((2, 2), |(r, c)| self.polarization.0[r][c]);
        let scale = self.flux_density * array.wavelength().powi(2) / (4.0 * PI);
        Ok(a.dot(&j).dot(&hermitian_transpose(&a)).mapv(|v| v * scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::antenna_models::Isotropic;
    use crate::conventions::TimeConvention;
    use crate::coordinates::transforms::RigidTransform;
    use lib_jsl_core::constants::em::C0;
    use std::sync::Arc;

    fn array(pol: [C64; 2]) -> SensorArray {
        let f = 1e9;
        let positions = SensorArray::linear_positions(3, 0.5 * C0 / f, [0.0, 1.0, 0.0]);
        SensorArray::from_antenna(&positions, Arc::new(Isotropic { polarization: pol }), &RigidTransform::identity(), f, TimeConvention::PlusJ, 1.0).unwrap()
    }

    #[test]
    fn received_power_and_rank_one_structure() {
        let a = array([C64::new(1.0, 0.0), C64::new(0.0, 0.0)]);
        let s = PointSource::new(1.0, 0.5, 2e-12, Coherency::theta()).unwrap();
        let r = s.covariance(&a).unwrap();
        let lambda = a.wavelength();
        let want = 2e-12 * lambda * lambda / (4.0 * PI); // isotropic gain 1
        for i in 0..3 {
            assert!((r[[i, i]].re - want).abs() < 1e-9 * want);
            for j in 0..3 {
                assert!((r[[i, j]].norm() - want).abs() < 1e-9 * want); // fully coherent
            }
        }
        // Polarization: cross-polarized source is invisible; unpolarized gives half the power.
        assert!(PointSource::new(1.0, 0.5, 2e-12, Coherency::phi()).unwrap().covariance(&a).unwrap().iter().all(|v| v.norm() < 1e-30));
        assert!((PointSource::new(1.0, 0.5, 2e-12, Coherency::unpolarized()).unwrap().covariance(&a).unwrap()[[0, 0]].re - want / 2.0).abs() < 1e-9 * want);
        assert!(PointSource::new(1.0, 0.5, -1.0, Coherency::theta()).is_err());
    }

    #[test]
    fn circularly_polarized_elements_match_the_same_handedness_only() {
        let conv = TimeConvention::PlusJ;
        // An element whose transmitted pattern is RHCP about r̂: (θ̂ - jφ̂)/√2.
        let rhcp_element = array([C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0), -conv.unit() * std::f64::consts::FRAC_1_SQRT_2]);
        let co = PointSource::new(0.7, 0.2, 1.0, Coherency::rhcp_incoming(conv)).unwrap().covariance(&rhcp_element).unwrap();
        let cross = PointSource::new(0.7, 0.2, 1.0, Coherency::lhcp_incoming(conv)).unwrap().covariance(&rhcp_element).unwrap();
        assert!(co[[0, 0]].re > 0.0 && cross[[0, 0]].norm() < 1e-12 * co[[0, 0]].re);
    }
}
