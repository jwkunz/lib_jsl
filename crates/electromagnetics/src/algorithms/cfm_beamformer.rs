//! The conjugate field match (CFM) beamformer.
//!
//! See the book entry [Arrays, scenes and beamformers](https://jwkunz.github.io/lib_jsl/book/em-arrays/beamforming.html).

use super::beamforming::normalize;
use crate::arrays::SensorArray;
use crate::conventions::C64;
use crate::linalg::{check_hermitian, hermitian_eigen};
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array1, Array2};

/// The conjugate field match: the matched filter for a known harmonic array excitation `a`
/// (the response the desired field produces across the elements), `w = a / ‖a‖`. With `y = wᴴ x`
/// this conjugates the excitation's phases, adding the desired contributions coherently; it
/// maximises the output for the desired field per unit weight norm and is the max-SNR solution in
/// spatially white noise.
pub fn cfm_weights(excitation: &Array1<C64>) -> Result<Array1<C64>, ErrorsJSL> {
    normalize(excitation)
}

/// The CFM weights for a plane wave from `(θ, φ)` with Jones polarization `p`.
pub fn cfm_for_direction(array: &SensorArray, theta: f64, phi: f64, polarization: [C64; 2]) -> Result<Array1<C64>, ErrorsJSL> {
    cfm_weights(&array.steering_vector(theta, phi, polarization)?)
}

/// The CFM for a *distributed* desired field described by its covariance `R_d`: the weights
/// maximising `wᴴ R_d w / wᴴ w`, i.e. the dominant eigenvector of `R_d`, returned with that
/// maximum (the eigenvalue). For a single point source this reduces to [`cfm_weights`].
pub fn cfm_beamformer(desired_covariance: &Array2<C64>) -> Result<(Array1<C64>, f64), ErrorsJSL> {
    check_hermitian(desired_covariance, desired_covariance.nrows())?;
    let (vals, vecs) = hermitian_eigen(desired_covariance)?;
    Ok((normalize(&vecs.column(0).to_owned())?, vals[0]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::beamforming::{output_power, response};
    use crate::antenna_models::Isotropic;
    use crate::conventions::TimeConvention;
    use crate::coordinates::transforms::RigidTransform;
    use crate::signal_models::{Coherency, PointSource};
    use lib_jsl_core::constants::em::C0;
    use std::sync::Arc;

    fn array() -> SensorArray {
        let f = 1e9;
        SensorArray::from_antenna(&SensorArray::linear_positions(8, 0.5 * C0 / f, [1.0, 0.0, 0.0]), Arc::new(Isotropic::theta_polarized()), &RigidTransform::identity(), f, TimeConvention::PlusJ, 1.0).unwrap()
    }

    #[test]
    fn matched_filter_adds_the_desired_signal_coherently() {
        let a = array();
        let theta_pol = [C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
        let w = cfm_for_direction(&a, 1.2, 0.3, theta_pol).unwrap();
        let s = a.steering_vector(1.2, 0.3, theta_pol).unwrap();
        // |wᴴa| = ‖a‖ = √N for unit-gain elements: full coherent gain.
        assert!((response(&w, &s).unwrap().norm() - 8f64.sqrt()).abs() < 1e-9);
        assert!((w.iter().map(|v| v.norm_sqr()).sum::<f64>() - 1.0).abs() < 1e-12);
        // Any other unit-norm weights collect less.
        let other = cfm_for_direction(&a, 0.9, 0.3, theta_pol).unwrap();
        assert!(response(&other, &s).unwrap().norm() < response(&w, &s).unwrap().norm());
    }

    #[test]
    fn eigen_form_reduces_to_the_matched_filter_for_a_point_source() {
        let a = array();
        let src = PointSource::new(1.0, 0.0, 1e-12, Coherency::theta()).unwrap();
        let rd = src.covariance(&a).unwrap();
        let (w, lambda) = cfm_beamformer(&rd).unwrap();
        let direct = cfm_for_direction(&a, 1.0, 0.0, [C64::new(1.0, 0.0), C64::new(0.0, 0.0)]).unwrap();
        assert!(w.iter().zip(direct.iter()).all(|(x, y)| (x - y).norm() < 1e-9), "same up to the normalised phase");
        assert!((output_power(&w, &rd).unwrap() - lambda).abs() < 1e-9 * lambda);
        assert!(cfm_weights(&Array1::zeros(3)).is_err());
    }
}
