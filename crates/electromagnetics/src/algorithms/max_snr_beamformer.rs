//! The maximum-SNR beamformer.
//!
//! See the book entry [Arrays, scenes and beamformers](https://jwkunz.github.io/lib_jsl/book/em-arrays/beamforming.html).

use super::beamforming::normalize;
use crate::conventions::C64;
use crate::linalg::{check_hermitian, hermitian_eigen, hermitian_transpose};
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array1, Array2};
use lib_jsl_linalg::{Cholesky, Diag, SolveTriangular, UPLO};

/// The weights maximising `wᴴ R_s w / wᴴ R_n w`: the dominant generalised eigenvector of
/// `R_s w = λ R_n w`, returned (unit norm) with the achieved SNR `λ`.
///
/// Solved by whitening with the Cholesky factor `R_n = L Lᴴ`: the eigenvectors `v` of
/// `L⁻¹ R_s L⁻ᴴ` give `w = L⁻ᴴ v`. `R_n` must be Hermitian positive definite (include receiver
/// noise); `R_s` Hermitian positive semi-definite. For a rank-one signal this equals MVDR.
pub fn max_snr_beamformer(signal: &Array2<C64>, noise: &Array2<C64>) -> Result<(Array1<C64>, f64), ErrorsJSL> {
    let n = noise.nrows();
    check_hermitian(noise, n)?;
    check_hermitian(signal, n)?;
    let l = noise.cholesky(UPLO::Lower).map_err(|_| ErrorsJSL::InvalidInputRange("The noise covariance must be positive definite."))?;
    let left = l.solve_triangular(UPLO::Lower, Diag::NonUnit, signal).map_err(|_| ErrorsJSL::RuntimeError("Whitening failed."))?;
    let whitened = l.solve_triangular(UPLO::Lower, Diag::NonUnit, &hermitian_transpose(&left)).map_err(|_| ErrorsJSL::RuntimeError("Whitening failed."))?;
    let whitened = (&whitened + &hermitian_transpose(&whitened)).mapv(|v| v * 0.5);
    let (vals, vecs) = hermitian_eigen(&whitened)?;
    let v = vecs.column(0).to_owned();
    let w = hermitian_transpose(&l).solve_triangular(UPLO::Upper, Diag::NonUnit, &v).map_err(|_| ErrorsJSL::RuntimeError("Back-substitution failed."))?;
    Ok((normalize(&w)?, vals[0]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::beamforming::sinr;
    use crate::algorithms::cfm_beamformer::cfm_for_direction;
    use crate::antenna_models::Isotropic;
    use crate::arrays::SensorArray;
    use crate::conventions::TimeConvention;
    use crate::coordinates::transforms::RigidTransform;
    use crate::signal_models::{Coherency, PointSource, Scene};
    use lib_jsl_core::constants::em::C0;
    use std::sync::Arc;

    fn setup() -> (SensorArray, Array2<C64>, Array2<C64>, Array2<C64>) {
        let f = 1e9;
        let a = SensorArray::from_antenna(&SensorArray::linear_positions(8, 0.5 * C0 / f, [1.0, 0.0, 0.0]), Arc::new(Isotropic::theta_polarized()), &RigidTransform::identity(), f, TimeConvention::PlusJ, 1.0).unwrap();
        let bw = 1e6;
        let desired = Scene::new(bw).unwrap().with_point_source(PointSource::new(1.4, 0.0, 1e-16, Coherency::theta()).unwrap());
        // A jammer about 40 dB above the receiver noise, in a sidelobe of the look direction.
        let jammer = Scene::new(bw).unwrap().with_point_source(PointSource::new(0.9, 0.0, 1e-8, Coherency::theta()).unwrap()).with_receiver_temperature(300.0);
        let white = Scene::new(bw).unwrap().with_receiver_temperature(300.0);
        (a.clone(), desired.covariance(&a).unwrap(), jammer.covariance(&a).unwrap(), white.covariance(&a).unwrap())
    }

    #[test]
    fn solves_the_generalised_eigenproblem() {
        let (_, rs, rn, _) = setup();
        let (w, snr) = max_snr_beamformer(&rs, &rn).unwrap();
        let lhs = rs.dot(&w);
        let rhs = rn.dot(&w).mapv(|v| v * snr);
        let scale = lhs.iter().map(|v| v.norm()).fold(0.0, f64::max);
        assert!(lhs.iter().zip(rhs.iter()).all(|(a, b)| (a - b).norm() < 1e-8 * scale));
        assert!((sinr(&w, &rs, &rn).unwrap() - snr).abs() < 1e-8 * snr);
    }

    #[test]
    fn matches_cfm_in_white_noise_and_beats_it_against_a_jammer() {
        let (a, rs, rn, white) = setup();
        let cfm = cfm_for_direction(&a, 1.4, 0.0, [C64::new(1.0, 0.0), C64::new(0.0, 0.0)]).unwrap();
        let (w_white, _) = max_snr_beamformer(&rs, &white).unwrap();
        assert!(w_white.iter().zip(cfm.iter()).all(|(x, y)| (x - y).norm() < 1e-8));
        let (w, best) = max_snr_beamformer(&rs, &rn).unwrap();
        let with_cfm = sinr(&cfm, &rs, &rn).unwrap();
        assert!(best > 100.0 * with_cfm, "max-SNR {best} vs CFM {with_cfm}"); // the jammer is nulled
        assert!((sinr(&w, &rs, &rn).unwrap() - best).abs() < 1e-8 * best);
        assert!(max_snr_beamformer(&rs, &Array2::zeros((8, 8))).is_err());
    }
}
