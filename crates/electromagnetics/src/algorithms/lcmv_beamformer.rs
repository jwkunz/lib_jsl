//! Linearly constrained minimum variance (LCMV) and MVDR beamformers.
//!
//! See the book entry [Arrays, scenes and beamformers](https://jwkunz.github.io/lib_jsl/book/em-arrays/beamforming.html).

use crate::conventions::C64;
use crate::linalg::{check_hermitian, hermitian_transpose};
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array1, Array2};
use lib_jsl_linalg::{Factorize, Solve};

/// The weights minimising the output power `wᴴ R w` subject to the linear constraints
/// `wᴴ c_k = f_k`: the beamformer's response to each constraint vector `c_k` (a column of `C`,
/// e.g. a steering vector) is the prescribed `f_k` (1 for a direction to pass undistorted, 0 for
/// one to null). Equivalently `Cᴴ w = f*`, giving
///
/// `w = R⁻¹ C (Cᴴ R⁻¹ C)⁻¹ f*`.
///
/// `R` must be Hermitian positive definite and the constraints linearly independent (at most `N`).
pub fn lcmv_beamformer(covariance: &Array2<C64>, constraints: &Array2<C64>, response: &Array1<C64>) -> Result<Array1<C64>, ErrorsJSL> {
    let n = covariance.nrows();
    check_hermitian(covariance, n)?;
    let m = constraints.ncols();
    if constraints.nrows() != n || m == 0 || m > n || response.len() != m {
        return Err(ErrorsJSL::InvalidInputRange("Need an N×M constraint matrix with 1 <= M <= N and M responses."));
    }
    // ANCHOR: lcmv
    let lu = covariance.factorize().map_err(|_| ErrorsJSL::InvalidInputRange("The covariance is singular."))?;
    let mut rinv_c = Array2::<C64>::zeros((n, m));
    for k in 0..m {
        let col = lu.solve(&constraints.column(k).to_owned()).map_err(|_| ErrorsJSL::InvalidInputRange("The covariance is singular."))?;
        rinv_c.column_mut(k).assign(&col);
    }
    let gram = hermitian_transpose(constraints).dot(&rinv_c);
    let g = gram.solve(&response.mapv(|v| v.conj())).map_err(|_| ErrorsJSL::InvalidInputRange("The constraints are linearly dependent."))?;
    Ok(rinv_c.dot(&g))
    // ANCHOR_END: lcmv
}

/// The minimum-variance distortionless response beamformer: LCMV with the single constraint
/// `wᴴ a = 1`, i.e. `w = R⁻¹ a / (aᴴ R⁻¹ a)`. Its output power is `1 / (aᴴ R⁻¹ a)` (the Capon
/// spectrum).
pub fn mvdr_beamformer(covariance: &Array2<C64>, steering: &Array1<C64>) -> Result<Array1<C64>, ErrorsJSL> {
    let c = steering.clone().insert_axis(ndarray::Axis(1));
    lcmv_beamformer(covariance, &c, &Array1::from_elem(1, C64::new(1.0, 0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::beamforming::{output_power, response, sinr};
    use crate::algorithms::max_snr_beamformer::max_snr_beamformer;
    use crate::antenna_models::Isotropic;
    use crate::arrays::SensorArray;
    use crate::conventions::TimeConvention;
    use crate::coordinates::transforms::RigidTransform;
    use crate::linalg::quadratic_form;
    use crate::signal_models::{Coherency, PointSource, Scene};
    use lib_jsl_core::constants::em::C0;
    use std::sync::Arc;

    const POL: [C64; 2] = [C64 { re: 1.0, im: 0.0 }, C64 { re: 0.0, im: 0.0 }];

    fn array() -> SensorArray {
        let f = 1e9;
        SensorArray::from_antenna(&SensorArray::linear_positions(10, 0.5 * C0 / f, [1.0, 0.0, 0.0]), Arc::new(Isotropic::theta_polarized()), &RigidTransform::identity(), f, TimeConvention::PlusJ, 1.0).unwrap()
    }

    #[test]
    fn mvdr_is_distortionless_and_nulls_a_jammer() {
        let a = array();
        // A jammer about 40 dB above the receiver noise.
        let scene = Scene::new(1e6).unwrap().with_point_source(PointSource::new(0.8, 0.0, 1e-8, Coherency::theta()).unwrap()).with_receiver_temperature(300.0);
        let r = scene.covariance(&a).unwrap();
        let look = a.steering_vector(1.5, 0.0, POL).unwrap();
        let w = mvdr_beamformer(&r, &look).unwrap();
        assert!((response(&w, &look).unwrap() - C64::new(1.0, 0.0)).norm() < 1e-9);
        let jam = a.steering_vector(0.8, 0.0, POL).unwrap();
        assert!(response(&w, &jam).unwrap().norm() < 1e-4, "{}", response(&w, &jam).unwrap().norm());
        // Output power equals the Capon value 1 / (aᴴ R⁻¹ a), and no distortionless weights do better.
        let rinv_a = r.solve(&look).unwrap();
        let capon = 1.0 / look.iter().zip(rinv_a.iter()).map(|(x, y)| (x.conj() * y).re).sum::<f64>();
        assert!((output_power(&w, &r).unwrap() - capon).abs() < 1e-9 * capon);
        let naive = look.mapv(|v| v / 10.0); // the conventional beamformer, also distortionless
        assert!(quadratic_form(&naive, &r) > 10.0 * capon);
    }

    #[test]
    fn lcmv_honours_every_constraint_and_matches_max_snr_for_one_constraint() {
        let a = array();
        let scene = Scene::new(1e6).unwrap().with_point_source(PointSource::new(1.0, 0.0, 1e-13, Coherency::theta()).unwrap()).with_receiver_temperature(300.0);
        let r = scene.covariance(&a).unwrap();
        let dirs = [1.5, 2.0, 0.6];
        let c = Array2::from_shape_fn((10, 3), |(i, k)| a.steering_vector(dirs[k], 0.0, POL).unwrap()[i]);
        let f = Array1::from(vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.5, -0.5)]);
        let w = lcmv_beamformer(&r, &c, &f).unwrap();
        for k in 0..3 {
            assert!((response(&w, &c.column(k).to_owned()).unwrap() - f[k]).norm() < 1e-9, "constraint {k}");
        }
        // MVDR on the interference-plus-noise covariance maximises SINR for a point target.
        let target = a.steering_vector(1.5, 0.0, POL).unwrap();
        let rs = PointSource::new(1.5, 0.0, 1e-16, Coherency::theta()).unwrap().covariance(&a).unwrap();
        let w_mvdr = mvdr_beamformer(&r, &target).unwrap();
        let (_, best) = max_snr_beamformer(&rs, &r).unwrap();
        assert!((sinr(&w_mvdr, &rs, &r).unwrap() - best).abs() < 1e-6 * best);
        // Validation.
        assert!(lcmv_beamformer(&r, &Array2::zeros((10, 0)), &Array1::zeros(0)).is_err());
        assert!(lcmv_beamformer(&r, &c, &Array1::zeros(2)).is_err());
        let dup = Array2::from_shape_fn((10, 2), |(i, _)| target[i]);
        assert!(lcmv_beamformer(&r, &dup, &Array1::from_elem(2, C64::new(1.0, 0.0))).is_err());
    }
}
