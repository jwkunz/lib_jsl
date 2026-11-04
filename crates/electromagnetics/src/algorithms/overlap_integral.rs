//! The overlap (coupling) integral between two field distributions.
//!
//! See the book entry [Fields on grids](https://jwkunz.github.io/lib_jsl/book/em-arrays/fields.html).

use crate::conventions::C64;
use crate::fields::field_container::HarmonicField;
use crate::fields::interpolation::{Extrapolation, InterpolationMethod};
use lib_jsl_core::ErrorsJSL;
use std::sync::Arc;

/// The result of an overlap integral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Overlap {
    /// `∫ a·b* dA / √(∫|a|² dA ∫|b|² dA)`: magnitude ≤ 1, phase is the relative phase.
    pub coefficient: C64,
    /// `|coefficient|²`, the power coupling efficiency between the two distributions.
    pub efficiency: f64,
}

/// The overlap of fields `a` and `b`: `|∫ a·b* dA|² / (∫|a|² dA ∫|b|² dA)`.
///
/// The integral is evaluated on `a`'s domain with that domain's quadrature weights (so the
/// Jacobian of every coordinate system is handled), in Cartesian components (so `a` and `b` may
/// use different polarization bases). If `b` lives on another domain it is first resampled onto
/// `a`'s with `method`, taking zero outside `b`'s domain (e.g. an aperture field that does not
/// cover the whole plane). `b` is converted to `a`'s time convention if needed; the fields must
/// share rank and frequency.
pub fn overlap_integral(a: &HarmonicField, b: &HarmonicField, method: InterpolationMethod) -> Result<Overlap, ErrorsJSL> {
    if a.is_scalar() != b.is_scalar() {
        return Err(ErrorsJSL::InvalidInputRange("Both fields must be scalar or both vector."));
    }
    if (a.frequency() - b.frequency()).abs() > 1e-9 * a.frequency() {
        return Err(ErrorsJSL::InvalidInputRange("The fields must be at the same frequency."));
    }
    let b = b.with_convention(a.convention());
    let b = if Arc::ptr_eq(a.domain(), b.domain()) || a.domain() == b.domain() { b } else { b.resample_to(a.domain().clone(), method, Extrapolation::Zero)? };
    let (na, nb) = (a.integrate_norm_squared(), b.integrate_norm_squared());
    if !(na > 0.0 && nb > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Both fields must be non-zero on the integration domain."));
    }
    let coefficient = a.inner_product(&b)? / (na * nb).sqrt();
    Ok(Overlap { coefficient, efficiency: coefficient.norm_sqr() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conventions::TimeConvention;
    use crate::coordinates::polarization::PolarizationBasis;
    use crate::coordinates::systems::CoordinateSystem;
    use crate::coordinates::transforms::RigidTransform;
    use crate::fields::domain::{Axis, Domain};
    use crate::fields::field_container::{FieldQuantity, Rank};
    use std::f64::consts::{FRAC_1_SQRT_2, PI, TAU};

    fn gaussian(d: Arc<Domain>, waist: f64, offset: f64, pol: [f64; 3]) -> HarmonicField {
        HarmonicField::vector_from_fn(d, FieldQuantity::ElectricField, PolarizationBasis::CARTESIAN, 1e9, TimeConvention::PlusJ, |p| {
            let [x, y, _] = p.to_cartesian();
            let g = (-((x - offset).powi(2) + y * y) / (waist * waist)).exp();
            pol.map(|v| C64::new(v * g, 0.0))
        })
        .unwrap()
    }

    fn plane(n: usize, half: f64) -> Arc<Domain> {
        Arc::new(Domain::plane(Axis::gauss_legendre(-half, half, n).unwrap(), Axis::gauss_legendre(-half, half, n).unwrap(), 0.0).unwrap())
    }

    #[test]
    fn gaussian_beam_coupling_matches_the_closed_form() {
        let d = plane(60, 6.0);
        let x = [1.0, 0.0, 0.0];
        let a = gaussian(d.clone(), 1.0, 0.0, x);
        assert!((overlap_integral(&a, &a, InterpolationMethod::Linear).unwrap().efficiency - 1.0).abs() < 1e-12);
        // Waist mismatch: η = (2 w₁ w₂ / (w₁² + w₂²))².
        let b = gaussian(d.clone(), 1.5, 0.0, x);
        let want = (2.0 * 1.0 * 1.5 / (1.0 + 2.25f64)).powi(2);
        assert!((overlap_integral(&a, &b, InterpolationMethod::Linear).unwrap().efficiency - want).abs() < 1e-10);
        // Lateral offset with equal waists: η = exp(-d²/w²).
        let c = gaussian(d.clone(), 1.0, 0.8, x);
        assert!((overlap_integral(&a, &c, InterpolationMethod::Linear).unwrap().efficiency - (-0.64f64).exp()).abs() < 1e-10);
        // Polarization mismatch: orthogonal → 0, 45° → ½.
        let ortho = gaussian(d.clone(), 1.0, 0.0, [0.0, 1.0, 0.0]);
        assert!(overlap_integral(&a, &ortho, InterpolationMethod::Linear).unwrap().efficiency < 1e-20);
        let tilted = gaussian(d.clone(), 1.0, 0.0, [FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0]);
        assert!((overlap_integral(&a, &tilted, InterpolationMethod::Linear).unwrap().efficiency - 0.5).abs() < 1e-12);
        // A phase difference shows up in the coefficient, not the efficiency.
        let shifted = a.scaled(C64::from_polar(1.0, 0.7));
        let o = overlap_integral(&a, &shifted, InterpolationMethod::Linear).unwrap();
        assert!((o.coefficient - C64::from_polar(1.0, -0.7)).norm() < 1e-12 && (o.efficiency - 1.0).abs() < 1e-12);
    }

    #[test]
    fn different_grids_bases_and_coordinate_systems() {
        // The same beam sampled on a coarse Cartesian grid and on a fine polar-like grid in a
        // cylindrical-basis representation still overlaps fully.
        let coarse = plane(40, 6.0);
        let fine = plane(90, 7.0);
        let a = gaussian(coarse, 1.2, 0.0, [1.0, 0.0, 0.0]);
        let b = gaussian(fine, 1.2, 0.0, [1.0, 0.0, 0.0]).to_basis(PolarizationBasis::CYLINDRICAL).unwrap();
        let o = overlap_integral(&a, &b, InterpolationMethod::Cubic).unwrap();
        assert!((o.efficiency - 1.0).abs() < 1e-6, "{}", o.efficiency);
        // On a sphere the sin θ Jacobian is in the weights: a dipole pattern overlaps itself fully
        // and a rotated copy only partially.
        let s = Arc::new(Domain::far_field_gauss(24, 48).unwrap());
        let dip = HarmonicField::vector_from_fn(s.clone(), FieldQuantity::ElectricField, PolarizationBasis::SPHERICAL, 1e9, TimeConvention::PlusJ, |p| [C64::new(0.0, 0.0), C64::new(p.coords[1].sin(), 0.0), C64::new(0.0, 0.0)]).unwrap();
        let same = dip.to_basis(PolarizationBasis::Circular(CoordinateSystem::Spherical)).unwrap();
        assert!((overlap_integral(&dip, &same, InterpolationMethod::Linear).unwrap().efficiency - 1.0).abs() < 1e-12);
        let x_dip = dip.transformed(&RigidTransform::rotation_about([0.0, 1.0, 0.0], PI / 2.0), InterpolationMethod::Cubic, Extrapolation::Error).unwrap();
        // ∫ sin θ θ̂ · (x-dipole pattern) dΩ vanishes by symmetry: orthogonal patterns.
        assert!(overlap_integral(&dip, &x_dip, InterpolationMethod::Linear).unwrap().efficiency < 1e-4);
        // Rotating about the dipole's own axis changes nothing.
        let spun = dip.transformed(&RigidTransform::rotation_about([0.0, 0.0, 1.0], 1.0), InterpolationMethod::Cubic, Extrapolation::Error).unwrap();
        assert!((overlap_integral(&dip, &spun, InterpolationMethod::Linear).unwrap().efficiency - 1.0).abs() < 1e-6);
        let _ = TAU;
    }

    #[test]
    fn validation_and_conventions() {
        let d = plane(10, 3.0);
        let a = gaussian(d.clone(), 1.0, 0.0, [1.0, 0.0, 0.0]);
        // The other convention describes the same physical field: still a perfect match.
        assert!((overlap_integral(&a, &a.with_convention(TimeConvention::MinusI), InterpolationMethod::Linear).unwrap().efficiency - 1.0).abs() < 1e-12);
        let zero = a.scaled(C64::new(0.0, 0.0));
        assert!(overlap_integral(&a, &zero, InterpolationMethod::Linear).is_err());
        let s = HarmonicField::zeros(d.clone(), FieldQuantity::Generic, Rank::Scalar, 1e9, TimeConvention::PlusJ).unwrap();
        assert!(overlap_integral(&a, &s, InterpolationMethod::Linear).is_err());
        let other_freq = HarmonicField::vector_from_fn(d, FieldQuantity::ElectricField, PolarizationBasis::CARTESIAN, 2e9, TimeConvention::PlusJ, |_| [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)]).unwrap();
        assert!(overlap_integral(&a, &other_freq, InterpolationMethod::Linear).is_err());
    }
}
