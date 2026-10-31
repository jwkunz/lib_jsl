//! Coordinate and basis conversion of whole fields, and rigid transforms of fields.
//!
//! - [`HarmonicField::to_basis`] / [`TimeField::to_basis`] re-express vector components in another
//!   polarization basis at every sample (positions are unchanged; they live on the domain).
//! - [`Domain::positions_in`] lists the sample positions in any coordinate system.
//! - [`HarmonicField::transformed`] applies a [`RigidTransform`] `x ↦ R x + t` to the field: the new
//!   value at a sample `q` is `R · F(T⁻¹ q)` (just `F(T⁻¹ q)` for scalars), obtained by
//!   interpolation. On the far-field sphere, positions are directions: a rotation maps directions,
//!   and a translation by `t` multiplies the pattern by `e^{jk r̂·t}` (the same expression in both
//!   time conventions, with the convention's imaginary unit).

use crate::conventions::{C64, Medium};
use crate::coordinates::polarization::PolarizationBasis;
use crate::coordinates::systems::{CoordinateSystem, Position};
use crate::coordinates::transforms::RigidTransform;
use crate::fields::domain::Domain;
use crate::fields::field_container::{HarmonicField, Rank, TimeField};
use crate::fields::interpolation::{Extrapolation, InterpolationMethod};
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array2, Array3};

impl Domain {
    /// Every sample position expressed in `system`, in flat order.
    pub fn positions_in(&self, system: CoordinateSystem) -> Vec<Position> {
        (0..self.len()).map(|i| self.position_flat(i).in_system(system)).collect()
    }
}

impl HarmonicField {
    /// The same vector field with components in `basis`.
    pub fn to_basis(&self, basis: PolarizationBasis) -> Result<HarmonicField, ErrorsJSL> {
        let Rank::Vector(from) = self.rank() else {
            return Err(ErrorsJSL::InvalidInputRange("Scalar fields have no polarization basis."));
        };
        let mut values = Array2::<C64>::zeros((self.len(), 3));
        for i in 0..self.len() {
            let v = from.convert(self.vector(i), basis, &self.domain().position_flat(i), self.convention());
            for k in 0..3 {
                values[[i, k]] = v[k];
            }
        }
        HarmonicField::new(self.domain().clone(), self.quantity(), Rank::Vector(basis), self.frequency(), self.convention(), values)
    }

    /// Applies a rigid transform in vacuum; see [`HarmonicField::transformed_in`].
    pub fn transformed(&self, transform: &RigidTransform, method: InterpolationMethod, outside: Extrapolation) -> Result<HarmonicField, ErrorsJSL> {
        self.transformed_in(transform, &Medium::VACUUM, method, outside)
    }

    /// Applies a rigid transform to the field, keeping its domain and basis. `medium` sets the
    /// wavenumber of the far-field translation phase.
    pub fn transformed_in(&self, transform: &RigidTransform, medium: &Medium, method: InterpolationMethod, outside: Extrapolation) -> Result<HarmonicField, ErrorsJSL> {
        let inverse = transform.inverse();
        let domain = self.domain().clone();
        let far = domain.is_far_field();
        let k = medium.wavenumber(self.frequency(), self.convention());
        let c = self.rank().components();
        let mut values = Array2::<C64>::zeros((self.len(), c));
        for i in 0..domain.len() {
            let q = domain.position_flat(i);
            // Where the value at q came from.
            let source = if far {
                let dir = inverse.rotation.rotate(q.to_cartesian());
                Position::cartesian(dir[0], dir[1], dir[2])
            } else {
                inverse.apply_position(&q)
            };
            let sampled: Result<Vec<C64>, ErrorsJSL> = match self.rank() {
                Rank::Scalar => self.sample_at(&source, method),
                // Rotate the Cartesian vector, then express it in the basis at q.
                Rank::Vector(basis) => self
                    .sample_cartesian_at(&source, method)
                    .map(|cart| basis.from_cartesian(transform.apply_complex_vector(cart), &q, self.convention()).to_vec()),
            };
            let row = match sampled {
                Ok(v) => v,
                Err(e) if outside == Extrapolation::Error => return Err(e),
                Err(_) => vec![C64::new(0.0, 0.0); c],
            };
            let phase = if far {
                let d = q.to_cartesian();
                let t = transform.translation;
                (self.convention().unit() * k * (d[0] * t[0] + d[1] * t[1] + d[2] * t[2])).exp()
            } else {
                C64::new(1.0, 0.0)
            };
            for kk in 0..c {
                values[[i, kk]] = row[kk] * phase;
            }
        }
        HarmonicField::new(domain, self.quantity(), self.rank(), self.frequency(), self.convention(), values)
    }

    /// The Cartesian components of a vector field at an arbitrary point.
    pub fn sample_cartesian_at(&self, p: &Position, method: InterpolationMethod) -> Result<[C64; 3], ErrorsJSL> {
        let Rank::Vector(basis) = self.rank() else {
            return Err(ErrorsJSL::InvalidInputRange("Scalar fields have no Cartesian components."));
        };
        let v = self.sample_at(p, method)?;
        Ok(basis.to_cartesian([v[0], v[1], v[2]], p, self.convention()))
    }
}

impl TimeField {
    /// The same vector field with components in another linear basis.
    pub fn to_basis(&self, basis: PolarizationBasis) -> Result<TimeField, ErrorsJSL> {
        let Rank::Vector(from) = self.rank() else {
            return Err(ErrorsJSL::InvalidInputRange("Scalar fields have no polarization basis."));
        };
        let (steps, n, _) = self.values().dim();
        let mut values = Array3::<f64>::zeros((steps, n, 3));
        for i in 0..n {
            let p = self.domain().position_flat(i);
            for t in 0..steps {
                let v = from.convert_real([self.values()[[t, i, 0]], self.values()[[t, i, 1]], self.values()[[t, i, 2]]], basis, &p)?;
                for k in 0..3 {
                    values[[t, i, k]] = v[k];
                }
            }
        }
        TimeField::new(self.domain().clone(), self.quantity(), Rank::Vector(basis), self.time_axis(), values)
    }

    /// Applies a rigid transform (resampling through `T⁻¹`). Far-field time series can only be
    /// rotated: a translation would need a time delay rather than a phase.
    pub fn transformed(&self, transform: &RigidTransform, method: InterpolationMethod, outside: Extrapolation) -> Result<TimeField, ErrorsJSL> {
        let domain = self.domain().clone();
        let far = domain.is_far_field();
        if far && transform.translation.iter().any(|v| *v != 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("Translating a time-domain far field needs a delay, not supported."));
        }
        let inverse = transform.inverse();
        let (steps, n, c) = self.values().dim();
        let mut values = Array3::<f64>::zeros((steps, n, c));
        for i in 0..n {
            let q = domain.position_flat(i);
            let source = if far {
                let d = inverse.rotation.rotate(q.to_cartesian());
                Position::cartesian(d[0], d[1], d[2])
            } else {
                inverse.apply_position(&q)
            };
            let sampled = match self.sample_at(&source, method) {
                Ok(s) => s,
                Err(e) if outside == Extrapolation::Error => return Err(e),
                Err(_) => continue,
            };
            for (t, v) in sampled.into_iter().enumerate() {
                let out = match self.rank() {
                    Rank::Scalar => v,
                    Rank::Vector(basis) => {
                        let cart = basis.convert_real([v[0], v[1], v[2]], PolarizationBasis::CARTESIAN, &source)?;
                        PolarizationBasis::CARTESIAN.convert_real(transform.apply_vector(cart), basis, &q)?.to_vec()
                    }
                };
                for k in 0..c {
                    values[[t, i, k]] = out[k];
                }
            }
        }
        TimeField::new(domain, self.quantity(), self.rank(), self.time_axis(), values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conventions::TimeConvention;
    use crate::fields::domain::Axis;
    use crate::fields::field_container::{FieldQuantity, TimeAxis};
    use lib_jsl_core::constants::em::C0;
    use std::f64::consts::{FRAC_PI_2, PI};
    use std::sync::Arc;

    fn c(re: f64) -> C64 {
        C64::new(re, 0.0)
    }

    fn dipole_pattern(d: Arc<Domain>) -> HarmonicField {
        // A z-directed short dipole: E = sin θ θ̂.
        HarmonicField::vector_from_fn(d, FieldQuantity::ElectricField, PolarizationBasis::SPHERICAL, 1e9, TimeConvention::PlusJ, |p| [c(0.0), c(p.coords[1].sin()), c(0.0)]).unwrap()
    }

    #[test]
    fn basis_conversion_of_whole_fields_round_trips() {
        let d = Arc::new(Domain::far_field_gauss(6, 12).unwrap());
        let e = dipole_pattern(d.clone());
        for basis in [PolarizationBasis::CARTESIAN, PolarizationBasis::CYLINDRICAL, PolarizationBasis::Circular(CoordinateSystem::Spherical)] {
            let there = e.to_basis(basis).unwrap();
            assert_eq!(there.basis(), Some(basis));
            let back = there.to_basis(PolarizationBasis::SPHERICAL).unwrap();
            assert!(back.values().iter().zip(e.values().iter()).all(|(a, b)| (a - b).norm() < 1e-12), "{basis:?}");
            assert!((there.integrate_norm_squared() - e.integrate_norm_squared()).abs() < 1e-12);
        }
        // A θ̂-polarized pattern has equal right- and left-hand circular parts.
        let circ = e.to_basis(PolarizationBasis::Circular(CoordinateSystem::Spherical)).unwrap();
        assert!((0..circ.len()).all(|i| (circ.vector(i)[0].norm() - circ.vector(i)[1].norm()).abs() < 1e-12));
        let s = HarmonicField::zeros(d.clone(), FieldQuantity::Generic, Rank::Scalar, 1.0, TimeConvention::PlusJ).unwrap();
        assert!(s.to_basis(PolarizationBasis::CARTESIAN).is_err());
        assert_eq!(d.positions_in(CoordinateSystem::Cartesian).len(), d.len());
    }

    #[test]
    fn rotating_a_far_field_turns_a_z_dipole_into_an_x_dipole() {
        let d = Arc::new(Domain::far_field_uniform(37, 72).unwrap());
        let e = dipole_pattern(d.clone());
        let r = RigidTransform::rotation_about([0.0, 1.0, 0.0], FRAC_PI_2); // ẑ → x̂
        let x_dipole = e.transformed(&r, InterpolationMethod::Cubic, Extrapolation::Error).unwrap();
        for i in (0..d.len()).step_by(37) {
            let dir = d.position_flat(i).to_cartesian();
            let want = (1.0 - dir[0] * dir[0]).max(0.0).sqrt(); // |sin| of the angle from x̂
            let got = x_dipole.norm_squared()[i].sqrt();
            assert!((got - want).abs() < 2e-3, "{dir:?}: {got} vs {want}");
        }
        assert!((x_dipole.integrate_norm_squared() - e.integrate_norm_squared()).abs() < 1e-3 * e.integrate_norm_squared());
    }

    #[test]
    fn translating_a_far_field_applies_the_phase_ramp_in_either_convention() {
        let d = Arc::new(Domain::far_field_uniform(19, 36).unwrap());
        let e = dipole_pattern(d.clone());
        let lambda = C0 / 1e9;
        let t = RigidTransform::translation([lambda / 4.0, 0.0, 0.0]);
        let moved = e.transformed(&t, InterpolationMethod::Linear, Extrapolation::Error).unwrap();
        let along_x = d.flatten(&[9, 0]); // θ = 90°, φ = 0: r̂ = x̂
        let ratio = moved.vector(along_x)[1] / e.vector(along_x)[1];
        assert!((ratio - C64::new(0.0, 1.0)).norm() < 1e-9, "{ratio}"); // e^{+jπ/2}
        assert!((moved.integrate_norm_squared() - e.integrate_norm_squared()).abs() < 1e-9);
        // In the e^{-iωt} convention the same physical shift is the conjugate phase.
        let other = e.with_convention(TimeConvention::MinusI).transformed(&t, InterpolationMethod::Linear, Extrapolation::Error).unwrap();
        assert!((other.with_convention(TimeConvention::PlusJ).vector(along_x)[1] - moved.vector(along_x)[1]).norm() < 1e-9);
        let _ = PI;
    }

    #[test]
    fn transforming_near_fields_resamples_through_the_inverse() {
        let axis = || Axis::uniform(-1.0, 1.0, 9).unwrap();
        let d = Arc::new(Domain::cartesian_box(axis(), axis(), axis()).unwrap());
        // F = (x, 0, 0): translated by (0.25, 0, 0) it becomes (x - 0.25, 0, 0).
        let f = HarmonicField::vector_from_fn(d.clone(), FieldQuantity::ElectricField, PolarizationBasis::CARTESIAN, 1.0, TimeConvention::PlusJ, |p| [c(p.coords[0]), c(0.0), c(0.0)]).unwrap();
        let moved = f.transformed(&RigidTransform::translation([0.25, 0.0, 0.0]), InterpolationMethod::Linear, Extrapolation::Zero).unwrap();
        let i = d.flatten(&[6, 4, 4]); // x = 0.5
        assert!((moved.vector(i)[0] - c(0.25)).norm() < 1e-12);
        assert_eq!(moved.vector(d.flatten(&[0, 4, 4]))[0], c(0.0)); // came from outside: zero
        assert!(f.transformed(&RigidTransform::translation([0.25, 0.0, 0.0]), InterpolationMethod::Linear, Extrapolation::Error).is_err());
        // A uniform ẑ field rotated 90° about x̂ becomes -ŷ everywhere it is defined.
        let z = HarmonicField::vector_from_fn(d.clone(), FieldQuantity::ElectricField, PolarizationBasis::CARTESIAN, 1.0, TimeConvention::PlusJ, |_| [c(0.0), c(0.0), c(1.0)]).unwrap();
        let rz = z.transformed(&RigidTransform::rotation_about([1.0, 0.0, 0.0], FRAC_PI_2), InterpolationMethod::Linear, Extrapolation::Error).unwrap();
        assert!((rz.vector(d.flatten(&[3, 5, 2]))[1] + c(1.0)).norm() < 1e-12);
        // Time fields: same rules; far-field translation is refused.
        let tf = TimeField::from_fn(d.clone(), FieldQuantity::ElectricField, Rank::Vector(PolarizationBasis::CARTESIAN), TimeAxis::new(0.0, 1.0, 2).unwrap(), |_, t| vec![0.0, 0.0, 1.0 + t]).unwrap();
        let rt = tf.transformed(&RigidTransform::rotation_about([1.0, 0.0, 0.0], FRAC_PI_2), InterpolationMethod::Linear, Extrapolation::Error).unwrap();
        assert!((rt.at(1, d.flatten(&[4, 4, 4]))[1] + 2.0).abs() < 1e-12);
        assert!(rt.to_basis(PolarizationBasis::SPHERICAL).is_ok());
        let ff = Arc::new(Domain::far_field_gauss(4, 8).unwrap());
        let tff = TimeField::from_fn(ff, FieldQuantity::Generic, Rank::Scalar, TimeAxis::new(0.0, 1.0, 1).unwrap(), |_, _| vec![1.0]).unwrap();
        assert!(tff.transformed(&RigidTransform::translation([1.0, 0.0, 0.0]), InterpolationMethod::Linear, Extrapolation::Error).is_err());
    }
}
