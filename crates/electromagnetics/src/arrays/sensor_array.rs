//! Arrays of sensors, each with a position and its own (embedded) far-field pattern.
//!
//! See the book entry [Arrays, scenes and beamformers](https://jwkunz.github.io/lib_jsl/book/em-arrays/beamforming.html).

use crate::antenna_models::Antenna;
use crate::conventions::{C64, Medium, TimeConvention};
use crate::coordinates::polarization::PolarizationBasis;
use crate::coordinates::systems::Position;
use crate::coordinates::transforms::RigidTransform;
use crate::fields::domain::Domain;
use crate::fields::field_container::{FieldQuantity, HarmonicField};
use crate::fields::interpolation::InterpolationMethod;
use lib_jsl_geometry::algorithms::rotations::Rotation3;
use lib_jsl_core::ErrorsJSL;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// How an element's far-field pattern is evaluated.
#[derive(Clone)]
pub enum ElementPattern {
    /// A sampled pattern container (measured or simulated, possibly an embedded pattern that
    /// includes mutual coupling), gain-normalised (`|E_θ|² + |E_φ|² = G(Ω)`), interpolated in
    /// Cartesian components.
    Sampled(Arc<HarmonicField>),
    /// An analytic antenna model, rotated into place and scaled to its gain normalisation;
    /// evaluated exactly.
    Analytic { antenna: Arc<dyn Antenna + Send + Sync>, rotation: Rotation3, scale: f64 },
}

impl std::fmt::Debug for ElementPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElementPattern::Sampled(p) => write!(f, "Sampled({} samples)", p.len()),
            ElementPattern::Analytic { scale, .. } => write!(f, "Analytic(scale {scale})"),
        }
    }
}

/// One sensor: its phase-centre position (m) and its far-field pattern.
#[derive(Debug, Clone)]
pub struct ArrayElement {
    pub position: [f64; 3],
    pub pattern: ElementPattern,
}

/// A narrowband array of sensors at one frequency.
///
/// The response of element `n` to a plane wave arriving *from* direction `Ω` (unit vector `r̂`)
/// with Jones vector `p = [p_θ, p_φ]` in the `(θ̂, φ̂)` basis at `Ω` is
/// `a_n = (g_n(Ω) · p) e^{jk r̂·p_n}` (the convention's imaginary unit in the exponent; the dot
/// product is *not* conjugated, by reciprocity), where `g_n` is the element's gain-normalised
/// pattern. Beamformer outputs are `y = wᴴ x`, so the array pattern of weights `w` is
/// `Σ w_n* g_n(Ω) e^{jk r̂·p_n}`.
#[derive(Debug, Clone)]
pub struct SensorArray {
    elements: Vec<ArrayElement>,
    frequency: f64,
    convention: TimeConvention,
    medium: Medium,
    method: InterpolationMethod,
}

impl SensorArray {
    /// An empty array in vacuum, sampling element pattern containers with cubic interpolation.
    pub fn new(frequency: f64, convention: TimeConvention) -> Result<Self, ErrorsJSL> {
        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("The frequency must be positive."));
        }
        Ok(SensorArray { elements: Vec::new(), frequency, convention, medium: Medium::VACUUM, method: InterpolationMethod::Cubic })
    }

    pub fn with_interpolation(mut self, method: InterpolationMethod) -> Self {
        self.method = method;
        self
    }

    /// Adds an element with a sampled pattern. The pattern must be an electric far field at the
    /// array frequency; it is converted to the array's convention.
    pub fn add_sampled_element(&mut self, position: [f64; 3], pattern: Arc<HarmonicField>) -> Result<(), ErrorsJSL> {
        if !pattern.domain().is_far_field() || pattern.quantity() != FieldQuantity::ElectricField || pattern.is_scalar() {
            return Err(ErrorsJSL::InvalidInputRange("Element patterns must be electric far fields."));
        }
        if (pattern.frequency() - self.frequency).abs() > 1e-9 * self.frequency {
            return Err(ErrorsJSL::InvalidInputRange("The element pattern is at a different frequency."));
        }
        let pattern = if pattern.convention() == self.convention { pattern } else { Arc::new(pattern.with_convention(self.convention)) };
        self.elements.push(ArrayElement { position, pattern: ElementPattern::Sampled(pattern) });
        Ok(())
    }

    /// Adds an analytic antenna element rotated by `rotation`, with radiation `efficiency`.
    pub fn add_antenna_element(&mut self, position: [f64; 3], antenna: Arc<dyn Antenna + Send + Sync>, rotation: Rotation3, efficiency: f64) -> Result<(), ErrorsJSL> {
        let power = antenna.radiated_power(self.frequency);
        if power.is_nan() || power <= 0.0 || !(0.0..=1.0).contains(&efficiency) {
            return Err(ErrorsJSL::InvalidInputRange("Need a radiating antenna and an efficiency in [0, 1]."));
        }
        let eta = self.medium.impedance(self.convention).re;
        let scale = (4.0 * std::f64::consts::PI * efficiency / (2.0 * eta * power)).sqrt();
        self.elements.push(ArrayElement { position, pattern: ElementPattern::Analytic { antenna, rotation, scale } });
        Ok(())
    }

    /// An array of identical antennas at `positions`, all rotated by `orientation` (its
    /// translation is ignored), evaluated analytically.
    pub fn from_antenna(positions: &[[f64; 3]], antenna: Arc<dyn Antenna + Send + Sync>, orientation: &RigidTransform, frequency: f64, convention: TimeConvention, efficiency: f64) -> Result<Self, ErrorsJSL> {
        let mut array = SensorArray::new(frequency, convention)?;
        for &p in positions {
            array.add_antenna_element(p, antenna.clone(), orientation.rotation, efficiency)?;
        }
        Ok(array)
    }

    /// An array of elements sharing one sampled pattern.
    pub fn from_pattern(positions: &[[f64; 3]], pattern: Arc<HarmonicField>) -> Result<Self, ErrorsJSL> {
        let mut array = SensorArray::new(pattern.frequency(), pattern.convention())?;
        for &p in positions {
            array.add_sampled_element(p, pattern.clone())?;
        }
        Ok(array)
    }

    /// `n` elements along `axis` (normalised) with spacing `spacing`, centred on the origin.
    pub fn linear_positions(n: usize, spacing: f64, axis: [f64; 3]) -> Vec<[f64; 3]> {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt().max(1e-300);
        let u = axis.map(|v| v / norm);
        (0..n).map(|i| u.map(|c| c * spacing * (i as f64 - (n as f64 - 1.0) / 2.0))).collect()
    }

    /// An `nx × ny` rectangular grid in the `xy` plane, centred on the origin.
    pub fn planar_positions(nx: usize, ny: usize, dx: f64, dy: f64) -> Vec<[f64; 3]> {
        let mut out = Vec::with_capacity(nx * ny);
        for i in 0..nx {
            for j in 0..ny {
                out.push([dx * (i as f64 - (nx as f64 - 1.0) / 2.0), dy * (j as f64 - (ny as f64 - 1.0) / 2.0), 0.0]);
            }
        }
        out
    }

    pub fn elements(&self) -> &[ArrayElement] {
        &self.elements
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    pub fn convention(&self) -> TimeConvention {
        self.convention
    }

    /// The free-space wavelength.
    pub fn wavelength(&self) -> f64 {
        self.medium.wavelength(self.frequency)
    }

    /// The (real) wavenumber.
    pub fn wavenumber(&self) -> f64 {
        self.medium.wavenumber(self.frequency, self.convention).re
    }

    /// Element `n`'s `[g_θ, g_φ]` including the geometric phase, for the direction `(θ, φ)`.
    pub fn element_response(&self, n: usize, theta: f64, phi: f64) -> Result<[C64; 2], ErrorsJSL> {
        let el = &self.elements[n];
        let dir = Position::spherical(1.0, theta, phi);
        let g: [C64; 2] = match &el.pattern {
            ElementPattern::Sampled(pattern) => {
                let cart = pattern.sample_cartesian_at(&dir, self.method)?;
                let v = PolarizationBasis::SPHERICAL.from_cartesian(cart, &dir, self.convention);
                [v[1], v[2]]
            }
            ElementPattern::Analytic { antenna, rotation, scale } => {
                if *rotation == Rotation3::identity() {
                    antenna.far_field(theta, phi, self.frequency, self.convention).map(|v| v * *scale)
                } else {
                    // Evaluate in the element's own frame, then rotate the field vector back.
                    let local = rotation.inverse().rotate(dir.to_cartesian());
                    let lp = Position::cartesian(local[0], local[1], local[2]).in_system(crate::coordinates::systems::CoordinateSystem::Spherical);
                    let [gt, gp] = antenna.far_field(lp.coords[1], lp.coords[2], self.frequency, self.convention);
                    let lv = PolarizationBasis::SPHERICAL.to_cartesian([C64::new(0.0, 0.0), gt * *scale, gp * *scale], &lp, self.convention);
                    let world = RigidTransform::rotation(*rotation).apply_complex_vector(lv);
                    let v = PolarizationBasis::SPHERICAL.from_cartesian(world, &dir, self.convention);
                    [v[1], v[2]]
                }
            }
        };
        // ANCHOR: geometric_phase
        let r = dir.to_cartesian();
        let p = el.position;
        let phase = (self.convention.unit() * self.wavenumber() * (r[0] * p[0] + r[1] * p[1] + r[2] * p[2])).exp();
        Ok([g[0] * phase, g[1] * phase])
        // ANCHOR_END: geometric_phase
    }

    /// The `N × 2` response matrix `A(Ω)` whose rows are the element responses.
    pub fn response_matrix(&self, theta: f64, phi: f64) -> Result<Array2<C64>, ErrorsJSL> {
        let mut a = Array2::zeros((self.len(), 2));
        for n in 0..self.len() {
            let r = self.element_response(n, theta, phi)?;
            a[[n, 0]] = r[0];
            a[[n, 1]] = r[1];
        }
        Ok(a)
    }

    /// The steering vector `A(Ω) p` for a wave of Jones polarization `p`.
    pub fn steering_vector(&self, theta: f64, phi: f64, polarization: [C64; 2]) -> Result<Array1<C64>, ErrorsJSL> {
        let a = self.response_matrix(theta, phi)?;
        Ok(Array1::from_shape_fn(self.len(), |n| a[[n, 0]] * polarization[0] + a[[n, 1]] * polarization[1]))
    }

    fn check_weights(&self, weights: &Array1<C64>) -> Result<(), ErrorsJSL> {
        if weights.len() != self.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((weights.len(), self.len())));
        }
        Ok(())
    }

    /// The array pattern `Σ w_n* g_n(Ω) e^{jk r̂·p_n}` as `[E_θ, E_φ]` in one direction.
    pub fn pattern_at(&self, weights: &Array1<C64>, theta: f64, phi: f64) -> Result<[C64; 2], ErrorsJSL> {
        self.check_weights(weights)?;
        let a = self.response_matrix(theta, phi)?;
        let mut out = [C64::new(0.0, 0.0); 2];
        for n in 0..self.len() {
            out[0] += weights[n].conj() * a[[n, 0]];
            out[1] += weights[n].conj() * a[[n, 1]];
        }
        Ok(out)
    }

    /// The array pattern of `weights` sampled on a far-field domain (electric far field,
    /// spherical basis), ready for the directivity and overlap tools.
    pub fn array_pattern(&self, weights: &Array1<C64>, domain: Arc<Domain>) -> Result<HarmonicField, ErrorsJSL> {
        self.check_weights(weights)?;
        let mut values = Array2::<C64>::zeros((domain.len(), 3));
        for i in 0..domain.len() {
            let p = domain.position_flat(i);
            let [et, ep] = self.pattern_at(weights, p.coords[1], p.coords[2])?;
            values[[i, 1]] = et;
            values[[i, 2]] = ep;
        }
        HarmonicField::new(domain, FieldQuantity::ElectricField, crate::fields::field_container::Rank::Vector(PolarizationBasis::SPHERICAL), self.frequency, self.convention, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::antenna_models::{HalfWaveDipole, Isotropic};
    use crate::fields::physics::peak_directivity;
    use lib_jsl_core::constants::em::C0;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn iso_array(n: usize, spacing_wavelengths: f64) -> SensorArray {
        let f = 1e9;
        let lambda = C0 / f;
        SensorArray::from_antenna(&SensorArray::linear_positions(n, spacing_wavelengths * lambda, [1.0, 0.0, 0.0]), Arc::new(Isotropic::theta_polarized()), &RigidTransform::identity(), f, TimeConvention::PlusJ, 1.0).unwrap()
    }

    #[test]
    fn geometric_phases_and_steering() {
        let a = iso_array(4, 0.5);
        assert_eq!(a.len(), 4);
        // Broadside (θ = 90°, φ = 90°: along +y) arrives in phase at every element.
        let s = a.steering_vector(FRAC_PI_2, FRAC_PI_2, [C64::new(1.0, 0.0), C64::new(0.0, 0.0)]).unwrap();
        assert!(s.iter().all(|v| (v - s[0]).norm() < 1e-9));
        // End-fire (+x): consecutive elements differ by k d = π.
        let e = a.steering_vector(FRAC_PI_2, 0.0, [C64::new(1.0, 0.0), C64::new(0.0, 0.0)]).unwrap();
        assert!((e[1] / e[0] - C64::from_polar(1.0, PI)).norm() < 1e-9);
        // A φ̂-polarized wave does not excite θ̂-polarized elements.
        let x = a.steering_vector(1.0, 0.3, [C64::new(0.0, 0.0), C64::new(1.0, 0.0)]).unwrap();
        assert!(x.iter().all(|v| v.norm() < 1e-12));
        // Steering the weights to a direction maximises the pattern there.
        let w = a.steering_vector(1.2, 0.4, [C64::new(1.0, 0.0), C64::new(0.0, 0.0)]).unwrap();
        let at = |t: f64, p: f64| a.pattern_at(&w, t, p).unwrap()[0].norm();
        assert!((at(1.2, 0.4) - 4.0).abs() < 1e-9 && at(0.5, 2.0) < 4.0);
    }

    #[test]
    fn broadside_array_directivity_and_element_orientation() {
        // An N-element half-wavelength broadside array of isotropic elements has directivity N.
        let a = iso_array(6, 0.5);
        let w = Array1::from_elem(6, C64::new(1.0, 0.0));
        let pat = a.array_pattern(&w, Arc::new(Domain::far_field_gauss(61, 120).unwrap())).unwrap();
        let d = peak_directivity(&pat).unwrap();
        assert!((d - 6.0).abs() < 0.05, "{d}");
        // Half-wave dipoles along z: |g|² is the gain, with nulls on the axis.
        let f = 1e9;
        let lambda = C0 / f;
        let dip = SensorArray::from_antenna(&SensorArray::planar_positions(2, 2, lambda / 2.0, lambda / 2.0), Arc::new(HalfWaveDipole::new(f)), &RigidTransform::identity(), f, TimeConvention::PlusJ, 1.0).unwrap();
        assert!(dip.element_response(0, 0.0, 0.0).unwrap()[0].norm() < 1e-9);
        assert!((dip.element_response(0, FRAC_PI_2, 0.0).unwrap()[0].norm_sqr() - 1.641).abs() < 2e-3);
        // Rotated to lie along x, the null moves to the x axis; towards +z it is x̂-polarized.
        let rot = RigidTransform::rotation_about([0.0, 1.0, 0.0], FRAC_PI_2);
        let horiz = SensorArray::from_antenna(&[[0.0; 3]], Arc::new(HalfWaveDipole::new(f)), &rot, f, TimeConvention::PlusJ, 1.0).unwrap();
        let r = horiz.element_response(0, FRAC_PI_2, 0.0).unwrap();
        assert!(r[0].norm() + r[1].norm() < 1e-9);
        let up = horiz.element_response(0, 1e-3, 0.0).unwrap(); // near +z, φ = 0: θ̂ ≈ x̂
        assert!((up[0].norm_sqr() - 1.641).abs() < 1e-2 && up[1].norm() < 1e-2);
        // A sampled copy of the same pattern agrees with the analytic element.
        let dom = Arc::new(Domain::far_field_uniform(91, 180).unwrap());
        let sampled = SensorArray::from_pattern(&[[0.0; 3]], Arc::new(HalfWaveDipole::new(f).gain_pattern(dom, f, TimeConvention::PlusJ, 1.0).unwrap())).unwrap();
        let analytic = SensorArray::from_antenna(&[[0.0; 3]], Arc::new(HalfWaveDipole::new(f)), &RigidTransform::identity(), f, TimeConvention::PlusJ, 1.0).unwrap();
        for (t, p) in [(1.0, 0.7), (0.2, 3.0), (2.9, -1.0)] {
            let (aa, ss) = (analytic.element_response(0, t, p).unwrap(), sampled.element_response(0, t, p).unwrap());
            assert!((aa[0] - ss[0]).norm() < 1e-3 && (aa[1] - ss[1]).norm() < 1e-3, "({t}, {p})");
        }
    }

    #[test]
    fn validation() {
        let mut a = SensorArray::new(1e9, TimeConvention::PlusJ).unwrap();
        let plane = Arc::new(Domain::plane(crate::fields::domain::Axis::uniform(0.0, 1.0, 2).unwrap(), crate::fields::domain::Axis::uniform(0.0, 1.0, 2).unwrap(), 0.0).unwrap());
        let near = Arc::new(HarmonicField::zeros(plane, FieldQuantity::ElectricField, crate::fields::field_container::Rank::Vector(PolarizationBasis::CARTESIAN), 1e9, TimeConvention::PlusJ).unwrap());
        assert!(a.add_sampled_element([0.0; 3], near).is_err());
        let ff = Arc::new(Domain::far_field_gauss(4, 8).unwrap());
        let wrong_f = Arc::new(HalfWaveDipole::new(1e9).gain_pattern(ff.clone(), 2e9, TimeConvention::PlusJ, 1.0).unwrap());
        assert!(a.add_sampled_element([0.0; 3], wrong_f).is_err());
        let other_conv = Arc::new(HalfWaveDipole::new(1e9).gain_pattern(ff, 1e9, TimeConvention::MinusI, 1.0).unwrap());
        a.add_sampled_element([0.0; 3], other_conv).unwrap(); // converted on insertion
        assert!(matches!(&a.elements()[0].pattern, ElementPattern::Sampled(p) if p.convention() == TimeConvention::PlusJ));
        assert!(a.pattern_at(&Array1::zeros(3), 0.0, 0.0).is_err());
        assert!(a.add_antenna_element([0.0; 3], Arc::new(Isotropic::theta_polarized()), Rotation3::identity(), 1.2).is_err());
        assert!(SensorArray::new(0.0, TimeConvention::PlusJ).is_err());
    }
}
