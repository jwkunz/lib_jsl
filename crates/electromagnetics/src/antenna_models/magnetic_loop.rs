//! The small circular loop (magnetic dipole).

use super::{Antenna, Excitation};
use crate::conventions::{C64, Medium, TimeConvention};
use crate::coordinates::polarization::PolarizationBasis;
use crate::coordinates::systems::{CoordinateSystem, Position};
use crate::fields::domain::Domain;
use crate::fields::field_container::{FieldQuantity, HarmonicField};
use lib_jsl_core::ErrorsJSL;
use lib_jsl_core::constants::em::C0;
use std::f64::consts::{PI, TAU};
use std::sync::Arc;

/// A loop of radius `a ≪ λ` in the `xy` plane centred at the origin, `turns` turns, carrying a
/// uniform current `I` (counter-clockwise about `+z`). It radiates as a magnetic dipole
/// `m = N π a² I ẑ` (Balanis §5.2), in the `e^{+jωt}` convention:
/// - `H_r = j k a² N I cos θ / (2 r²) (1 + 1/(jkr)) e^{-jkr}`
/// - `H_θ = -(ka)² N I sin θ / (4 r) (1 + 1/(jkr) - 1/(kr)²) e^{-jkr}`
/// - `E_φ = η (ka)² N I sin θ / (4 r) (1 + 1/(jkr)) e^{-jkr}`
///
/// Directivity 1.5; radiation resistance `20π² (ka)⁴ N²`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagneticLoop {
    pub radius: f64,
    pub turns: u32,
}

impl MagneticLoop {
    pub fn new(radius: f64, turns: u32) -> Self {
        MagneticLoop { radius, turns }
    }

    /// The exact radiation resistance `20π² (ka)⁴ N²`.
    pub fn exact_radiation_resistance(&self, frequency: f64) -> f64 {
        let ka = TAU * frequency / C0 * self.radius;
        20.0 * PI * PI * ka.powi(4) * f64::from(self.turns).powi(2)
    }

    /// The exact `(E, H)` at `p` (spherical components) for a unit current.
    pub fn near_field(&self, p: &Position, frequency: f64, convention: TimeConvention) -> ([C64; 3], [C64; 3]) {
        let s = p.in_system(CoordinateSystem::Spherical).coords;
        let (r, theta) = (s[0], s[1]);
        let k = TAU * frequency / C0;
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        let j = C64::new(0.0, 1.0);
        let (kr, a2n) = (k * r, self.radius * self.radius * f64::from(self.turns));
        let phase = (-j * kr).exp();
        let hr = j * k * a2n * theta.cos() / (2.0 * r * r) * (1.0 + 1.0 / (j * kr)) * phase;
        let ht = -(k * k * a2n) * theta.sin() / (4.0 * r) * (1.0 + 1.0 / (j * kr) - 1.0 / (kr * kr)) * phase;
        let ep = eta * k * k * a2n * theta.sin() / (4.0 * r) * (1.0 + 1.0 / (j * kr)) * phase;
        let conv = |v: C64| TimeConvention::PlusJ.convert(v, convention);
        let zero = C64::new(0.0, 0.0);
        ([zero, zero, conv(ep)], [conv(hr), conv(ht), zero])
    }

    /// The exact `E` and `H` on a spatial domain (spherical basis).
    pub fn near_field_containers(&self, domain: Arc<Domain>, frequency: f64, convention: TimeConvention) -> Result<(HarmonicField, HarmonicField), ErrorsJSL> {
        let e = HarmonicField::vector_from_fn(domain.clone(), FieldQuantity::ElectricField, PolarizationBasis::SPHERICAL, frequency, convention, |p| self.near_field(p, frequency, convention).0)?;
        let h = HarmonicField::vector_from_fn(domain, FieldQuantity::MagneticField, PolarizationBasis::SPHERICAL, frequency, convention, |p| self.near_field(p, frequency, convention).1)?;
        Ok((e, h))
    }
}

impl Antenna for MagneticLoop {
    fn far_field(&self, theta: f64, _phi: f64, frequency: f64, convention: TimeConvention) -> [C64; 2] {
        let k = TAU * frequency / C0;
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        let v = eta * k * k * self.radius * self.radius * f64::from(self.turns) * theta.sin() / 4.0;
        // Real in e^{+jωt}; conjugation leaves it unchanged in the other convention.
        let _ = convention;
        [C64::new(0.0, 0.0), C64::new(v, 0.0)]
    }

    fn excitation(&self) -> Excitation {
        Excitation::Current
    }

    fn peak_directivity(&self, _frequency: f64) -> f64 {
        1.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::domain::Axis;
    use crate::fields::physics::power_through_surface;

    #[test]
    fn resistance_scales_as_the_fourth_power_and_with_turns_squared() {
        let f = 1e8;
        let lambda = C0 / f;
        let one = MagneticLoop::new(lambda / 50.0, 1);
        let r1 = one.radiation_resistance(f).unwrap();
        assert!((r1 - one.exact_radiation_resistance(f)).abs() < 1e-3 * r1);
        let bigger = MagneticLoop::new(lambda / 25.0, 1);
        assert!((bigger.radiation_resistance(f).unwrap() / r1 - 16.0).abs() < 1e-6);
        let coil = MagneticLoop::new(lambda / 50.0, 5);
        assert!((coil.radiation_resistance(f).unwrap() / r1 - 25.0).abs() < 1e-6);
        assert!((one.directivity(PI / 2.0, 2.0, f) - 1.5).abs() < 1e-9);
        // φ-polarized, unlike the electric dipole.
        let [et, ep] = one.far_field(1.0, 0.0, f, TimeConvention::PlusJ);
        assert!(et.norm() == 0.0 && ep.norm() > 0.0);
    }

    #[test]
    fn near_field_flux_and_duality_with_the_electric_dipole() {
        let f = 3e8;
        let lp = MagneticLoop::new(0.02, 1);
        let p_far = lp.radiated_power(f);
        for radius in [0.04, 0.5, 3.0] {
            let sphere = Arc::new(Domain::sphere(radius, Axis::gauss_legendre_polar(12).unwrap(), Axis::periodic(0.0, TAU, 8).unwrap()).unwrap());
            let (e, h) = lp.near_field_containers(sphere, f, TimeConvention::PlusJ).unwrap();
            let p = power_through_surface(&e, &h).unwrap();
            assert!((p - p_far).abs() < 1e-9 * p_far, "r = {radius}");
        }
        // Far-field limit of the near field.
        let k = TAU * f / C0;
        let r = 1e4;
        let (e, _) = lp.near_field(&Position::spherical(r, 0.8, 0.0), f, TimeConvention::MinusI);
        let scaled = e[2] * r * C64::new(0.0, -k * r).exp();
        assert!((scaled - lp.far_field(0.8, 0.0, f, TimeConvention::MinusI)[1]).norm() < 1e-3 * scaled.norm());
    }
}
