//! The Hertzian (infinitesimal) electric dipole.
//!
//! See the book entry [Antenna models](https://jwkunz.github.io/lib_jsl/book/em-arrays/antennas.html).

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

/// A uniform current `I` on a short wire of length `l ≪ λ` along `+z` at the origin.
///
/// Exact fields for all `r > 0` (Balanis §4.2), in the `e^{+jωt}` convention:
/// - `E_r = η I l cos θ / (2π r²) (1 + 1/(jkr)) e^{-jkr}`
/// - `E_θ = j η k I l sin θ / (4π r) (1 + 1/(jkr) - 1/(kr)²) e^{-jkr}`
/// - `H_φ = j k I l sin θ / (4π r) (1 + 1/(jkr)) e^{-jkr}`
///
/// Directivity 1.5; radiation resistance `80π² (l/λ)²`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HertzianDipole {
    pub length: f64,
}

impl HertzianDipole {
    pub fn new(length: f64) -> Self {
        HertzianDipole { length }
    }

    /// The exact radiation resistance `80π² (l/λ)²`.
    pub fn exact_radiation_resistance(&self, frequency: f64) -> f64 {
        80.0 * PI * PI * (self.length * frequency / C0).powi(2)
    }

    /// The exact electric and magnetic fields `(E, H)` at `p` (spherical components
    /// `[r, θ, φ]`) for a unit current.
    pub fn near_field(&self, p: &Position, frequency: f64, convention: TimeConvention) -> ([C64; 3], [C64; 3]) {
        let s = p.in_system(CoordinateSystem::Spherical).coords;
        let (r, theta) = (s[0], s[1]);
        let k = TAU * frequency / C0;
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        let j = C64::new(0.0, 1.0);
        let kr = k * r;
        let phase = (-j * kr).exp();
        let il = self.length;
        let er = eta * il * theta.cos() / (TAU * r * r) * (1.0 + 1.0 / (j * kr)) * phase;
        let et = j * eta * k * il * theta.sin() / (4.0 * PI * r) * (1.0 + 1.0 / (j * kr) - 1.0 / (kr * kr)) * phase;
        let hp = j * k * il * theta.sin() / (4.0 * PI * r) * (1.0 + 1.0 / (j * kr)) * phase;
        let conv = |v: C64| TimeConvention::PlusJ.convert(v, convention);
        let zero = C64::new(0.0, 0.0);
        ([conv(er), conv(et), zero], [zero, zero, conv(hp)])
    }

    /// The exact `E` and `H` on a spatial domain (spherical basis).
    pub fn near_field_containers(&self, domain: Arc<Domain>, frequency: f64, convention: TimeConvention) -> Result<(HarmonicField, HarmonicField), ErrorsJSL> {
        let e = HarmonicField::vector_from_fn(domain.clone(), FieldQuantity::ElectricField, PolarizationBasis::SPHERICAL, frequency, convention, |p| self.near_field(p, frequency, convention).0)?;
        let h = HarmonicField::vector_from_fn(domain, FieldQuantity::MagneticField, PolarizationBasis::SPHERICAL, frequency, convention, |p| self.near_field(p, frequency, convention).1)?;
        Ok((e, h))
    }
}

impl Antenna for HertzianDipole {
    fn far_field(&self, theta: f64, _phi: f64, frequency: f64, convention: TimeConvention) -> [C64; 2] {
        let k = TAU * frequency / C0;
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        [convention.unit() * (eta * k * self.length * theta.sin() / (4.0 * PI)), C64::new(0.0, 0.0)]
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
    fn textbook_resistance_and_directivity() {
        let f = 1e9;
        let d = HertzianDipole::new(0.01 * C0 / f); // l = λ/100
        let r = d.radiation_resistance(f).unwrap();
        assert!((r - d.exact_radiation_resistance(f)).abs() < 1e-3 * r, "{r}");
        assert!((r - 80.0 * PI * PI * 1e-4).abs() < 2e-3 * r); // ≈ 0.079 Ω
        assert!((d.directivity(PI / 2.0, 0.7, f) - 1.5).abs() < 1e-9);
        assert!(d.directivity(0.0, 0.0, f).abs() < 1e-12);
        let generic_peak = Antenna::directivity(&d, PI / 2.0, 0.0, f);
        assert!((generic_peak - d.peak_directivity(f)).abs() < 1e-9);
    }

    #[test]
    fn near_field_power_flux_equals_the_far_field_power_at_any_radius() {
        let f = 3e8; // λ = 1 m
        let d = HertzianDipole::new(0.05);
        let p_far = d.radiated_power(f);
        for radius in [0.05, 0.3, 4.0] {
            let sphere = Arc::new(Domain::sphere(radius, Axis::gauss_legendre_polar(12).unwrap(), Axis::periodic(0.0, TAU, 8).unwrap()).unwrap());
            for conv in [TimeConvention::PlusJ, TimeConvention::MinusI] {
                let (e, h) = d.near_field_containers(sphere.clone(), f, conv).unwrap();
                let p = power_through_surface(&e, &h).unwrap();
                assert!((p - p_far).abs() < 1e-9 * p_far, "r = {radius}: {p} vs {p_far}");
            }
        }
        // Far away, r e^{jkr} E approaches the far-field pattern.
        let k = TAU * f / C0;
        let r = 1e4;
        let (e, _) = d.near_field(&Position::spherical(r, 1.0, 0.0), f, TimeConvention::PlusJ);
        let scaled = e[1] * r * C64::new(0.0, k * r).exp();
        assert!((scaled - d.far_field(1.0, 0.0, f, TimeConvention::PlusJ)[0]).norm() < 1e-3 * scaled.norm());
    }
}
