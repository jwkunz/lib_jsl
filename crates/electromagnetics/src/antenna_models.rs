//! Antenna models: analytic far fields, and the derived pattern, power and directivity quantities.
//!
//! Every model implements [`Antenna`]. Its required method gives the far field
//! `(E_θ, E_φ)` with the `e^{-jkr}/r` factor removed, for a unit excitation of the model's kind
//! ([`Antenna::excitation`]: a terminal current in amperes for wire antennas, the peak aperture
//! field in V/m for apertures, the edge voltage for patches), in the requested time convention.
//! Models are defined in their own frame (documented per model); place them with
//! [`HarmonicField::transformed`] on their pattern containers.

pub mod half_wave_dipole;
pub mod dish;
pub mod hertzian_dipole;
pub mod magnetic_loop;
pub mod patch_antenna;

pub use dish::Dish;
pub use half_wave_dipole::{Dipole, HalfWaveDipole};
pub use hertzian_dipole::HertzianDipole;
pub use magnetic_loop::MagneticLoop;
pub use patch_antenna::PatchAntenna;

use crate::conventions::{C64, Medium, TimeConvention};
use crate::coordinates::polarization::PolarizationBasis;
use crate::fields::domain::Domain;
use crate::fields::field_container::{FieldQuantity, HarmonicField};
use lib_jsl_core::ErrorsJSL;
use std::f64::consts::PI;
use std::sync::Arc;

/// The physical kind of a model's unit excitation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Excitation {
    /// A terminal (or reference) current of 1 A peak.
    Current,
    /// A peak aperture field of 1 V/m.
    ApertureField,
    /// A radiating-edge voltage of 1 V peak.
    Voltage,
    /// A dimensionless unit pattern (idealised sources).
    Normalized,
}

/// An antenna with an analytic far field.
pub trait Antenna {
    /// The far field `[E_θ, E_φ]` in the direction `(θ, φ)` with `e^{-jkr}/r` removed (V), per
    /// unit excitation, in vacuum.
    fn far_field(&self, theta: f64, phi: f64, frequency: f64, convention: TimeConvention) -> [C64; 2];

    /// What the unit excitation is.
    fn excitation(&self) -> Excitation;

    /// The far-field pattern as an electric-field container on a far-field domain (spherical basis).
    fn far_field_pattern(&self, domain: Arc<Domain>, frequency: f64, convention: TimeConvention) -> Result<HarmonicField, ErrorsJSL> {
        if !domain.is_far_field() {
            return Err(ErrorsJSL::InvalidInputRange("Far-field patterns live on the far-field sphere."));
        }
        HarmonicField::vector_from_fn(domain, FieldQuantity::ElectricField, PolarizationBasis::SPHERICAL, frequency, convention, |p| {
            let [et, ep] = self.far_field(p.coords[1], p.coords[2], frequency, convention);
            [C64::new(0.0, 0.0), et, ep]
        })
    }

    /// The `(θ nodes, φ samples)` of the grid used for numerical power integration.
    fn power_grid(&self, _frequency: f64) -> (usize, usize) {
        (48, 96)
    }

    /// The radiated power `∫ |E|² / (2η) dΩ` (W) for a unit excitation.
    fn radiated_power(&self, frequency: f64) -> f64 {
        let (nt, np) = self.power_grid(frequency);
        let d = Domain::far_field_gauss(nt, np).expect("valid grid");
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        d.integrate_fn(|p| {
            let [a, b] = self.far_field(p.coords[1], p.coords[2], frequency, TimeConvention::PlusJ);
            (a.norm_sqr() + b.norm_sqr()) / (2.0 * eta)
        })
    }

    /// The radiation intensity `U = |E|² / (2η)` (W/sr) for a unit excitation.
    fn radiation_intensity(&self, theta: f64, phi: f64, frequency: f64) -> f64 {
        let [a, b] = self.far_field(theta, phi, frequency, TimeConvention::PlusJ);
        (a.norm_sqr() + b.norm_sqr()) / (2.0 * Medium::VACUUM.impedance(TimeConvention::PlusJ).re)
    }

    /// The directivity `4π U(θ, φ) / P_rad` in a direction.
    fn directivity(&self, theta: f64, phi: f64, frequency: f64) -> f64 {
        4.0 * PI * self.radiation_intensity(theta, phi, frequency) / self.radiated_power(frequency)
    }

    /// The peak directivity, searched on a half-degree grid.
    fn peak_directivity(&self, frequency: f64) -> f64 {
        let p = self.radiated_power(frequency);
        let mut best: f64 = 0.0;
        for i in 0..=360 {
            for j in 0..720 {
                let (t, f) = (PI * i as f64 / 360.0, PI * j as f64 / 360.0);
                best = best.max(self.radiation_intensity(t, f, frequency));
            }
        }
        4.0 * PI * best / p
    }

    /// The radiation resistance `2 P / |I|²` (Ω) referred to the unit excitation current, for
    /// current-excited models.
    fn radiation_resistance(&self, frequency: f64) -> Option<f64> {
        (self.excitation() == Excitation::Current).then(|| 2.0 * self.radiated_power(frequency))
    }

    /// The pattern scaled so that `|E_θ|² + |E_φ|²` equals the directivity times `efficiency`
    /// (the gain): the normalisation the array signal models expect.
    fn gain_pattern(&self, domain: Arc<Domain>, frequency: f64, convention: TimeConvention, efficiency: f64) -> Result<HarmonicField, ErrorsJSL> {
        let p = self.radiated_power(frequency);
        if p.is_nan() || p <= 0.0 || !(0.0..=1.0).contains(&efficiency) {
            return Err(ErrorsJSL::InvalidInputRange("Need a radiating antenna and an efficiency in [0, 1]."));
        }
        let eta = Medium::VACUUM.impedance(convention).re;
        let scale = (4.0 * PI * efficiency / (2.0 * eta * p)).sqrt();
        Ok(self.far_field_pattern(domain, frequency, convention)?.scaled(C64::new(scale, 0.0)))
    }
}

/// An idealised point source radiating equally in every direction with a fixed polarization
/// `[p_θ, p_φ]` (not physically realisable for all directions at once, but the standard array
/// model element). Its unit pattern has directivity 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Isotropic {
    pub polarization: [C64; 2],
}

impl Isotropic {
    /// θ̂-polarized.
    pub fn theta_polarized() -> Self {
        Isotropic { polarization: [C64::new(1.0, 0.0), C64::new(0.0, 0.0)] }
    }
}

impl Antenna for Isotropic {
    fn far_field(&self, _theta: f64, _phi: f64, _frequency: f64, _convention: TimeConvention) -> [C64; 2] {
        let n = (self.polarization[0].norm_sqr() + self.polarization[1].norm_sqr()).sqrt().max(1e-300);
        self.polarization.map(|v| v / n)
    }

    fn excitation(&self) -> Excitation {
        Excitation::Normalized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotropic_source_has_unit_directivity_and_gain_patterns_are_normalised() {
        let iso = Isotropic::theta_polarized();
        let f = 1e9;
        assert!((iso.directivity(0.3, 1.0, f) - 1.0).abs() < 1e-12);
        assert!((iso.peak_directivity(f) - 1.0).abs() < 1e-12);
        assert_eq!(iso.radiation_resistance(f), None);
        let d = Arc::new(Domain::far_field_gauss(8, 16).unwrap());
        let g = iso.gain_pattern(d.clone(), f, TimeConvention::PlusJ, 0.5).unwrap();
        assert!(g.norm_squared().iter().all(|v| (v - 0.5).abs() < 1e-12)); // |g|² = efficiency × D
        assert!((g.integrate_norm_squared() - 2.0 * PI).abs() < 1e-12); // ∫ G dΩ = 4π η
        assert!(iso.gain_pattern(d.clone(), f, TimeConvention::PlusJ, 1.5).is_err());
        assert!(iso.far_field_pattern(Arc::new(Domain::plane(crate::fields::domain::Axis::uniform(0.0, 1.0, 2).unwrap(), crate::fields::domain::Axis::uniform(0.0, 1.0, 2).unwrap(), 0.0).unwrap()), f, TimeConvention::PlusJ).is_err());
    }
}
