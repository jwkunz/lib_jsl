//! Thin-wire dipoles of finite length, and the half-wave dipole.

use super::{Antenna, Excitation};
use crate::conventions::{C64, Medium, TimeConvention};
use lib_jsl_core::constants::em::C0;
use std::f64::consts::{PI, TAU};

/// A centre-fed thin wire of total length `L` along `z` with the standard sinusoidal current
/// `I(z) = I_m sin(k (L/2 - |z|))`. Far field per unit *maximum* current `I_m`
/// (Balanis §4.6), in the `e^{+jωt}` convention:
///
/// `E_θ = j η I_m / (2π) · [cos(kL/2 cos θ) - cos(kL/2)] / sin θ`.
///
/// Radiation resistance is referred to the current maximum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dipole {
    pub length: f64,
}

impl Dipole {
    pub fn new(length: f64) -> Self {
        Dipole { length }
    }
}

impl Antenna for Dipole {
    fn far_field(&self, theta: f64, _phi: f64, frequency: f64, convention: TimeConvention) -> [C64; 2] {
        let half_kl = 0.5 * TAU * frequency / C0 * self.length;
        let s = theta.sin();
        let eta = Medium::VACUUM.impedance(TimeConvention::PlusJ).re;
        let shape = if s.abs() < 1e-12 { 0.0 } else { ((half_kl * theta.cos()).cos() - half_kl.cos()) / s };
        [convention.unit() * (eta / TAU * shape), C64::new(0.0, 0.0)]
    }

    fn excitation(&self) -> Excitation {
        Excitation::Current
    }

    fn power_grid(&self, frequency: f64) -> (usize, usize) {
        // Longer dipoles have more lobes; keep several nodes per lobe.
        let electrical = self.length * frequency / C0;
        ((48.0 + 24.0 * electrical).ceil() as usize, 8)
    }

    fn peak_directivity(&self, frequency: f64) -> f64 {
        // The pattern is rotationally symmetric: search θ only, finely.
        let p = self.radiated_power(frequency);
        let best = (0..=20_000).map(|i| self.radiation_intensity(PI * i as f64 / 20_000.0, 0.0, frequency)).fold(0.0, f64::max);
        4.0 * PI * best / p
    }
}

/// A dipole cut to half a wavelength at `design_frequency` (`L = λ₀/2`). At the design frequency
/// it has directivity ≈ 1.64 (2.15 dBi) and radiation resistance ≈ 73.1 Ω; at other frequencies
/// it behaves as the corresponding finite [`Dipole`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfWaveDipole {
    pub design_frequency: f64,
}

impl HalfWaveDipole {
    pub fn new(design_frequency: f64) -> Self {
        HalfWaveDipole { design_frequency }
    }

    /// The physical length `λ₀ / 2`.
    pub fn length(&self) -> f64 {
        0.5 * C0 / self.design_frequency
    }

    fn dipole(&self) -> Dipole {
        Dipole::new(self.length())
    }
}

impl Antenna for HalfWaveDipole {
    fn far_field(&self, theta: f64, phi: f64, frequency: f64, convention: TimeConvention) -> [C64; 2] {
        self.dipole().far_field(theta, phi, frequency, convention)
    }

    fn excitation(&self) -> Excitation {
        Excitation::Current
    }

    fn power_grid(&self, frequency: f64) -> (usize, usize) {
        self.dipole().power_grid(frequency)
    }

    fn peak_directivity(&self, frequency: f64) -> f64 {
        self.dipole().peak_directivity(frequency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_wave_dipole_textbook_values() {
        let f = 1e9;
        let d = HalfWaveDipole::new(f);
        assert!((d.length() - 0.149_896_229).abs() < 1e-9);
        let r = d.radiation_resistance(f).unwrap();
        assert!((r - 73.08).abs() < 0.02, "{r}"); // 30 Cin(2π) = 73.08 Ω
        let dmax = d.peak_directivity(f);
        assert!((dmax - 1.6409).abs() < 1e-3, "{dmax}");
        assert!((10.0 * dmax.log10() - 2.15).abs() < 0.01);
        assert!(d.far_field(0.0, 0.0, f, TimeConvention::PlusJ)[0].norm() < 1e-12); // nulls on axis
        // The two conventions give conjugate phasors.
        let (a, b) = (d.far_field(1.0, 0.0, f, TimeConvention::PlusJ)[0], d.far_field(1.0, 0.0, f, TimeConvention::MinusI)[0]);
        assert!((a - b.conj()).norm() < 1e-12);
    }

    #[test]
    fn short_dipoles_approach_the_hertzian_limit_and_long_ones_sharpen() {
        let f = 3e8;
        let short = Dipole::new(0.01);
        assert!((short.peak_directivity(f) - 1.5).abs() < 1e-3);
        // A full-wave dipole is more directive (≈ 2.41) with R ≈ 199 Ω at its current maximum.
        let full = Dipole::new(1.0);
        assert!((full.peak_directivity(f) - 2.41).abs() < 0.01, "{}", full.peak_directivity(f));
        assert!((full.radiation_resistance(f).unwrap() - 199.1).abs() < 0.5, "{}", full.radiation_resistance(f).unwrap());
    }
}
