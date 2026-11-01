//! Conventions shared by the whole crate: the harmonic time convention, media, and dB helpers.
//!
//! Units are SI. Harmonic quantities are complex peak phasors; time-averaged products carry the
//! factor ½ (for example `S = ½ E × H*`).

use lib_jsl_core::constants::em::{C0, EPS_0, MU_0};
use num_complex::Complex;
use std::f64::consts::TAU;

pub type C64 = Complex<f64>;

/// The sign convention relating a phasor `X` to its real signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeConvention {
    /// `x(t) = Re{X e^{+jωt}}` (engineering). Outgoing waves are `e^{-jkr}`, `∂/∂t → +jω`.
    #[default]
    PlusJ,
    /// `x(t) = Re{X e^{-iωt}}` (physics). Outgoing waves are `e^{+ikr}`, `∂/∂t → -iω`.
    MinusI,
}

impl TimeConvention {
    /// `+1` for `e^{+jωt}`, `-1` for `e^{-iωt}`: the sign of `ωt` in the exponent.
    pub fn sign(self) -> f64 {
        match self {
            TimeConvention::PlusJ => 1.0,
            TimeConvention::MinusI => -1.0,
        }
    }

    /// The imaginary unit as it appears in `e^{(this) ωt}`: `+j` or `-i`.
    pub fn unit(self) -> C64 {
        C64::new(0.0, self.sign())
    }

    /// The factor that `∂/∂t` becomes: `+jω` or `-iω`.
    pub fn time_derivative(self, omega: f64) -> C64 {
        self.unit() * omega
    }

    /// The outgoing spherical wave phase `e^{-jkr}` (or `e^{+ikr}`) for a possibly complex `k`.
    pub fn outgoing(self, k: C64, r: f64) -> C64 {
        (-self.unit() * k * r).exp()
    }

    /// Re-expresses a phasor written in this convention in the `other` convention (complex
    /// conjugation when they differ), so both describe the same real signal.
    pub fn convert(self, value: C64, other: TimeConvention) -> C64 {
        if self == other { value } else { value.conj() }
    }

    /// The real signal of a phasor at time `t`.
    pub fn instantaneous(self, phasor: C64, omega: f64, t: f64) -> f64 {
        (phasor * C64::from_polar(1.0, self.sign() * omega * t)).re
    }
}

/// A linear, homogeneous, isotropic medium described by complex relative permittivity and
/// permeability. Losses are given in the [`TimeConvention::PlusJ`] form (`ε_r = ε' - jε''` with
/// `ε'' >= 0`); the medium's methods convert to the requested convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Medium {
    pub eps_r: C64,
    pub mu_r: C64,
}

impl Default for Medium {
    fn default() -> Self {
        Medium::VACUUM
    }
}

impl Medium {
    pub const VACUUM: Medium = Medium { eps_r: C64 { re: 1.0, im: 0.0 }, mu_r: C64 { re: 1.0, im: 0.0 } };

    /// A lossless dielectric with relative permittivity `eps_r`.
    pub fn dielectric(eps_r: f64) -> Self {
        Medium { eps_r: C64::new(eps_r, 0.0), mu_r: C64::new(1.0, 0.0) }
    }

    /// A lossy dielectric from relative permittivity and loss tangent `tan δ`.
    pub fn lossy_dielectric(eps_r: f64, loss_tangent: f64) -> Self {
        Medium { eps_r: C64::new(eps_r, -eps_r * loss_tangent), mu_r: C64::new(1.0, 0.0) }
    }

    /// A conductor-like medium: relative permittivity plus conductivity `σ` at `frequency`.
    pub fn with_conductivity(eps_r: f64, sigma: f64, frequency: f64) -> Self {
        Medium { eps_r: C64::new(eps_r, -sigma / (TAU * frequency * EPS_0)), mu_r: C64::new(1.0, 0.0) }
    }

    fn in_convention(&self, convention: TimeConvention) -> (C64, C64) {
        (TimeConvention::PlusJ.convert(self.eps_r, convention), TimeConvention::PlusJ.convert(self.mu_r, convention))
    }

    /// The wavenumber `k = ω √(μ ε)` at `frequency`, on the branch for which outgoing waves decay
    /// (`Im k <= 0` for `e^{+jωt}`, `Im k >= 0` for `e^{-iωt}`), with `Re k >= 0`.
    pub fn wavenumber(&self, frequency: f64, convention: TimeConvention) -> C64 {
        let (eps, mu) = self.in_convention(convention);
        let mut k = (eps * mu).sqrt() * (TAU * frequency / C0);
        if k.re < 0.0 {
            k = -k;
        }
        k
    }

    /// The intrinsic impedance `η = √(μ/ε)` (`≈ 376.73 Ω` in vacuum), with `Re η >= 0`.
    pub fn impedance(&self, convention: TimeConvention) -> C64 {
        let (eps, mu) = self.in_convention(convention);
        let mut z = (mu * MU_0 / (eps * EPS_0)).sqrt();
        if z.re < 0.0 {
            z = -z;
        }
        z
    }

    /// The wavelength `2π / Re k`.
    pub fn wavelength(&self, frequency: f64) -> f64 {
        TAU / self.wavenumber(frequency, TimeConvention::PlusJ).re
    }

    /// The phase velocity `ω / Re k`.
    pub fn phase_velocity(&self, frequency: f64) -> f64 {
        TAU * frequency / self.wavenumber(frequency, TimeConvention::PlusJ).re
    }
}

/// A power ratio in decibels.
pub fn power_to_db(ratio: f64) -> f64 {
    10.0 * ratio.log10()
}

/// Decibels to a power ratio.
pub fn db_to_power(db: f64) -> f64 {
    10f64.powf(db / 10.0)
}

/// A field (amplitude) ratio in decibels.
pub fn amplitude_to_db(ratio: f64) -> f64 {
    20.0 * ratio.log10()
}

/// Decibels to an amplitude ratio.
pub fn db_to_amplitude(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conventions_describe_the_same_real_signal() {
        let x = C64::from_polar(2.0, 0.7);
        let omega = 3.0;
        for t in [0.0, 0.1, 0.37, 1.9] {
            let a = TimeConvention::PlusJ.instantaneous(x, omega, t);
            let b = TimeConvention::MinusI.instantaneous(TimeConvention::PlusJ.convert(x, TimeConvention::MinusI), omega, t);
            assert!((a - b).abs() < 1e-12);
            assert!((a - 2.0 * (omega * t + 0.7).cos()).abs() < 1e-12);
        }
        assert_eq!(TimeConvention::default(), TimeConvention::PlusJ);
        assert_eq!(TimeConvention::PlusJ.convert(x, TimeConvention::PlusJ), x);
        // d/dt of Re{X e^{sjωt}} equals Re{(sjω) X e^{sjωt}} in both conventions.
        for conv in [TimeConvention::PlusJ, TimeConvention::MinusI] {
            let (t, h) = (0.4, 1e-6);
            let numeric = (conv.instantaneous(x, omega, t + h) - conv.instantaneous(x, omega, t - h)) / (2.0 * h);
            assert!((numeric - conv.instantaneous(conv.time_derivative(omega) * x, omega, t)).abs() < 1e-6);
        }
    }

    #[test]
    fn vacuum_and_lossy_media() {
        let f = 1e9;
        let k = Medium::VACUUM.wavenumber(f, TimeConvention::PlusJ);
        assert!((k.re - TAU * f / C0).abs() < 1e-9 && k.im == 0.0);
        assert!((Medium::VACUUM.impedance(TimeConvention::PlusJ).re - 376.730_313).abs() < 1e-3);
        assert!((Medium::VACUUM.wavelength(f) - C0 / f).abs() < 1e-12);
        let glass = Medium::dielectric(4.0);
        assert!((glass.wavelength(f) - C0 / f / 2.0).abs() < 1e-12 && (glass.phase_velocity(f) - C0 / 2.0).abs() < 1e-3);
        // A lossy medium attenuates outgoing waves in either convention.
        let lossy = Medium::lossy_dielectric(4.0, 0.1);
        for conv in [TimeConvention::PlusJ, TimeConvention::MinusI] {
            let kc = lossy.wavenumber(f, conv);
            assert!(kc.re > 0.0);
            assert!(conv.outgoing(kc, 2.0).norm() < conv.outgoing(kc, 1.0).norm(), "{conv:?}");
            assert!((conv.outgoing(k, 3.0).norm() - 1.0).abs() < 1e-12);
        }
        assert!(Medium::with_conductivity(1.0, 5.8e7, 1e6).eps_r.im < -1e10);
    }

    #[test]
    fn decibels() {
        assert!((power_to_db(100.0) - 20.0).abs() < 1e-12 && (db_to_power(3.0) - 1.995_262_314_968_88).abs() < 1e-12);
        assert!((amplitude_to_db(10.0) - 20.0).abs() < 1e-12 && (db_to_amplitude(-6.0) - 0.501_187_233_627_272_3).abs() < 1e-12);
    }
}
