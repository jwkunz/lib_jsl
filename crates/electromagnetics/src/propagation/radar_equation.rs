//! The radar range equation and basic radar relations.
//!
//! See the book entry [Propagation](https://jwkunz.github.io/lib_jsl/book/em-waves/propagation.html).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_core::constants::em::C0;
use lib_jsl_core::constants::physics::K_B;
use std::f64::consts::PI;

/// Parameters of a monostatic radar (linear units unless noted).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Radar {
    /// Peak transmit power (W).
    pub transmit_power: f64,
    /// Transmit and receive antenna gain (the same antenna).
    pub gain: f64,
    pub frequency: f64,
    /// Receiver noise bandwidth (Hz).
    pub bandwidth: f64,
    /// Receiver noise figure (linear).
    pub noise_figure: f64,
    /// System losses (linear, ≥ 1).
    pub losses: f64,
    /// Reference noise temperature (K), normally 290.
    pub noise_temperature: f64,
}

impl Radar {
    pub fn wavelength(&self) -> f64 {
        C0 / self.frequency
    }

    /// The echo power `P_t G² λ² σ / ((4π)³ R⁴ L)` from a target of cross-section `rcs` (m²) at
    /// range `range` (m).
    pub fn received_power(&self, rcs: f64, range: f64) -> f64 {
        self.transmit_power * self.gain * self.gain * self.wavelength().powi(2) * rcs / ((4.0 * PI).powi(3) * range.powi(4) * self.losses)
    }

    /// The receiver noise power `k T B F`.
    pub fn noise_power(&self) -> f64 {
        K_B * self.noise_temperature * self.bandwidth * self.noise_figure
    }

    /// The single-pulse signal-to-noise ratio (linear).
    pub fn snr(&self, rcs: f64, range: f64) -> f64 {
        self.received_power(rcs, range) / self.noise_power()
    }

    /// The SNR after integrating `pulses` pulses coherently (gain `n`).
    pub fn coherent_snr(&self, rcs: f64, range: f64, pulses: usize) -> f64 {
        pulses as f64 * self.snr(rcs, range)
    }

    /// The maximum range at which the SNR (after an integration gain) reaches `required_snr`:
    /// `R = (P_t G² λ² σ g / ((4π)³ k T B F L SNR))^{1/4}`.
    pub fn maximum_range(&self, rcs: f64, required_snr: f64, integration_gain: f64) -> Result<f64, ErrorsJSL> {
        if !(required_snr > 0.0) || !(rcs > 0.0) || !(integration_gain > 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("Need a positive RCS, required SNR and integration gain."));
        }
        Ok((self.received_power(rcs, 1.0) * integration_gain / (self.noise_power() * required_snr)).powf(0.25))
    }
}

/// The bistatic echo power `P_t G_t G_r λ² σ_b / ((4π)³ R_t² R_r² L)`.
#[allow(clippy::too_many_arguments)]
pub fn bistatic_received_power(transmit_power: f64, gt: f64, gr: f64, frequency: f64, rcs: f64, rt: f64, rr: f64, losses: f64) -> f64 {
    let lambda = C0 / frequency;
    transmit_power * gt * gr * lambda * lambda * rcs / ((4.0 * PI).powi(3) * rt * rt * rr * rr * losses)
}

/// The range resolution `c / (2B)` of a waveform of bandwidth `B`.
pub fn range_resolution(bandwidth: f64) -> f64 {
    C0 / (2.0 * bandwidth)
}

/// The maximum unambiguous range `c / (2 PRF)`.
pub fn unambiguous_range(prf: f64) -> f64 {
    C0 / (2.0 * prf)
}

/// The Doppler shift `2 v / λ` of a target closing at radial speed `v` (m/s).
pub fn doppler_shift(radial_velocity: f64, frequency: f64) -> f64 {
    2.0 * radial_velocity * frequency / C0
}

/// The radar horizon (m) for antenna height `h` over a smooth earth with the standard 4/3
/// effective radius: `√(2 k_e R_E h)`.
pub fn radar_horizon(height: f64) -> f64 {
    (2.0 * 4.0 / 3.0 * 6.371e6 * height).sqrt()
}

/// The radar cross-section of a perfectly conducting sphere of radius `a` in the optical region
/// (`a ≫ λ`): `πa²`.
pub fn sphere_rcs_optical(radius: f64) -> f64 {
    PI * radius * radius
}

/// The peak radar cross-section of a flat conducting plate of area `A` at normal incidence:
/// `4πA²/λ²`.
pub fn flat_plate_rcs(area: f64, frequency: f64) -> f64 {
    let lambda = C0 / frequency;
    4.0 * PI * area * area / (lambda * lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn radar() -> Radar {
        Radar { transmit_power: 1e6, gain: 10f64.powf(3.5), frequency: 3e9, bandwidth: 1e6, noise_figure: 10f64.powf(0.3), losses: 10f64.powf(0.5), noise_temperature: 290.0 }
    }

    #[test]
    fn range_equation_scales_and_inverts() {
        let r = radar();
        let snr = r.snr(1.0, 50e3);
        // Fourth-power law, and the maximum range inverts the SNR.
        assert!((r.snr(1.0, 100e3) * 16.0 / snr - 1.0).abs() < 1e-12);
        let rmax = r.maximum_range(1.0, snr, 1.0).unwrap();
        assert!((rmax - 50e3).abs() < 1e-6);
        // Ten coherent pulses extend the range by 10^{1/4}.
        assert!((r.maximum_range(1.0, snr, 10.0).unwrap() / rmax - 10f64.powf(0.25)).abs() < 1e-12);
        assert!((r.coherent_snr(1.0, 50e3, 10) - 10.0 * snr).abs() < 1e-9 * snr);
        assert!((r.noise_power() - K_B * 290.0 * 1e6 * 10f64.powf(0.3)).abs() < 1e-30);
        // Monostatic is the bistatic case with coincident antennas.
        let b = bistatic_received_power(r.transmit_power, r.gain, r.gain, r.frequency, 1.0, 50e3, 50e3, r.losses);
        assert!((b / r.received_power(1.0, 50e3) - 1.0).abs() < 1e-12);
        assert!(r.maximum_range(1.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn waveform_relations() {
        assert!((range_resolution(150e6) - 0.999_308_2).abs() < 1e-6);
        assert!((unambiguous_range(1e3) - 149_896.229).abs() < 1e-3);
        assert!((doppler_shift(300.0, 10e9) - 20_013.8).abs() < 0.1);
        assert!((radar_horizon(10.0) / 1000.0 - 13.03).abs() < 0.01);
        assert!((sphere_rcs_optical(1.0) - PI).abs() < 1e-15);
        assert!(flat_plate_rcs(1.0, 10e9) > 1000.0);
    }
}
