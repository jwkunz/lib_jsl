//! Free-space transmission (Friis) and empirical path loss models.
//!
//! See the book entry [Propagation](https://jwkunz.github.io/lib_jsl/book/em-waves/propagation.html).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_core::constants::em::C0;
use std::f64::consts::PI;

/// Linear power ratio from decibels.
pub fn from_db(db: f64) -> f64 {
    10f64.powf(db / 10.0)
}

/// Decibels from a linear power ratio.
pub fn to_db(ratio: f64) -> f64 {
    10.0 * ratio.log10()
}

/// The free-space path loss `(4πd/λ)²` in dB at distance `d` (m) and frequency `f` (Hz).
pub fn free_space_path_loss_db(distance: f64, frequency: f64) -> f64 {
    20.0 * (4.0 * PI * distance * frequency / C0).log10()
}

/// The Friis transmission equation: received power `P_t G_t G_r (λ / 4πd)²` (linear units), for
/// antennas in each other's far field.
pub fn friis_received_power(transmit_power: f64, transmit_gain: f64, receive_gain: f64, frequency: f64, distance: f64) -> Result<f64, ErrorsJSL> {
    if !(distance > 0.0) || !(frequency > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Need a positive distance and frequency."));
    }
    let lambda = C0 / frequency;
    Ok(transmit_power * transmit_gain * receive_gain * (lambda / (4.0 * PI * distance)).powi(2))
}

/// The far-field (Fraunhofer) distance `2D²/λ` of an aperture of largest dimension `D`.
pub fn fraunhofer_distance(aperture: f64, frequency: f64) -> f64 {
    2.0 * aperture * aperture * frequency / C0
}

/// A link budget in decibels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkBudget {
    pub transmit_power_dbm: f64,
    pub transmit_gain_dbi: f64,
    pub receive_gain_dbi: f64,
    /// Extra losses (cables, polarization, fading margin), dB.
    pub other_losses_db: f64,
}

impl LinkBudget {
    /// The received power in dBm for the given path loss (dB).
    pub fn received_power_dbm(&self, path_loss_db: f64) -> f64 {
        self.transmit_power_dbm + self.transmit_gain_dbi + self.receive_gain_dbi - path_loss_db - self.other_losses_db
    }

    /// The largest free-space range (m) at which the received power reaches `sensitivity_dbm`.
    pub fn free_space_range(&self, sensitivity_dbm: f64, frequency: f64) -> f64 {
        let allowed = self.transmit_power_dbm + self.transmit_gain_dbi + self.receive_gain_dbi - self.other_losses_db - sensitivity_dbm;
        10f64.powf(allowed / 20.0) * C0 / (4.0 * PI * frequency)
    }
}

/// The log-distance model `PL(d) = PL(d₀) + 10 n log₁₀(d/d₀)` with path loss exponent `n`
/// (2 in free space, 2.7 to 3.5 in urban areas, 4 to 6 in buildings).
pub fn log_distance_path_loss_db(distance: f64, reference_distance: f64, reference_loss_db: f64, exponent: f64) -> f64 {
    reference_loss_db + 10.0 * exponent * (distance / reference_distance).log10()
}

/// Environment for the Okumura-Hata model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HataEnvironment {
    /// Urban, small or medium city.
    Urban,
    /// Urban, large city.
    LargeCity,
    Suburban,
    OpenArea,
}

/// The Okumura-Hata median path loss (dB) for `150 ≤ f ≤ 1500` MHz, base station height
/// `30–200` m, mobile height `1–10` m and distance `1–20` km.
pub fn okumura_hata_db(frequency_mhz: f64, base_height: f64, mobile_height: f64, distance_km: f64, environment: HataEnvironment) -> Result<f64, ErrorsJSL> {
    if !(150.0..=1500.0).contains(&frequency_mhz) || !(30.0..=200.0).contains(&base_height) || !(1.0..=10.0).contains(&mobile_height) || !(1.0..=20.0).contains(&distance_km) {
        return Err(ErrorsJSL::InvalidInputRange("Outside the validity range of the Okumura-Hata model."));
    }
    let lf = frequency_mhz.log10();
    let correction = match environment {
        HataEnvironment::LargeCity if frequency_mhz >= 300.0 => 3.2 * (11.75 * mobile_height).log10().powi(2) - 4.97,
        HataEnvironment::LargeCity => 8.29 * (1.54 * mobile_height).log10().powi(2) - 1.1,
        _ => (1.1 * lf - 0.7) * mobile_height - (1.56 * lf - 0.8),
    };
    let urban = 69.55 + 26.16 * lf - 13.82 * base_height.log10() - correction + (44.9 - 6.55 * base_height.log10()) * distance_km.log10();
    Ok(match environment {
        HataEnvironment::Urban | HataEnvironment::LargeCity => urban,
        HataEnvironment::Suburban => urban - 2.0 * (frequency_mhz / 28.0).log10().powi(2) - 5.4,
        HataEnvironment::OpenArea => urban - 4.78 * lf * lf + 18.33 * lf - 40.94,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_space_loss_and_friis_agree() {
        assert!((free_space_path_loss_db(1000.0, 1e9) - 92.45).abs() < 0.01);
        let pr = friis_received_power(1.0, 1.0, 1.0, 1e9, 1000.0).unwrap();
        assert!((to_db(pr) + free_space_path_loss_db(1000.0, 1e9)).abs() < 1e-9);
        // Doubling the distance costs 6 dB.
        assert!((free_space_path_loss_db(2000.0, 1e9) - free_space_path_loss_db(1000.0, 1e9) - 6.0206).abs() < 1e-3);
        let budget = LinkBudget { transmit_power_dbm: 30.0, transmit_gain_dbi: 10.0, receive_gain_dbi: 3.0, other_losses_db: 2.0 };
        let range = budget.free_space_range(-90.0, 2.4e9);
        assert!((budget.received_power_dbm(free_space_path_loss_db(range, 2.4e9)) + 90.0).abs() < 1e-9);
        assert!((fraunhofer_distance(1.0, 3e9) - 2.0 / (C0 / 3e9)).abs() < 1e-12);
        assert!(friis_received_power(1.0, 1.0, 1.0, 1e9, 0.0).is_err());
        assert!((from_db(to_db(37.0)) - 37.0).abs() < 1e-12);
    }

    #[test]
    fn empirical_models_match_reference_values() {
        // 900 MHz, 30 m base, 1.5 m mobile, 1 km, medium city: 126.4 dB.
        let urban = okumura_hata_db(900.0, 30.0, 1.5, 1.0, HataEnvironment::Urban).unwrap();
        assert!((urban - 126.42).abs() < 0.05, "{urban}");
        let suburban = okumura_hata_db(900.0, 30.0, 1.5, 1.0, HataEnvironment::Suburban).unwrap();
        let open = okumura_hata_db(900.0, 30.0, 1.5, 1.0, HataEnvironment::OpenArea).unwrap();
        assert!(open < suburban && suburban < urban);
        // The urban distance slope is 35.2 dB per decade at a 30 m base height.
        let ten = okumura_hata_db(900.0, 30.0, 1.5, 10.0, HataEnvironment::Urban).unwrap();
        assert!((ten - urban - (44.9 - 6.55 * 30f64.log10())).abs() < 1e-9);
        assert!(okumura_hata_db(2400.0, 30.0, 1.5, 1.0, HataEnvironment::Urban).is_err());
        assert!((log_distance_path_loss_db(100.0, 1.0, 40.0, 3.0) - 100.0).abs() < 1e-12);
    }
}
