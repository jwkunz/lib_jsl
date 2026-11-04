//! Multipath propagation: the two-ray ground reflection model, Fresnel reflection, Fresnel zones
//! and tapped-delay-line channels.
//!
//! See the book entry [Propagation](https://jwkunz.github.io/lib_jsl/book/em-waves/propagation.html).

use crate::conventions::C64;
use lib_jsl_core::ErrorsJSL;
use lib_jsl_core::constants::em::C0;
use std::f64::consts::{PI, TAU};

/// The polarization of a wave reflecting from a plane boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflectionPolarization {
    /// Electric field perpendicular to the plane of incidence (TE, "horizontal" over ground).
    Perpendicular,
    /// Electric field in the plane of incidence (TM, "vertical" over ground).
    Parallel,
}

/// The Fresnel reflection coefficient from vacuum onto a half-space of complex relative
/// permittivity `epsilon` (`ε' − jε''`, engineering convention) at grazing angle `psi` (radians
/// above the surface).
pub fn fresnel_reflection(epsilon: C64, grazing_angle: f64, polarization: ReflectionPolarization) -> C64 {
    let (s, c) = grazing_angle.sin_cos();
    let root = (epsilon - c * c).sqrt();
    match polarization {
        ReflectionPolarization::Perpendicular => (s - root) / (s + root),
        ReflectionPolarization::Parallel => (epsilon * s - root) / (epsilon * s + root),
    }
}

/// Ground conductivity and permittivity as a complex relative permittivity `ε_r − j σ/(ωε₀)`.
pub fn ground_permittivity(relative_permittivity: f64, conductivity: f64, frequency: f64) -> C64 {
    C64::new(relative_permittivity, -conductivity / (TAU * frequency * lib_jsl_core::constants::em::EPS_0))
}

/// The two-ray model: the field of the direct ray plus the ground-reflected ray between antennas
/// at heights `ht`, `hr` (m) a ground distance `d` apart, relative to the direct ray alone in free
/// space: `1 + Γ (r₁/r₂) e^{−jk(r₂ − r₁)}`. The received power is the free-space power times
/// `|factor|²`. With `reflection = None`, a perfect reflector `Γ = −1` is assumed; otherwise the
/// Fresnel coefficient of the given ground is used.
pub fn two_ray_factor(distance: f64, ht: f64, hr: f64, frequency: f64, ground: Option<(C64, ReflectionPolarization)>) -> Result<C64, ErrorsJSL> {
    if !(distance > 0.0) || ht < 0.0 || hr < 0.0 || !(frequency > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Need a positive distance and frequency and non-negative heights."));
    }
    // ANCHOR: two_ray
    let r1 = (distance * distance + (ht - hr).powi(2)).sqrt();
    let r2 = (distance * distance + (ht + hr).powi(2)).sqrt();
    let grazing = ((ht + hr) / distance).atan();
    let gamma = match ground {
        None => C64::new(-1.0, 0.0),
        Some((eps, pol)) => fresnel_reflection(eps, grazing, pol),
    };
    let k = TAU * frequency / C0;
    Ok(1.0 + gamma * (r1 / r2) * C64::from_polar(1.0, -k * (r2 - r1)))
    // ANCHOR_END: two_ray
}

/// The large-distance two-ray approximation of the received power,
/// `P_t G_t G_r h_t² h_r² / d⁴` (independent of frequency).
pub fn two_ray_asymptotic_power(transmit_power: f64, gt: f64, gr: f64, ht: f64, hr: f64, distance: f64) -> f64 {
    transmit_power * gt * gr * (ht * hr).powi(2) / distance.powi(4)
}

/// The crossover distance `4π h_t h_r / λ` beyond which the two-ray power falls as `d⁻⁴`.
pub fn two_ray_crossover_distance(ht: f64, hr: f64, frequency: f64) -> f64 {
    4.0 * PI * ht * hr * frequency / C0
}

/// The radius of the `n`-th Fresnel zone at distances `d1`, `d2` from the two ends:
/// `√(n λ d₁ d₂ / (d₁ + d₂))`.
pub fn fresnel_zone_radius(n: u32, d1: f64, d2: f64, frequency: f64) -> f64 {
    (n as f64 * C0 / frequency * d1 * d2 / (d1 + d2)).sqrt()
}

/// A tapped-delay-line multipath channel `h(t) = Σ a_i δ(t − τ_i)`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultipathChannel {
    /// Path delays (s).
    pub delays: Vec<f64>,
    /// Complex path amplitudes.
    pub gains: Vec<C64>,
}

impl MultipathChannel {
    pub fn new(delays: Vec<f64>, gains: Vec<C64>) -> Result<Self, ErrorsJSL> {
        if delays.is_empty() || delays.len() != gains.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((delays.len(), gains.len())));
        }
        Ok(MultipathChannel { delays, gains })
    }

    /// The frequency response `H(f) = Σ a_i e^{−j2πfτ_i}`.
    pub fn frequency_response(&self, frequency: f64) -> C64 {
        self.delays.iter().zip(&self.gains).map(|(&t, &a)| a * C64::from_polar(1.0, -TAU * frequency * t)).sum()
    }

    /// The power-weighted mean delay and the RMS delay spread (s).
    pub fn delay_statistics(&self) -> (f64, f64) {
        let p: Vec<f64> = self.gains.iter().map(|g| g.norm_sqr()).collect();
        let total: f64 = p.iter().sum();
        let mean = self.delays.iter().zip(&p).map(|(t, w)| t * w).sum::<f64>() / total;
        let second = self.delays.iter().zip(&p).map(|(t, w)| t * t * w).sum::<f64>() / total;
        (mean, (second - mean * mean).max(0.0).sqrt())
    }

    /// The 50% coherence bandwidth estimate `1 / (5 σ_τ)` (Hz).
    pub fn coherence_bandwidth(&self) -> f64 {
        1.0 / (5.0 * self.delay_statistics().1)
    }

    /// Filters complex baseband samples at rate `fs` through the channel (delays rounded to the
    /// nearest sample).
    pub fn apply(&self, x: &[C64], fs: f64) -> Vec<C64> {
        let mut y = vec![C64::new(0.0, 0.0); x.len()];
        for (&t, &a) in self.delays.iter().zip(&self.gains) {
            let shift = (t * fs).round() as usize;
            for n in shift..x.len() {
                y[n] += a * x[n - shift];
            }
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresnel_coefficients_have_the_textbook_limits() {
        let eps = C64::new(4.0, 0.0);
        // Normal incidence: (1 − √ε)/(1 + √ε) = −1/3 for perpendicular, +1/3 for parallel.
        let normal = std::f64::consts::FRAC_PI_2;
        assert!((fresnel_reflection(eps, normal, ReflectionPolarization::Perpendicular) - C64::new(-1.0 / 3.0, 0.0)).norm() < 1e-12);
        assert!((fresnel_reflection(eps, normal, ReflectionPolarization::Parallel) - C64::new(1.0 / 3.0, 0.0)).norm() < 1e-12);
        // Brewster: the parallel coefficient vanishes at grazing angle atan(1/√ε) from the surface normal complement.
        let brewster_grazing = std::f64::consts::FRAC_PI_2 - 2f64.atan();
        assert!(fresnel_reflection(eps, brewster_grazing, ReflectionPolarization::Parallel).norm() < 1e-12);
        // Grazing incidence reflects totally with a phase reversal.
        assert!((fresnel_reflection(eps, 1e-9, ReflectionPolarization::Perpendicular) + 1.0).norm() < 1e-8);
        let soil = ground_permittivity(15.0, 0.005, 1e8);
        assert!(soil.im < 0.0 && (soil.im + 0.005 / (TAU * 1e8 * lib_jsl_core::constants::em::EPS_0)).abs() < 1e-12);
    }

    #[test]
    fn two_ray_model_reaches_the_fourth_power_law() {
        let (ht, hr, f) = (30.0, 2.0, 900e6);
        let crossover = two_ray_crossover_distance(ht, hr, f);
        assert!((crossover - 4.0 * PI * 60.0 / (C0 / f)).abs() < 1e-9);
        let d = 50.0 * crossover;
        let factor = two_ray_factor(d, ht, hr, f, None).unwrap().norm_sqr();
        let free = crate::propagation::path_loss::friis_received_power(1.0, 1.0, 1.0, f, d).unwrap();
        let asymptotic = two_ray_asymptotic_power(1.0, 1.0, 1.0, ht, hr, d);
        assert!((free * factor / asymptotic - 1.0).abs() < 0.01, "{}", free * factor / asymptotic);
        // Near the antennas the two rays interfere: nulls dip far below free space.
        let dips = (100..2000).map(|i| two_ray_factor(i as f64, ht, hr, f, None).unwrap().norm()).fold(f64::INFINITY, f64::min);
        assert!(dips < 0.05);
        let ground = Some((ground_permittivity(15.0, 0.005, f), ReflectionPolarization::Parallel));
        assert!(two_ray_factor(1000.0, ht, hr, f, ground).unwrap().norm() > 0.0);
        assert!((fresnel_zone_radius(1, 5000.0, 5000.0, 3e9) - (C0 / 3e9 * 2500.0).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn delay_spread_and_channel_filtering() {
        // Two equal paths 1 µs apart: mean 0.5 µs, spread 0.5 µs, notches every 1 MHz.
        let ch = MultipathChannel::new(vec![0.0, 1e-6], vec![C64::new(1.0, 0.0), C64::new(1.0, 0.0)]).unwrap();
        let (mean, spread) = ch.delay_statistics();
        assert!((mean - 0.5e-6).abs() < 1e-18 && (spread - 0.5e-6).abs() < 1e-18);
        assert!(ch.frequency_response(0.5e6).norm() < 1e-9 && (ch.frequency_response(1e6).norm() - 2.0).abs() < 1e-9);
        assert!((ch.coherence_bandwidth() - 400e3).abs() < 1e-6);
        let y = ch.apply(&[C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)], 2e6);
        assert_eq!(y, vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)]);
        assert!(MultipathChannel::new(vec![0.0], vec![]).is_err());
    }
}
