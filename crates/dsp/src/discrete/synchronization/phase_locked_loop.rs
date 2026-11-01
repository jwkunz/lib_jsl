//! A digital phase-locked loop with a second-order (proportional-integral) loop filter.

use lib_jsl_core::ErrorsJSL;
use num::Complex;
use std::f64::consts::{PI, TAU};

type C64 = Complex<f64>;

/// Proportional-integral loop filter and NCO shared by the carrier-tracking loops.
///
/// Gains follow the standard design for a type-2 loop with unit detector and NCO gains (Rice,
/// *Digital Communications*, appendix C): with `θ = B_n T / (ζ + 1/(4ζ))`,
/// `K_p = 4ζθ / (1 + 2ζθ + θ²)` and `K_i = 4θ² / (1 + 2ζθ + θ²)`. Such a loop tracks a phase and a
/// constant frequency offset with zero steady-state error.
#[derive(Debug, Clone)]
pub struct LoopFilter {
    proportional: f64,
    integral: f64,
    /// NCO phase (radians).
    phase: f64,
    /// Integrator state: the frequency estimate (radians per sample).
    frequency: f64,
}

impl LoopFilter {
    /// `normalized_bandwidth` is the noise bandwidth times the sample period (`B_n T`, typically
    /// 0.001 to 0.05); `damping` is the damping factor ζ (0.707 is the usual choice).
    pub fn new(normalized_bandwidth: f64, damping: f64) -> Result<Self, ErrorsJSL> {
        if !(normalized_bandwidth > 0.0 && normalized_bandwidth < 0.5) || !(damping > 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("Need 0 < BnT < 0.5 and a positive damping factor."));
        }
        let theta = normalized_bandwidth / (damping + 0.25 / damping);
        let denominator = 1.0 + 2.0 * damping * theta + theta * theta;
        Ok(LoopFilter { proportional: 4.0 * damping * theta / denominator, integral: 4.0 * theta * theta / denominator, phase: 0.0, frequency: 0.0 })
    }

    /// Advances the NCO with the detector error (radians).
    pub fn update(&mut self, error: f64) {
        self.frequency += self.integral * error;
        self.phase = (self.phase + self.proportional * error + self.frequency + PI).rem_euclid(TAU) - PI;
    }

    pub fn phase(&self) -> f64 {
        self.phase
    }

    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Presets the frequency estimate (radians per sample), e.g. from a coarse estimator.
    pub fn set_frequency(&mut self, frequency: f64) {
        self.frequency = frequency;
    }
}

/// A phase-locked loop that locks a numerically controlled oscillator to a complex tone. The
/// phase detector is `arg(x e^{−jφ})`, exact for all phase errors, so the loop pulls in from any
/// initial phase; frequency offsets well inside the loop bandwidth are acquired directly.
#[derive(Debug, Clone)]
pub struct PhaseLockedLoop {
    filter: LoopFilter,
}

impl PhaseLockedLoop {
    pub fn new(normalized_bandwidth: f64, damping: f64) -> Result<Self, ErrorsJSL> {
        Ok(PhaseLockedLoop { filter: LoopFilter::new(normalized_bandwidth, damping)? })
    }

    /// Processes one sample: returns the input derotated by the NCO and the phase error.
    pub fn step(&mut self, x: C64) -> (C64, f64) {
        let y = x * C64::from_polar(1.0, -self.filter.phase);
        let error = if y.norm() > 0.0 { y.arg() } else { 0.0 };
        self.filter.update(error);
        (y, error)
    }

    /// Derotates a block.
    pub fn process(&mut self, x: &[C64]) -> Vec<C64> {
        x.iter().map(|&v| self.step(v).0).collect()
    }

    pub fn phase(&self) -> f64 {
        self.filter.phase()
    }

    /// The tracked frequency in radians per sample.
    pub fn frequency(&self) -> f64 {
        self.filter.frequency()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_to_phase_and_frequency_offsets() {
        let (omega, phi0) = (0.03, 2.5);
        let mut pll = PhaseLockedLoop::new(0.02, 0.707).unwrap();
        let x: Vec<C64> = (0..3000).map(|n| C64::from_polar(1.0, omega * n as f64 + phi0)).collect();
        let y = pll.process(&x);
        assert!((pll.frequency() - omega).abs() < 1e-9, "{}", pll.frequency());
        assert!(y[2900..].iter().all(|v| v.arg().abs() < 1e-6));
    }

    #[test]
    fn noise_jitter_scales_with_bandwidth_and_validation() {
        // Phase noise on the tone: the narrower loop leaves less jitter on its phase estimate.
        let mut s: u64 = 7;
        let mut uniform = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let x: Vec<C64> = (0..20_000).map(|n| C64::from_polar(1.0, 0.01 * n as f64 + 0.5 * uniform())).collect();
        let jitter = |bandwidth: f64| {
            let mut pll = PhaseLockedLoop::new(bandwidth, 0.707).unwrap();
            pll.process(&x[..5000]);
            let errors: Vec<f64> = (5000..20_000).map(|n| { pll.step(x[n]); ((pll.phase() - 0.01 * (n + 1) as f64 + PI).rem_euclid(TAU) - PI).powi(2) }).collect();
            (errors.iter().sum::<f64>() / errors.len() as f64).sqrt()
        };
        let (wide, narrow) = (jitter(0.05), jitter(0.005));
        assert!(narrow < 0.5 * wide, "{narrow} vs {wide}");
        assert!(PhaseLockedLoop::new(0.0, 0.7).is_err() && PhaseLockedLoop::new(0.01, -1.0).is_err());
    }
}
