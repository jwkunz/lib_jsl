//! Costas loops: carrier recovery for suppressed-carrier BPSK and QPSK.

use super::phase_locked_loop::LoopFilter;
use lib_jsl_core::ErrorsJSL;
use num_complex::Complex;

type C64 = Complex<f64>;

/// The modulation whose carrier the loop recovers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostasModulation {
    /// Detector `sgn(I)·Q`: insensitive to the ±1 data, locks with a 180° ambiguity.
    Bpsk,
    /// Detector `sgn(I)·Q − sgn(Q)·I`: insensitive to the four symbols, locks with a 90°
    /// ambiguity (resolve it with differential coding or a known preamble).
    Qpsk,
}

/// A decision-directed Costas loop. Symbols (one complex sample per symbol, timing already
/// recovered) are derotated by the NCO, and the data-stripping phase detector, applied to the
/// unit-magnitude derotated sample (so the detector gain is one near lock), drives the same
/// second-order loop filter as [`super::phase_locked_loop::PhaseLockedLoop`].
#[derive(Debug, Clone)]
pub struct CostasLoop {
    filter: LoopFilter,
    modulation: CostasModulation,
}

impl CostasLoop {
    pub fn new(modulation: CostasModulation, normalized_bandwidth: f64, damping: f64) -> Result<Self, ErrorsJSL> {
        Ok(CostasLoop { filter: LoopFilter::new(normalized_bandwidth, damping)?, modulation })
    }

    /// Processes one symbol: returns the derotated symbol and the detector error.
    pub fn step(&mut self, x: C64) -> (C64, f64) {
        let y = x * C64::from_polar(1.0, -self.filter.phase());
        let magnitude = y.norm();
        let error = if magnitude > 0.0 {
            let u = y / magnitude;
            match self.modulation {
                CostasModulation::Bpsk => u.re.signum() * u.im,
                CostasModulation::Qpsk => (u.re.signum() * u.im - u.im.signum() * u.re) / std::f64::consts::SQRT_2,
            }
        } else {
            0.0
        };
        self.filter.update(error);
        (y, error)
    }

    pub fn process(&mut self, x: &[C64]) -> Vec<C64> {
        x.iter().map(|&v| self.step(v).0).collect()
    }

    pub fn phase(&self) -> f64 {
        self.filter.phase()
    }

    pub fn frequency(&self) -> f64 {
        self.filter.frequency()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_4;

    fn bits(n: usize, seed: u64) -> Vec<bool> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                s >> 63 == 1
            })
            .collect()
    }

    #[test]
    fn recovers_bpsk_with_a_frequency_offset() {
        let data = bits(6000, 1);
        let (omega, phi0) = (0.004, 0.9);
        let x: Vec<C64> = data.iter().enumerate().map(|(n, &b)| C64::from_polar(if b { 1.0 } else { -1.0 }, omega * n as f64 + phi0)).collect();
        let mut costas = CostasLoop::new(CostasModulation::Bpsk, 0.01, 0.707).unwrap();
        let y = costas.process(&x);
        assert!((costas.frequency() - omega).abs() < 1e-6, "{}", costas.frequency());
        // After lock the symbols lie on the real axis; the sign is ambiguous by 180°.
        let tail = 5000..6000;
        assert!(tail.clone().all(|n| y[n].im.abs() < 1e-4));
        let agree = tail.clone().filter(|&n| (y[n].re > 0.0) == data[n]).count();
        assert!(agree == 1000 || agree == 0, "{agree}");
    }

    #[test]
    fn recovers_qpsk_up_to_a_quarter_turn_and_validates() {
        let (a, b) = (bits(8000, 2), bits(8000, 3));
        let symbols: Vec<C64> = a.iter().zip(&b).map(|(&i, &q)| C64::new(if i { 1.0 } else { -1.0 }, if q { 1.0 } else { -1.0 }) / 2f64.sqrt()).collect();
        let x: Vec<C64> = symbols.iter().enumerate().map(|(n, s)| s * C64::from_polar(1.0, 0.002 * n as f64 - 0.4)).collect();
        let mut costas = CostasLoop::new(CostasModulation::Qpsk, 0.005, 0.707).unwrap();
        let y = costas.process(&x);
        // Every recovered symbol sits on a diagonal, and one fixed rotation by kπ/2 maps them to
        // the transmitted ones.
        assert!(y[7000..].iter().all(|v| ((v.arg() - FRAC_PI_4).rem_euclid(std::f64::consts::FRAC_PI_2)).min(std::f64::consts::FRAC_PI_2 - (v.arg() - FRAC_PI_4).rem_euclid(std::f64::consts::FRAC_PI_2)) < 1e-3));
        let rotation = (0..4).find(|&k| (7000..8000).all(|n| (y[n] * C64::from_polar(1.0, k as f64 * std::f64::consts::FRAC_PI_2) - symbols[n]).norm() < 1e-2));
        assert!(rotation.is_some());
        assert!(CostasLoop::new(CostasModulation::Bpsk, 0.6, 0.7).is_err());
    }
}
