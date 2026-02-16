/// A Biquad filter implementation for scalar streams, using direct form I. Coefficients are based on the RBJ cookbook formulas.
/// This is a simple implementation for demonstration and may not be the most efficient or numerically stable for all use cases. For more complex filters or higher performance requirements, consider using a dedicated DSP library or implementing in a lower-level language with SIMD support.
/// Example usage:
/// ```
/// let mut biquad = BiquadFilter::lowpass(0.1).unwrap();
/// let input = vec![1.0; 256];
/// let output = biquad.process(&input).unwrap().unwrap();
use std::f64::consts::PI;

use crate::{
    dsp::stream_operator::{StreamOperator, StreamOperatorManagement},
    prelude::ErrorsJSL,
};

#[derive(Clone, Copy, Debug)]
pub struct BiquadCoefficients {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl BiquadCoefficients {
    /// RBJ cookbook low-pass coefficients with normalized cutoff in (0, 0.5).
    pub fn lowpass(cutoff_normalized: f64, q: f64) -> Result<Self, ErrorsJSL> {
        if !(0.0..0.5).contains(&cutoff_normalized) {
            return Err(ErrorsJSL::InvalidInputRange(
                "cutoff_normalized must be in (0, 0.5)",
            ));
        }
        if !q.is_finite() || q <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("q must be finite and > 0"));
        }
        let w0 = 2.0 * PI * cutoff_normalized;
        let cw = w0.cos();
        let sw = w0.sin();
        let alpha = sw / (2.0 * q);

        let b0 = (1.0 - cw) * 0.5;
        let b1 = 1.0 - cw;
        let b2 = (1.0 - cw) * 0.5;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cw;
        let a2 = 1.0 - alpha;

        Ok(Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        })
    }
}

/// Direct-form biquad filter for scalar streams.
pub struct BiquadFilter {
    coeffs: BiquadCoefficients,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl BiquadFilter {
    pub fn new(coeffs: BiquadCoefficients) -> Self {
        Self {
            coeffs,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    pub fn lowpass(cutoff_normalized: f64) -> Result<Self, ErrorsJSL> {
        let q = 1.0 / 2.0_f64.sqrt();
        Ok(Self::new(BiquadCoefficients::lowpass(cutoff_normalized, q)?))
    }

    pub fn step(&mut self, x: f64) -> f64 {
        let c = self.coeffs;
        let y = c.b0 * x + c.b1 * self.x1 + c.b2 * self.x2 - c.a1 * self.y1 - c.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

impl StreamOperatorManagement for BiquadFilter {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<f64, f64> for BiquadFilter {
    fn process(&mut self, data_in: &[f64]) -> Result<Option<Vec<f64>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        Ok(Some(data_in.iter().map(|&x| self.step(x)).collect()))
    }

    fn flush(&mut self) -> Result<Option<Vec<f64>>, ErrorsJSL> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquad_lowpass_dc_converges() {
        let mut dut = BiquadFilter::lowpass(0.1).unwrap();
        let y = dut.process(&vec![1.0; 256]).unwrap().unwrap();
        assert!((y.last().copied().unwrap_or(0.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_biquad_reset() {
        let mut dut = BiquadFilter::lowpass(0.1).unwrap();
        let _ = dut.process(&[1.0, 1.0, 1.0]).unwrap();
        dut.reset().unwrap();
        let y = dut.process(&[0.0]).unwrap().unwrap();
        assert!(y[0].abs() < 1e-12);
    }
}

