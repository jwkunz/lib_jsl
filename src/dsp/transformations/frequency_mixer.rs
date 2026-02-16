use std::f64::consts::PI;

use num::Complex;

/// A frequency mixer is used to frequency shift a signal by multiplying it with a complex exponential. 
/// This operation is commonly used in applications such as modulation, demodulation, and frequency translation in communication systems. 
/// The `FrequencyMixer` struct maintains the current phase of the complex exponential and the normalized frequency increment, which determines how much the phase is advanced for each sample processed.
/// The `process` method takes an input signal (either real or complex) and multiplies each sample by the corresponding value of the complex exponential based on the current phase. 
/// The phase is then incremented for the next sample, ensuring that the frequency shift is applied consistently across the entire signal.

use crate::prelude::{ErrorsJSL};
use crate::dsp::stream_operator::{StreamOperator, StreamOperatorManagement};

pub struct FrequencyMixer {
    phase: f64,
    frequency_increment : f64,
}

impl FrequencyMixer {
    /// The `new` method initializes a `FrequencyMixer` instance with the specified frequency in Hz, sample rate in Hz, and initial phase in radians.
    /// The `frequency_increment_normalized` is calculated as the ratio of the desired frequency shift to the sample rate, which determines how much the phase will be incremented for each sample processed.
    /// The `initial_phase` parameter allows you to set the starting phase of the complex exponential, which can be useful for synchronizing the mixer with other components in a signal processing chain or for achieving specific phase relationships in modulation schemes.
    pub fn new(frequency_hz: f64, sample_rate_hz: f64, initial_phase: Option<f64>) -> Self {
        Self { 
            phase: initial_phase.unwrap_or(0.0),
            frequency_increment: 2.0*PI*frequency_hz / sample_rate_hz,
        }
    }

    // Increment the phase with wrapping to keep it in the range [0, 2*PI) for positive frequencies or (-2*PI, 0] for negative frequencies. This ensures that the phase remains bounded and prevents numerical issues that can arise from unbounded phase growth.
    fn increment_phase(&mut self) {
        self.phase += self.frequency_increment;
        if self.frequency_increment > 0.0 {
            if self.phase > 2.0*PI {
                self.phase -= 2.0*PI; // Wrap around to keep phase in [0, 2*PI)
            }
        }else{
            if self.phase < 0.0 {
                self.phase += 2.0*PI; // Wrap around to keep phase in [0, 2*PI)
            }
        }

    }
}

impl StreamOperatorManagement for FrequencyMixer {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.phase = 0.0;
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

/// This is a real-valued mixer that produces a real-valued output by multiplying the input signal with the cosine of the current phase.
/// This type of mixer is commonly used for applications such as amplitude modulation (AM) or for down-converting a signal to baseband in a receiver.
impl StreamOperator<f64, f64> for FrequencyMixer {
    fn process(&mut self, input: &[f64]) -> Result<Option<Vec<f64>>, ErrorsJSL> {
        if input.is_empty() {
            return Ok(None);
        }
        input.iter().map(|&x| {
            let mixer = self.phase.cos();
            self.increment_phase();
            Ok(x * mixer)
        }).collect::<Result<Vec<f64>, ErrorsJSL>>().map(Some)
    }

    fn flush(&mut self) -> Result<Option<Vec<f64>>, ErrorsJSL> {
        Ok(None)
    }
}

/// This is a complex-valued mixer that produces a complex-valued output by multiplying the input signal with a complex exponential based on the current phase.
/// This type of mixer is commonly used for applications such as quadrature amplitude modulation (QAM) or for up-converting a baseband signal to a higher frequency in a transmitter.   
impl StreamOperator<f64, Complex<f64>> for FrequencyMixer {
    fn process(&mut self, input: &[f64]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if input.is_empty() {
            return Ok(None);
        }
        input.iter().map(|&x| {
            let mixer = Complex::from_polar(1.0, self.phase);
            self.increment_phase();
            Ok(x * mixer)
        }).collect::<Result<Vec<Complex<f64>>, ErrorsJSL>>().map(Some)
    }

    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        Ok(None)
    }
}

/// This is a complex-valued mixer that takes a complex-valued input signal and produces a complex-valued output by multiplying the input signal with a complex exponential based on the current phase.
/// This type of mixer is commonly used for applications such as quadrature amplitude modulation (QAM) or for up-converting a baseband signal to a higher frequency in a transmitter, as well as for down-converting a signal to baseband in a receiver. 
/// It can also be used for frequency translation in spectral processing applications, such as shifting the frequency of a signal for analysis or filtering purposes
impl StreamOperator<Complex<f64>, Complex<f64>> for FrequencyMixer {
    fn process(&mut self, input: &[Complex<f64>]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if input.is_empty() {
            return Ok(None);
        }
        input.iter().map(|&x| {
            let mixer = Complex::from_polar(1.0, self.phase);
            self.increment_phase();
            Ok(x * mixer)
        }).collect::<Result<Vec<Complex<f64>>, ErrorsJSL>>().map(Some)
    }

    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn approx_eq(a: Complex<f64>, b: Complex<f64>, tol: f64) -> bool {
        (a - b).norm() < tol
    }
    #[test]
    fn test_mixer(){
        let mut dut = FrequencyMixer::new(1000.0, 8000.0, None);
        let input = vec![1.0; 8];
        let output = dut.process(&input).unwrap().unwrap();
        let expected = input.iter().enumerate().map(|(n, &x)| {
            let phase = dut.frequency_increment * (n) as f64;
            x * Complex::from_polar(1.0, phase)
        }).collect::<Vec<Complex<f64>>>();
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!(approx_eq(*o, *e, 1e-6), "Output {o} is not approximately equal to expected {e}");
        }   
    }
}