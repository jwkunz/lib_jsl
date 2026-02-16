/// This is Direct Form II Transposed IIR Filter implementation. It is a common structure for implementing IIR filters, where the feed-forward (numerator) coefficients are applied directly to the input samples, and the feedback (denominator) coefficients are applied to the output samples.
/// The implementation uses a delay line to store intermediate values, and the `step` method processes each input sample to produce the corresponding output sample.
/// The `process` method allows for processing a block of samples at once, and the `reset` method clears the internal state of the filter.
use std::collections::VecDeque;

use num::Complex;

use crate::{
    dsp::stream_operator::{StreamOperator, StreamOperatorManagement},
    prelude::{ErrorsJSL, IsLinearOperatable},
};

pub struct DiscreteLinearFilter<T : IsLinearOperatable> {
    /// Feed-forward (numerator) coefficients
    b: Vec<T>,

    /// Feedback (denominator) coefficients; a[0] must be 1
    a: Vec<T>,

    /// Delay line of size = max(b.len(), a.len()) - 1
    delay: VecDeque<T>,
}

impl<T : IsLinearOperatable> DiscreteLinearFilter<T> {
    /// Create a new IIR DF2 filter from b[] and a[] taps.
    /// a[0] must equal 1.0.
    pub fn new(b: &[T], a: &[T]) -> Self {
        let order = usize::max(b.len(), a.len()).saturating_sub(1);
        Self {
            b: b.to_vec(),
            a: a.to_vec(),
            delay: VecDeque::from(vec![T::zero(); order]),
        }
    }

    /// Process a single sample
    pub fn step(&mut self, sample: &T) -> T {
        let order = self.delay.len();
        // d[0] is added directly to output
        let d0 = if order > 0 { self.delay[0] } else { T::zero() };

        // Output
        let y = self.b[0] * *sample + d0;

        // Update delay registers
        // d[k] = d[k+1] + b[k+1]*x − a[k+1]*y
        for k in 0..order {
            let next_d = if k + 1 < order {
                self.delay[k + 1]
            } else {
                T::zero()
            };

            // b[k+1] or 0
            let bkp1 = if k + 1 < self.b.len() {
                self.b[k + 1]
            } else {
                T::zero()
            };

            // a[k+1] or 0
            let akp1 = if k + 1 < self.a.len() {
                self.a[k + 1]
            } else {
                T::zero()
            };

            self.delay[k] = next_d + bkp1 * *sample - akp1 * y;
        }
        y
    }

    pub fn longest_buffer_length(&self) -> usize {
        self.delay.len()
    }

    pub fn clear_buffer(&mut self) {
        for i in self.delay.iter_mut() {
            *i = T::zero();
        }
    }
}

/// For real-valued filters, we can use the same implementation but with f64 as the type parameter. This allows us to reuse the same code for both real and complex filters, while still providing the necessary functionality for each type.

impl StreamOperatorManagement for DiscreteLinearFilter<f64> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.clear_buffer();
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<f64, f64> for DiscreteLinearFilter<f64> {
    fn process(&mut self, samples: &[f64]) -> Result<Option<Vec<f64>>, ErrorsJSL> {
        Ok(Some(samples.iter().map(|x| self.step(x)).collect()))
    }
    fn flush(&mut self) -> Result<Option<Vec<f64>>, ErrorsJSL> {
        let zeros = vec![0.0; self.longest_buffer_length()];
        self.process(&zeros)
    }
}

/// For complex-valued filters, we can use Complex<f64> as the type parameter. This allows us to handle complex input and output samples, which is common in many DSP applications such as modulation, demodulation, and spectral processing.

impl StreamOperatorManagement for DiscreteLinearFilter<Complex<f64>> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.clear_buffer();
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<Complex<f64>, Complex<f64>> for DiscreteLinearFilter<Complex<f64>> {
    fn process(&mut self, samples: &[Complex<f64>]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if samples.is_empty() {
            return Ok(None);
        }
        Ok(Some(samples.iter().map(|x| self.step(x)).collect()))
    }
    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        let zeros = vec![Complex::new(0.0, 0.0); self.longest_buffer_length()];
        self.process(&zeros)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple smoke test: identity-like filter (tap = [1]) with L=1, M=1 should forward samples.
    #[test]
    fn test_linear_filter() {
        let b = [0.5, 0.25, 0.1];
        let a = [
            1.0, // must be 1.0
            -0.3,
        ];

        let mut iir = DiscreteLinearFilter::new(&b, &a);

        let input = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let output = iir.process(&input).unwrap().unwrap();
        let golden = vec![0.0, 0.5, 1.4, 2.52, 3.706, 4.9117999999999995, 6.12354];
        assert_eq!(output, golden)
    }
}
