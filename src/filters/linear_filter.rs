use std::collections::VecDeque;

use crate::prelude::ErrorsJSL;

/// ---------------------------------------------------------------------------
/// Direct Form II Transposed IIR Filter
/// ---------------------------------------------------------------------------
/// Uses `b` (numerator) and `a` (denominator) taps.
/// a[0] must be 1.0.
/// Output is a single sample per input (rate = 1), so step() returns either
/// {Some(vec![y])} or None.
/// ---------------------------------------------------------------------------
pub struct LinearFilter {
    /// Feed-forward (numerator) coefficients
    b: Vec<f64>,

    /// Feedback (denominator) coefficients; a[0] must be 1
    a: Vec<f64>,

    /// Delay line of size = max(b.len(), a.len()) - 1
    delay: VecDeque<f64>,
}

impl LinearFilter {
    /// Create a new IIR DF2 filter from b[] and a[] taps.
    /// a[0] must equal 1.0.
    pub fn new(b: &[f64], a: &[f64]) -> Self {
        let order = usize::max(b.len(), a.len()).saturating_sub(1);
        Self {
            b: b.to_vec(),
            a: a.to_vec(),
            delay: VecDeque::from(vec![0.0; order]),
        }
    }

    /// Process a single sample
    pub fn step(&mut self, sample: &f64) -> f64 {
        let order = self.delay.len();
        // d[0] is added directly to output
        let d0 = if order > 0 {
            self.delay[0]
        } else {
            0.0
        };

        // Output
        let y = self.b[0] * *sample + d0;

        // Update delay registers
        // d[k] = d[k+1] + b[k+1]*x − a[k+1]*y
        for k in 0..order {
            let next_d = if k + 1 < order {
                self.delay[k + 1]
            } else {
                0.0
            };

            // b[k+1] or 0
            let bkp1 = if k + 1 < self.b.len() {
                self.b[k + 1]
            } else {
                0.0
            };

            // a[k+1] or 0
            let akp1 = if k + 1 < self.a.len() {
                self.a[k + 1]
            } else {
                0.0
            };

            self.delay[k] = next_d + bkp1 * *sample - akp1 * y;
        }
        y
    }

    pub fn reset(&mut self) {
        for i in self.delay.iter_mut() {
            *i = 0.0;
        }
    }

    pub fn process(&mut self, samples: &[f64]) -> Result<Option<Vec<f64>>, ErrorsJSL> {
        Ok(Some(samples.iter().map(|x| self.step(x)).collect()))
    }
    pub fn longest_buffer_length(&self) -> usize {
        self.delay.len()
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

        let mut iir = LinearFilter::new(&b, &a);

        let input = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let output = iir.process(&input).unwrap().unwrap();
        let golden = vec![0.0, 0.5, 1.4, 2.52, 3.706, 4.9117999999999995, 6.12354];
        assert_eq!(output, golden)
    }
}
