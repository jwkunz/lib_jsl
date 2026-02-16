use std::collections::VecDeque;

/// A boxcar intergrator takes the sum of the last n values.  This has an efficient streaming implementation that only requires one multiplication and two additions per output value, regardless of the number of taps.
/// The `numtaps` parameter specifies the number of taps in the boxcar integrator, which determines how many past values are summed to produce each output value. 
/// A larger number of taps will result in a smoother output but may also introduce more latency and reduce the responsiveness of the filter to changes in the input signal. 
/// The `scale` parameter allows you to control the overall gain of the boxcar integrator.  

use crate::{dsp::stream_operator::{StreamOperator, StreamOperatorManagement}, prelude::{ErrorsJSL, IsLinearOperatable}};
pub struct BoxcarIntegrator<T: IsLinearOperatable> {
    numtaps: usize,
    scale: T,
    buffer: VecDeque<T>,
    sum: T,
}

impl<T: IsLinearOperatable> BoxcarIntegrator<T>{
    pub fn new(numtaps: usize, scale: T) -> Result<Self, ErrorsJSL> {
        if numtaps == 0 {
            return Err(ErrorsJSL::InvalidInputRange("numtaps must be greater than 0"));
        }
        Ok(Self {
            numtaps,
            scale,
            buffer: VecDeque::from(vec![T::zero(); numtaps]),
            sum: T::zero(),
        })
    }

    pub fn step(&mut self, input: &T) -> T {
        // Subtract the oldest value from the sum
        self.sum = self.sum - *self.buffer.front().unwrap();
        // Update the buffer with the new input value
        self.buffer.pop_front();
        // Add the new input value to the sum
        self.sum = self.sum + *input;
        self.buffer.push_back(*input);
        // Return the scaled sum
        self.sum * self.scale 
    }
}

impl StreamOperatorManagement for BoxcarIntegrator<f64> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.buffer = VecDeque::from(vec![0.0; self.numtaps]);
        self.sum = 0.0;
        Ok(())
    }
    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}   

impl StreamOperator<f64, f64> for BoxcarIntegrator<f64> {
    fn process(&mut self, input: &[f64]) -> Result<Option<Vec<f64>>, ErrorsJSL> {
        if input.is_empty() {
            return Ok(None);
        }
        Ok(Some(input.iter().map(|x| self.step(x)).collect()))  
    }
    fn flush(&mut self) -> Result<Option<Vec<f64>>, ErrorsJSL> {
        Ok(None)
    }
}   

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boxcar_integrator() {
        let mut integrator = BoxcarIntegrator::new(3, 1.0).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = integrator.process(&input).unwrap().unwrap();
        let expected = vec![1.0, 3.0, 6.0, 9.0, 12.0];
        assert_eq!(output, expected);
    }
}