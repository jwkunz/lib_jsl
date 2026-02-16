use std::sync::Arc;

use num::Complex;
use rustfft::{Fft, FftPlanner};

use crate::{
    prelude::ErrorsJSL,
    spectral::fft_enginer_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
};

pub struct RustFftWrapper {
    size: usize,
    direction: FftDirection,
    scale: FftScaleFactor,
    ordering: FftOrdering,
    bit_reverse_map: Vec<usize>,
    plan: Option<Arc<dyn Fft<f64>>>,
    scratch: Vec<Complex<f64>>,
}

impl RustFftWrapper {
    pub fn new() -> Self {
        Self {
            size: 0,
            direction: FftDirection::Forward,
            scale: FftScaleFactor::None,
            ordering: FftOrdering::Standard,
            bit_reverse_map: Vec::new(),
            plan: None,
            scratch: Vec::new(),
        }
    }
}

#[inline]
fn bit_reverse(mut n: usize, bits: usize) -> usize {
    let mut reversed = 0usize;
    for _ in 0..bits {
        reversed = (reversed << 1) | (n & 1);
        n >>= 1;
    }
    reversed
}

impl FfftEngine1D for RustFftWrapper {
    fn execute(&mut self, input: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        if input.len() != self.size {
            return Err(ErrorsJSL::InvalidInputRange(
                "Input size must match the planned size",
            ));
        }
        let plan = self
            .plan
            .as_ref()
            .ok_or(ErrorsJSL::InvalidInputRange("FFT plan was not initialized"))?;

        let mut buffer = vec![Complex::new(0.0, 0.0); self.size];
        match self.ordering {
            FftOrdering::Standard => buffer.copy_from_slice(input),
            FftOrdering::BitReversed => {
                // RustFFT expects natural-order input, so convert bit-reversed input back.
                for i in 0..self.size {
                    buffer[i] = input[self.bit_reverse_map[i]];
                }
            }
        }

        plan.process_with_scratch(&mut buffer, &mut self.scratch);

        Ok(match self.scale {
            FftScaleFactor::None => buffer,
            FftScaleFactor::SqrtN => buffer
                .into_iter()
                .map(|x| x / (self.size as f64).sqrt())
                .collect(),
            FftScaleFactor::N => buffer.into_iter().map(|x| x / self.size as f64).collect(),
        })
    }

    fn plan(
        &mut self,
        size: usize,
        scale: FftScaleFactor,
        direction: FftDirection,
        ordering: FftOrdering,
    ) -> Result<(), ErrorsJSL> {
        if !size.is_power_of_two() {
            return Err(ErrorsJSL::InvalidInputRange("Size must be a power of 2"));
        }

        self.size = size;
        self.scale = scale;
        self.direction = direction;
        self.ordering = ordering;

        let mut planner = FftPlanner::<f64>::new();
        let plan = match direction {
            FftDirection::Forward => planner.plan_fft_forward(size),
            FftDirection::Inverse => planner.plan_fft_inverse(size),
        };

        self.scratch = vec![Complex::new(0.0, 0.0); plan.get_inplace_scratch_len()];
        let bits = size.trailing_zeros() as usize;
        self.bit_reverse_map = (0..size).map(|i| bit_reverse(i, bits)).collect();
        self.plan = Some(plan);
        Ok(())
    }

    fn get_size(&self) -> usize {
        self.size
    }

    fn get_scale_factor(&self) -> FftScaleFactor {
        self.scale
    }

    fn get_direction(&self) -> FftDirection {
        self.direction
    }

    fn get_ordering(&self) -> FftOrdering {
        self.ordering
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::spectral::test_bench_data::{fft_gaussian_32768_golden, fft_gaussian_32768_input};

    fn assert_complex_close(actual: Complex<f64>, expected: Complex<f64>, tol: f64) {
        assert!(
            (actual.re - expected.re).abs() < tol && (actual.im - expected.im).abs() < tol,
            "actual={actual:?}, expected={expected:?}, tol={tol}"
        );
    }

    fn bit_reverse_index(mut n: usize, bits: usize) -> usize {
        let mut reversed = 0;
        for _ in 0..bits {
            reversed <<= 1;
            reversed |= n & 1;
            n >>= 1;
        }
        reversed
    }

    #[test]
    fn test_rust_fft_wrapper_standard_ordering() {
        let input = fft_gaussian_32768_input();
        let expected = fft_gaussian_32768_golden();

        let mut fft = RustFftWrapper::new();
        fft.plan(
            input.len(),
            FftScaleFactor::None,
            FftDirection::Forward,
            FftOrdering::Standard,
        )
        .unwrap();

        let start = Instant::now();
        let output = fft.execute(&input).unwrap();
        let elapsed = start.elapsed();
        dbg!("rust_fft_wrapper::standard execute elapsed", elapsed);

        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, 1e-9);
        }
    }

    #[test]
    fn test_rust_fft_wrapper_bit_reversed_ordering() {
        let input = fft_gaussian_32768_input();
        let expected = fft_gaussian_32768_golden();

        let bits = input.len().trailing_zeros() as usize;
        let mut bit_reversed_input = vec![Complex::new(0.0, 0.0); input.len()];
        for (i, value) in input.iter().enumerate() {
            bit_reversed_input[bit_reverse_index(i, bits)] = *value;
        }

        let mut fft = RustFftWrapper::new();
        fft.plan(
            input.len(),
            FftScaleFactor::None,
            FftDirection::Forward,
            FftOrdering::BitReversed,
        )
        .unwrap();

        let start = Instant::now();
        let output = fft.execute(&bit_reversed_input).unwrap();
        let elapsed = start.elapsed();
        dbg!("rust_fft_wrapper::bit_reversed execute elapsed", elapsed);

        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, 1e-9);
        }
    }
}
