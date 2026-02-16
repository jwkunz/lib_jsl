use std::sync::Arc;

use lib_jsl::ffts::fft_enginer_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor};
use lib_jsl::prelude::ErrorsJSL;
use num::Complex;
use rustfft::{Fft, FftPlanner};

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
