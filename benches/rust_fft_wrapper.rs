/// A wrapper around the RustFFT library to conform to our FfftEngine1D trait, allowing us to benchmark RustFFT against our own implementations.
/// This wrapper provides a way to use the RustFFT library within our benchmarking framework, allowing us to compare its performance against our own FFT implementations. The `RustFftWrapper` struct maintains the
/// necessary state for the FFT computation, including the size of the transform, direction, scaling factor, ordering, bit-reversal map, and the FFT plan and scratch space required by RustFFT. 
/// The `execute` method performs the FFT computation using RustFFT, while the `plan` method prepares the necessary precomputations based on the specified parameters. 
/// This wrapper allows us to evaluate the performance of RustFFT in a consistent manner alongside our own implementations, providing valuable insights into the trade-offs between different FFT algorithms and optimizations.
/// The implementation includes error handling to ensure that the input size matches the planned size and that the FFT plan is properly initialized before execution. 
/// It also handles different input orderings (standard vs. bit-reversed) and applies the appropriate scaling factor to the output based on the specified parameters.
/// Overall, this wrapper serves as a bridge between the RustFFT library and our benchmarking framework, enabling us to conduct comprehensive performance comparisons across multiple FFT implementations.
/// The `RustFftWrapper` is designed to be flexible and efficient, making it a valuable tool for evaluating the performance of RustFFT in various contexts and guiding decisions about which FFT implementation to use for different applications.
/// By integrating RustFFT into our benchmarking suite, we can gain a deeper understanding of the performance characteristics of different FFT implementations and make informed choices about which one to use based on our specific requirements and constraints.
/// The use of RustFFT in our benchmarks allows us to leverage a well-established and optimized FFT library, providing a strong baseline for comparison against our own implementations and helping us identify areas for potential optimization in our codebase.
/// In summary, the `RustFftWrapper` provides a seamless integration of the RustFFT library into our benchmarking framework, enabling us to evaluate its performance alongside our own FFT implementations and make informed decisions about which implementation to use in different contexts.
use std::sync::Arc;

use lib_jsl::ffts::fft_engine_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor};
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
                if self.bit_reverse_map.is_empty() {
                    return Err(ErrorsJSL::InvalidInputRange(
                        "BitReversed ordering requires a power-of-two transform size",
                    ));
                }
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
        if size == 0 {
            return Err(ErrorsJSL::InvalidInputRange("Size must be greater than 0"));
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
        if size.is_power_of_two() {
            let bits = size.trailing_zeros() as usize;
            self.bit_reverse_map = (0..size).map(|i| bit_reverse(i, bits)).collect();
        } else {
            self.bit_reverse_map.clear();
        }
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
