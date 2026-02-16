/// Bluestein's algorithm (chirp-z transform) implementation for 1D FFTs of arbitrary size. 
/// This algorithm allows us to compute the DFT of sequences whose length is not a power of two by transforming the problem into a convolution, which can be efficiently computed using FFTs. 
/// The implementation includes planning and execution methods, and supports different scaling factors and input orderings.   
/// The `BluesteinFft` struct maintains the necessary state for the FFT computation, including the size of the transform, direction, scaling factor, ordering, convolution size, precomputed chirp factors, and the FFT engines used for the convolution step. 
/// The `execute` method performs the FFT computation using Bluestein's algorithm, while the `plan` method prepares the necessary precomputations based on the specified parameters. 

use num::Complex;

use crate::{
    ffts::{
        best_fft::BestFft,
        fft_enginer_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
    },
    prelude::ErrorsJSL,
};

pub struct BluesteinFft {
    size: usize,
    direction: FftDirection,
    scale: FftScaleFactor,
    ordering: FftOrdering,
    conv_size: usize,
    chirp: Vec<Complex<f64>>,
    b_fft: Vec<Complex<f64>>,
    fft_engine: BestFft,
    ifft_engine: BestFft,
    bit_reverse_map: Vec<usize>,
}

impl BluesteinFft {
    pub fn new() -> Self {
        Self {
            size: 0,
            direction: FftDirection::Forward,
            scale: FftScaleFactor::None,
            ordering: FftOrdering::Standard,
            conv_size: 0,
            chirp: Vec::new(),
            b_fft: Vec::new(),
            fft_engine: BestFft::new(),
            ifft_engine: BestFft::new(),
            bit_reverse_map: Vec::new(),
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

impl FfftEngine1D for BluesteinFft {
    fn execute(&mut self, input: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        if input.len() != self.size {
            return Err(ErrorsJSL::InvalidInputRange(
                "Input size must match the planned size",
            ));
        }
        // Prepare the input sequence in the appropriate order based on the planned ordering
        let x: Vec<Complex<f64>> = match self.ordering {
            FftOrdering::Standard => input.to_vec(),
            FftOrdering::BitReversed => {
                if !self.size.is_power_of_two() {
                    return Err(ErrorsJSL::InvalidInputRange(
                        "BitReversed ordering requires a power-of-two transform size",
                    ));
                }
                let mut natural = vec![Complex::new(0.0, 0.0); self.size];
                for i in 0..self.size {
                    natural[i] = input[self.bit_reverse_map[i]];
                }
                natural
            }
        };
        // Step 1: Multiply input by chirp factors to prepare for convolution
        let mut a = vec![Complex::new(0.0, 0.0); self.conv_size];
        for n in 0..self.size {
            a[n] = x[n] * self.chirp[n];
        }
        // Step 2: Perform convolution using FFTs - compute the FFT of a, multiply by precomputed FFT of b, and then compute the inverse FFT to get the convolution result  
        let a_fft = self.fft_engine.execute(&a)?;
        let mut prod = vec![Complex::new(0.0, 0.0); self.conv_size];
        for i in 0..self.conv_size {
            prod[i] = a_fft[i] * self.b_fft[i];
        }
        let conv = self.ifft_engine.execute(&prod)?;
        // Step 3: Multiply the convolution result by the chirp factors to get the final FFT output 
        let mut out = vec![Complex::new(0.0, 0.0); self.size];
        for k in 0..self.size {
            out[k] = conv[k] * self.chirp[k];
        }
        // Step 4: Apply scaling if necessary and return the result
        Ok(match self.scale {
            FftScaleFactor::None => out,
            FftScaleFactor::SqrtN => out
                .into_iter()
                .map(|x| x / (self.size as f64).sqrt())
                .collect(),
            FftScaleFactor::N => out.into_iter().map(|x| x / self.size as f64).collect(),
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
        if size.is_power_of_two() {
            let bits = size.trailing_zeros() as usize;
            self.bit_reverse_map = (0..size).map(|i| bit_reverse(i, bits)).collect();
        } else {
            self.bit_reverse_map.clear();
        }
        // The convolution size must be at least 2N-1 to avoid circular convolution issues, and we round up to the next power of two for efficient FFT computation  
        self.conv_size = (2 * size - 1).next_power_of_two();

        let sign = match direction {
            FftDirection::Forward => -1.0_f64,
            FftDirection::Inverse => 1.0_f64,
        };
        // Precompute the chirp factors and the FFT of b for the convolution step   
        self.chirp = (0..size)
            .map(|k| {
                let theta = sign * std::f64::consts::PI * (k as f64) * (k as f64) / size as f64;
                Complex::new(theta.cos(), theta.sin())
            })
            .collect();
        // Prepare the b sequence for convolution, which is the chirp sequence extended to the convolution size with appropriate symmetry  
        let mut b = vec![Complex::new(0.0, 0.0); self.conv_size];
        for n in 0..size {
            let theta = -sign * std::f64::consts::PI * (n as f64) * (n as f64) / size as f64;
            let val = Complex::new(theta.cos(), theta.sin());
            b[n] = val;
            if n != 0 {
                b[self.conv_size - n] = val;
            }
        }
        // Plan the FFTs for the convolution step using the underlying FFT engines  
        self.fft_engine.plan(
            self.conv_size,
            FftScaleFactor::None,
            FftDirection::Forward,
            FftOrdering::Standard,
        )?;
        // We only need to compute the FFT of b once since it doesn't depend on the input
        self.ifft_engine.plan(
            self.conv_size,
            FftScaleFactor::N,
            FftDirection::Inverse,
            FftOrdering::Standard,
        )?;
        // Compute the FFT of b for the convolution step and store it for reuse during execution
        self.b_fft = self.fft_engine.execute(&b)?;

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
    use crate::ffts::test_bench_data::{fft_gaussian_63_golden, fft_gaussian_63_input};

    fn assert_complex_close(actual: Complex<f64>, expected: Complex<f64>, tol: f64) {
        assert!(
            (actual.re - expected.re).abs() < tol && (actual.im - expected.im).abs() < tol,
            "actual={actual:?}, expected={expected:?}, tol={tol}"
        );
    }

    #[test]
    fn test_bluestein_fft_len_63_standard() {
        let input = fft_gaussian_63_input();
        let expected = fft_gaussian_63_golden();

        let mut fft = BluesteinFft::new();
        fft.plan(
            input.len(),
            FftScaleFactor::None,
            FftDirection::Forward,
            FftOrdering::Standard,
        )
        .unwrap();

        let start = Instant::now();
        let output = fft.execute(&input).unwrap();
        dbg!("bluestein_fft::len63 standard execute elapsed", start.elapsed());

        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, 1e-8);
        }
    }
}
