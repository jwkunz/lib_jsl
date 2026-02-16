use num::Complex;

/// This is a simple implementation of an FFT according to the original cooley tukey algorithm.  It is lightweight and does not require any external dependencies, but it is not optimized for performance. It is intended for educational purposes and for small input sizes, and should not be used for large input sizes or for performance-critical applications. For larger input sizes or performance-critical applications, it is recommended to use a more optimized FFT implementation, such as the RustFFT library, which is also included in this codebase as an alternative FFT engine. The simple cooley tukey implementation can be useful for understanding the basic principles of the FFT algorithm and for testing purposes, but it may not be suitable for all applications.


use crate::{prelude::ErrorsJSL, spectral::fft_enginer_trait::{FfftEngine1D, FftDirection, FftScaleFactor, FftOrdering}};
 pub struct CooleyTukeyFFT {
    size: usize,
    direction: FftDirection,
    scale: FftScaleFactor,
    ordering: FftOrdering,
}

impl CooleyTukeyFFT {
    pub fn new() -> Self {
        CooleyTukeyFFT {
            size: 0,
            direction: FftDirection::Forward,
            scale: FftScaleFactor::None,
            ordering: FftOrdering::Standard,
        }
    }
}

fn bit_reverse(mut n: usize, bits: usize) -> usize {
    let mut reversed = 0;
    for _ in 0..bits {
        reversed <<= 1;
        reversed |= n & 1;
        n >>= 1;
    }
    reversed
}   

impl FfftEngine1D for CooleyTukeyFFT {
    fn plan(&mut self, size: usize, scale: FftScaleFactor, direction: FftDirection, ordering: FftOrdering) -> Result<(), ErrorsJSL> {
        if !size.is_power_of_two() {
            return Err(ErrorsJSL::InvalidInputRange("Size must be a power of 2"));
        }
        self.size = size;
        self.scale = scale;
        self.direction = direction;
        self.ordering = ordering;
        Ok(())
    }

    
    fn execute(&mut self, input: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        if input.len() != self.size {
            return Err(ErrorsJSL::InvalidInputRange("Input size must match the planned size"));
        }
        // Fast log2 calculation using trailing_zeros, since size is a power of 2
        let log2n = self.size.trailing_zeros() as usize;

        let mut buffer = input.to_vec();
        if matches!(self.ordering, FftOrdering::Standard) {
            // Do pre bit reversal permutation to compensate for in-place FFT
            for i in 0..self.size {
                let j = bit_reverse(i, log2n);
                if i < j {
                    buffer.swap(i, j);
                }
            }
        }
        // Do in place radix-2 Cooley-Tukey FFT with bit reversal
        for s in 1..=log2n {
            // m = 2^s is the size of the sub-FFTs being combined
            let m = 1 << s;
            // Compute the twiddle factor W_m = exp(-2*pi*i/m) for forward FFT or exp(2*pi*i/m) for inverse FFT
            let theta = match self.direction {
                FftDirection::Forward => -2.0 * std::f64::consts::PI / m as f64,
                FftDirection::Inverse => 2.0 * std::f64::consts::PI / m as f64,
            };
            let wm = Complex::new(theta.cos(), theta.sin());
            // Combine sub-FFTs of size m/2 into sub-FFTs of size m
            for k in (0..self.size).step_by(m) {
                // w starts at 1 and is multiplied by W_m in each iteration to get the next twiddle factor
                let mut w = Complex::new(1.0, 0.0);
                // Perform the butterfly operations for the current stage
                for j in 0..(m / 2) {
                    // t = w * buffer[k + j + m/2] is the contribution from the second half of the sub-FFT
                    // u = buffer[k + j] is the contribution from the first half of the sub-FFT before the butterfly operation  
                    let t = w * buffer[k + j + m / 2];
                    let u = buffer[k + j];
                    // Update the buffer with the results of the butterfly operation    
                    buffer[k + j] = u + t;
                    buffer[k + j + m / 2] = u - t;
                    // Update w to the next twiddle factor for the next iteration of the butterfly operation
                    w *= wm;
                }
            }
        }
        // Scale the output if necessary according to the planned scale factor
        Ok(match self.scale {
            FftScaleFactor::None => buffer,
            FftScaleFactor::SqrtN => buffer.into_iter().map(|x| x / (self.size as f64).sqrt()).collect(),
            FftScaleFactor::N => buffer.into_iter().map(|x| x / self.size as f64).collect(),
        })
    }

    fn get_direction(&self) -> FftDirection {
        self.direction
    }

    fn get_scale_factor(&self) -> FftScaleFactor {
        self.scale
    }

    fn get_size(&self) -> usize {
        self.size
    }
    fn get_ordering(&self) -> super::fft_enginer_trait::FftOrdering {
        self.ordering
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::spectral::test_bench_data::{fft_gaussian_1024_golden, fft_gaussian_1024_input};

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
    fn test_cooley_tukey_fft() {
        let input = fft_gaussian_1024_input();
        let mut fft = CooleyTukeyFFT::new();
        fft.plan(input.len(), FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();

        let start = Instant::now();
        let output = fft.execute(input.as_slice()).unwrap();
        let elapsed = start.elapsed();
        dbg!("simple_cooley_tukey::standard execute elapsed", elapsed);

        let expected = fft_gaussian_1024_golden();

        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, 1e-9);
        }
    }

    #[test]
    fn test_cooley_tukey_fft_bit_reversed_ordering() {
        let input = fft_gaussian_1024_input();

        let bits = input.len().trailing_zeros() as usize;
        let mut bit_reversed_input = vec![Complex::new(0.0, 0.0); input.len()];
        for (i, value) in input.iter().enumerate() {
            bit_reversed_input[bit_reverse_index(i, bits)] = *value;
        }

        let mut fft = CooleyTukeyFFT::new();
        fft.plan(
            bit_reversed_input.len(),
            FftScaleFactor::None,
            FftDirection::Forward,
            FftOrdering::BitReversed,
        )
        .unwrap();

        let start = Instant::now();
        let output = fft.execute(bit_reversed_input.as_slice()).unwrap();
        let elapsed = start.elapsed();
        dbg!("simple_cooley_tukey::bit_reversed execute elapsed", elapsed);

        let expected = fft_gaussian_1024_golden();

        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, 1e-9);
        }
    }
}   
