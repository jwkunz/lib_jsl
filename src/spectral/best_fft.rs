use num::Complex;

use crate::{
    prelude::ErrorsJSL,
    spectral::fft_enginer_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[path = "best_fft_x86.rs"]
mod best_fft_x86;
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[path = "best_fft_arm.rs"]
mod best_fft_arm;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[path = "best_fft_riscv.rs"]
mod best_fft_riscv;

pub struct BestFft {
    size: usize,
    direction: FftDirection,
    scale: FftScaleFactor,
    ordering: FftOrdering,
    stage_twiddles: Vec<Complex<f64>>,
    stage_offsets: Vec<usize>,
    bit_reverse_map: Vec<usize>,
}

impl BestFft {
    pub fn new() -> Self {
        Self {
            size: 0,
            direction: FftDirection::Forward,
            scale: FftScaleFactor::None,
            ordering: FftOrdering::Standard,
            stage_twiddles: Vec::new(),
            stage_offsets: Vec::new(),
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

#[inline]
fn radix2_pass_scalar(buffer: &mut [Complex<f64>], twiddles: &[Complex<f64>], size: usize, m: usize) {
    let half = m / 2;
    for k in (0..size).step_by(m) {
        let mut j = 0usize;
        while j + 4 <= half {
            // SAFETY:
            // - k in [0, size), j..j+3 < half and half <= m/2 ensure all indexes are valid.
            // - k steps by m and m <= size, so k + m - 1 < size.
            unsafe {
                let w0 = *twiddles.get_unchecked(j);
                let w1 = *twiddles.get_unchecked(j + 1);
                let w2 = *twiddles.get_unchecked(j + 2);
                let w3 = *twiddles.get_unchecked(j + 3);

                let a0 = k + j;
                let a1 = k + j + 1;
                let a2 = k + j + 2;
                let a3 = k + j + 3;
                let b0 = a0 + half;
                let b1 = a1 + half;
                let b2 = a2 + half;
                let b3 = a3 + half;

                let u0 = *buffer.get_unchecked(a0);
                let u1 = *buffer.get_unchecked(a1);
                let u2 = *buffer.get_unchecked(a2);
                let u3 = *buffer.get_unchecked(a3);

                let t0 = w0 * *buffer.get_unchecked(b0);
                let t1 = w1 * *buffer.get_unchecked(b1);
                let t2 = w2 * *buffer.get_unchecked(b2);
                let t3 = w3 * *buffer.get_unchecked(b3);

                *buffer.get_unchecked_mut(a0) = u0 + t0;
                *buffer.get_unchecked_mut(b0) = u0 - t0;
                *buffer.get_unchecked_mut(a1) = u1 + t1;
                *buffer.get_unchecked_mut(b1) = u1 - t1;
                *buffer.get_unchecked_mut(a2) = u2 + t2;
                *buffer.get_unchecked_mut(b2) = u2 - t2;
                *buffer.get_unchecked_mut(a3) = u3 + t3;
                *buffer.get_unchecked_mut(b3) = u3 - t3;
            }
            j += 4;
        }
        while j < half {
            // SAFETY: same bounds reasoning as the unrolled loop.
            unsafe {
                let w = *twiddles.get_unchecked(j);
                let a = k + j;
                let b = a + half;
                let u = *buffer.get_unchecked(a);
                let t = w * *buffer.get_unchecked(b);
                *buffer.get_unchecked_mut(a) = u + t;
                *buffer.get_unchecked_mut(b) = u - t;
            }
            j += 1;
        }
    }
}

#[inline]
fn radix2_pass_arch(buffer: &mut [Complex<f64>], twiddles: &[Complex<f64>], size: usize, m: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        best_fft_x86::radix2_pass(buffer, twiddles, size, m);
        return;
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    {
        best_fft_arm::radix2_pass(buffer, twiddles, size, m);
        return;
    }
    #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
    {
        best_fft_riscv::radix2_pass(buffer, twiddles, size, m);
        return;
    }
    #[allow(unreachable_code)]
    radix2_pass_scalar(buffer, twiddles, size, m);
}

impl FfftEngine1D for BestFft {
    fn execute(&mut self, input: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        if input.len() != self.size {
            return Err(ErrorsJSL::InvalidInputRange(
                "Input size must match the planned size",
            ));
        }

        let mut buffer = vec![Complex::new(0.0, 0.0); self.size];
        if matches!(self.ordering, FftOrdering::Standard) {
            for (i, &value) in input.iter().enumerate() {
                buffer[self.bit_reverse_map[i]] = value;
            }
        } else {
            buffer.copy_from_slice(input);
        }

        let log2n = self.size.trailing_zeros() as usize;
        for stage in 1..=log2n {
            let m = 1usize << stage;
            let start = self.stage_offsets[stage - 1];
            let end = self.stage_offsets[stage];
            let tw = &self.stage_twiddles[start..end];
            radix2_pass_arch(&mut buffer, tw, self.size, m);
        }

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

        let log2n = size.trailing_zeros() as usize;
        self.bit_reverse_map = (0..size).map(|i| bit_reverse(i, log2n)).collect();

        let sign = match direction {
            FftDirection::Forward => -1.0_f64,
            FftDirection::Inverse => 1.0_f64,
        };
        self.stage_offsets.clear();
        self.stage_offsets.reserve(log2n + 1);
        self.stage_twiddles.clear();
        self.stage_offsets.push(0);
        for stage in 1..=log2n {
            let m = 1usize << stage;
            let half = m / 2;
            for j in 0..half {
                let theta = sign * 2.0 * std::f64::consts::PI * j as f64 / m as f64;
                self.stage_twiddles.push(Complex::new(theta.cos(), theta.sin()));
            }
            self.stage_offsets.push(self.stage_twiddles.len());
        }

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
    fn test_best_fft_standard_ordering() {
        let input = fft_gaussian_1024_input();
        let expected = fft_gaussian_1024_golden();
        let mut fft = BestFft::new();
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
        dbg!("best_fft::standard execute elapsed", elapsed);
        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, 1e-9);
        }
    }

    #[test]
    fn test_best_fft_bit_reversed_ordering() {
        let input = fft_gaussian_1024_input();
        let expected = fft_gaussian_1024_golden();
        let bits = input.len().trailing_zeros() as usize;
        let mut bit_reversed_input = vec![Complex::new(0.0, 0.0); input.len()];
        for (i, value) in input.iter().enumerate() {
            bit_reversed_input[bit_reverse_index(i, bits)] = *value;
        }
        let mut fft = BestFft::new();
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
        dbg!("best_fft::bit_reversed execute elapsed", elapsed);
        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, 1e-9);
        }
    }
}
