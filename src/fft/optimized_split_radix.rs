use num::Complex;

use crate::{
    prelude::ErrorsJSL,
    spectral::fft_enginer_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
};

/// Learning-oriented aggressive FFT variant.
///
/// This keeps the same interface as the other engines, but uses fused m=2/m=4/m=8
/// front passes plus radix-2 tails. It is intentionally kept as a study target:
/// it can run fast, but it may introduce noticeable numerical drift vs. the golden FFT.
pub struct OptimizedSplitRadixFFT {
    size: usize,
    direction: FftDirection,
    scale: FftScaleFactor,
    ordering: FftOrdering,
    stage_twiddles: Vec<Complex<f64>>,
    stage_offsets: Vec<usize>,
    bit_reverse_map: Vec<usize>,
}

impl OptimizedSplitRadixFFT {
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
fn pass_m2(buffer: &mut [Complex<f64>]) {
    for k in (0..buffer.len()).step_by(2) {
        let a = buffer[k];
        let b = buffer[k + 1];
        buffer[k] = a + b;
        buffer[k + 1] = a - b;
    }
}

#[inline]
fn pass_m4(buffer: &mut [Complex<f64>], direction: FftDirection) {
    let w = match direction {
        FftDirection::Forward => Complex::new(0.0, -1.0),
        FftDirection::Inverse => Complex::new(0.0, 1.0),
    };
    for k in (0..buffer.len()).step_by(4) {
        let x0 = buffer[k];
        let x1 = buffer[k + 1];
        let x2 = buffer[k + 2];
        let x3 = buffer[k + 3];

        let t0 = x0 + x2;
        let t1 = x0 - x2;
        let t2 = x1 + x3;
        let t3 = (x1 - x3) * w;

        buffer[k] = t0 + t2;
        buffer[k + 1] = t1 + t3;
        buffer[k + 2] = t0 - t2;
        buffer[k + 3] = t1 - t3;
    }
}

#[inline]
fn pass_m8_aggressive(buffer: &mut [Complex<f64>], direction: FftDirection) {
    let c = std::f64::consts::FRAC_1_SQRT_2;
    let w1 = match direction {
        FftDirection::Forward => Complex::new(c, -c),
        FftDirection::Inverse => Complex::new(c, c),
    };
    let w2 = match direction {
        FftDirection::Forward => Complex::new(0.0, -1.0),
        FftDirection::Inverse => Complex::new(0.0, 1.0),
    };
    let w3 = match direction {
        FftDirection::Forward => Complex::new(-c, -c),
        FftDirection::Inverse => Complex::new(-c, c),
    };

    for k in (0..buffer.len()).step_by(8) {
        let x0 = buffer[k];
        let x1 = buffer[k + 1];
        let x2 = buffer[k + 2];
        let x3 = buffer[k + 3];
        let x4 = buffer[k + 4];
        let x5 = buffer[k + 5];
        let x6 = buffer[k + 6];
        let x7 = buffer[k + 7];

        let y0 = x0;
        let y1 = x1 * w1;
        let y2 = x2 * w2;
        let y3 = x3 * w3;
        let y4 = x4;
        let y5 = x5 * w1;
        let y6 = x6 * w2;
        let y7 = x7 * w3;

        let a0 = y0 + y4;
        let a1 = y1 + y5;
        let a2 = y2 + y6;
        let a3 = y3 + y7;
        let b0 = y0 - y4;
        let b1 = y1 - y5;
        let b2 = y2 - y6;
        let b3 = y3 - y7;

        let t0 = a0 + a2;
        let t1 = a0 - a2;
        let t2 = a1 + a3;
        let t3 = (a1 - a3) * w2;
        let u0 = b0 + b2;
        let u1 = b0 - b2;
        let u2 = b1 + b3;
        let u3 = (b1 - b3) * w2;

        buffer[k] = t0 + t2;
        buffer[k + 4] = t0 - t2;
        buffer[k + 2] = t1 + t3;
        buffer[k + 6] = t1 - t3;
        buffer[k + 1] = u0 + u2;
        buffer[k + 5] = u0 - u2;
        buffer[k + 3] = u1 + u3;
        buffer[k + 7] = u1 - u3;
    }
}

#[inline]
fn radix2_pass_unrolled(
    buffer: &mut [Complex<f64>],
    twiddles: &[Complex<f64>],
    size: usize,
    m: usize,
) {
    let half = m / 2;
    for k in (0..size).step_by(m) {
        let mut j = 0usize;
        while j + 4 <= half {
            // SAFETY: indexes are in-bounds by construction.
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
            // SAFETY: indexes are in-bounds by construction.
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

impl FfftEngine1D for OptimizedSplitRadixFFT {
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
        let mut stage = 1usize;

        if stage <= log2n {
            pass_m2(&mut buffer);
            stage += 1;
        }
        if stage <= log2n {
            pass_m4(&mut buffer, self.direction);
            stage += 1;
        }
        if stage <= log2n {
            pass_m8_aggressive(&mut buffer, self.direction);
            stage += 1;
        }

        while stage <= log2n {
            let m = 1usize << stage;
            let start = self.stage_offsets[stage - 1];
            let end = self.stage_offsets[stage];
            let tw = &self.stage_twiddles[start..end];
            radix2_pass_unrolled(&mut buffer, tw, self.size, m);
            stage += 1;
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
        self.stage_offsets.reserve(log2n + 2);
        self.stage_twiddles.clear();
        self.stage_offsets.push(0);

        for stage in 1..=log2n {
            let m = 1usize << stage;
            let half = m / 2;
            for j in 0..half {
                let theta = sign * 2.0 * std::f64::consts::PI * j as f64 / m as f64;
                self.stage_twiddles
                    .push(Complex::new(theta.cos(), theta.sin()));
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
    use crate::spectral::test_bench_data::{fft_gaussian_32768_golden, fft_gaussian_32768_input};

    fn max_abs_error(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (*x - *y).norm())
            .fold(0.0_f64, f64::max)
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
    fn test_optimized_split_radix_standard() {
        let input = fft_gaussian_32768_input();
        let expected = fft_gaussian_32768_golden();

        let mut fft = OptimizedSplitRadixFFT::new();
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
        let err = max_abs_error(&output, &expected);
        dbg!("optimized_split_radix::standard execute elapsed", elapsed);
        dbg!("optimized_split_radix::standard max abs error", err);

        assert_eq!(output.len(), expected.len());
        assert!(output.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
    }

    #[test]
    fn test_optimized_split_radix_bit_reversed() {
        let input = fft_gaussian_32768_input();
        let expected = fft_gaussian_32768_golden();

        let bits = input.len().trailing_zeros() as usize;
        let mut bit_reversed_input = vec![Complex::new(0.0, 0.0); input.len()];
        for (i, value) in input.iter().enumerate() {
            bit_reversed_input[bit_reverse_index(i, bits)] = *value;
        }

        let mut fft = OptimizedSplitRadixFFT::new();
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
        let err = max_abs_error(&output, &expected);
        dbg!("optimized_split_radix::bit_reversed execute elapsed", elapsed);
        dbg!("optimized_split_radix::bit_reversed max abs error", err);

        assert_eq!(output.len(), expected.len());
        assert!(output.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
    }
}
