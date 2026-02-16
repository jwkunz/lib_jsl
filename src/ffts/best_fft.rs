/// BestFft: A high-performance FFT implementation optimized for power-of-two sizes.
/// This implementation uses an iterative Cooley-Tukey algorithm with precomputed twiddle factors and bit-reversal indexing for efficient in-place computation. 
/// The `BestFft` struct maintains the necessary state for the FFT computation, including the size of the transform, direction, scaling factor, ordering, bit-reversal map, stage offsets for twiddle factors, and the twiddle factors themselves. 
/// The `execute` method performs the FFT computation using an iterative approach, while the `plan` method prepares the necessary precomputations based on the specified parameters.    

use num::Complex;

use crate::{
    prelude::ErrorsJSL,
    ffts::fft_enginer_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
};

pub struct BestFft {
    size: usize,
    direction: FftDirection,
    scale: FftScaleFactor,
    ordering: FftOrdering,
    bit_reverse_map: Vec<usize>,
    stage_offsets: Vec<usize>,
    twiddle_re: Vec<f64>,
    twiddle_im: Vec<f64>,
    work: Vec<Complex<f64>>,
}

impl BestFft {
    pub fn new() -> Self {
        Self {
            size: 0,
            direction: FftDirection::Forward,
            scale: FftScaleFactor::None,
            ordering: FftOrdering::Standard,
            bit_reverse_map: Vec::new(),
            stage_offsets: Vec::new(),
            twiddle_re: Vec::new(),
            twiddle_im: Vec::new(),
            work: Vec::new(),
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
fn pass_m2(work: &mut [Complex<f64>]) {
    for k in (0..work.len()).step_by(2) {
        let x0 = work[k];
        let x1 = work[k + 1];
        work[k] = x0 + x1;
        work[k + 1] = x0 - x1;
    }
}

#[inline]
fn pass_m4(work: &mut [Complex<f64>], direction: FftDirection) {
    let w = match direction {
        FftDirection::Forward => Complex::new(0.0, -1.0),
        FftDirection::Inverse => Complex::new(0.0, 1.0),
    };
    for k in (0..work.len()).step_by(4) {
        let a0 = work[k];
        let b0 = work[k + 2];
        work[k] = a0 + b0;
        work[k + 2] = a0 - b0;

        let a1 = work[k + 1];
        let b1 = work[k + 3] * w;
        work[k + 1] = a1 + b1;
        work[k + 3] = a1 - b1;
    }
}

#[inline]
fn pass_generic_unrolled(
    work: &mut [Complex<f64>],
    tw_re: &[f64],
    tw_im: &[f64],
    size: usize,
    m: usize,
) {
    let half = m >> 1;
    for k in (0..size).step_by(m) {
        let mut j = 0usize;
        while j + 4 <= half {
            // SAFETY:
            // - j..j+3 within [0, half), so twiddle reads are in range.
            // - a/b indexes remain within [k, k+m), and k+m <= size.
            unsafe {
                let wr0 = *tw_re.get_unchecked(j);
                let wi0 = *tw_im.get_unchecked(j);
                let wr1 = *tw_re.get_unchecked(j + 1);
                let wi1 = *tw_im.get_unchecked(j + 1);
                let wr2 = *tw_re.get_unchecked(j + 2);
                let wi2 = *tw_im.get_unchecked(j + 2);
                let wr3 = *tw_re.get_unchecked(j + 3);
                let wi3 = *tw_im.get_unchecked(j + 3);

                let a0 = k + j;
                let a1 = a0 + 1;
                let a2 = a0 + 2;
                let a3 = a0 + 3;
                let b0 = a0 + half;
                let b1 = b0 + 1;
                let b2 = b0 + 2;
                let b3 = b0 + 3;

                let u0 = *work.get_unchecked(a0);
                let u1 = *work.get_unchecked(a1);
                let u2 = *work.get_unchecked(a2);
                let u3 = *work.get_unchecked(a3);
                let v0 = *work.get_unchecked(b0);
                let v1 = *work.get_unchecked(b1);
                let v2 = *work.get_unchecked(b2);
                let v3 = *work.get_unchecked(b3);

                let t0 = Complex::new(v0.re * wr0 - v0.im * wi0, v0.re * wi0 + v0.im * wr0);
                let t1 = Complex::new(v1.re * wr1 - v1.im * wi1, v1.re * wi1 + v1.im * wr1);
                let t2 = Complex::new(v2.re * wr2 - v2.im * wi2, v2.re * wi2 + v2.im * wr2);
                let t3 = Complex::new(v3.re * wr3 - v3.im * wi3, v3.re * wi3 + v3.im * wr3);

                *work.get_unchecked_mut(a0) = u0 + t0;
                *work.get_unchecked_mut(b0) = u0 - t0;
                *work.get_unchecked_mut(a1) = u1 + t1;
                *work.get_unchecked_mut(b1) = u1 - t1;
                *work.get_unchecked_mut(a2) = u2 + t2;
                *work.get_unchecked_mut(b2) = u2 - t2;
                *work.get_unchecked_mut(a3) = u3 + t3;
                *work.get_unchecked_mut(b3) = u3 - t3;
            }
            j += 4;
        }
        while j < half {
            let a = k + j;
            let b = a + half;
            let u = work[a];
            let v = work[b];
            let wr = tw_re[j];
            let wi = tw_im[j];
            let t = Complex::new(v.re * wr - v.im * wi, v.re * wi + v.im * wr);
            work[a] = u + t;
            work[b] = u - t;
            j += 1;
        }
    }
}

impl FfftEngine1D for BestFft {
    fn execute(&mut self, input: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        if input.len() != self.size {
            return Err(ErrorsJSL::InvalidInputRange(
                "Input size must match the planned size",
            ));
        }

        if self.work.len() != self.size {
            self.work.resize(self.size, Complex::new(0.0, 0.0));
        }

        match self.ordering {
            FftOrdering::Standard => {
                for (i, &value) in input.iter().enumerate() {
                    self.work[self.bit_reverse_map[i]] = value;
                }
            }
            FftOrdering::BitReversed => self.work.copy_from_slice(input),
        }

        let log2n = self.size.trailing_zeros() as usize;
        let mut stage = 1usize;
        if stage <= log2n {
            pass_m2(&mut self.work);
            stage += 1;
        }
        if stage <= log2n {
            pass_m4(&mut self.work, self.direction);
            stage += 1;
        }
        while stage <= log2n {
            let m = 1usize << stage;
            let start = self.stage_offsets[stage - 1];
            let end = self.stage_offsets[stage];
            pass_generic_unrolled(
                &mut self.work,
                &self.twiddle_re[start..end],
                &self.twiddle_im[start..end],
                self.size,
                m,
            );
            stage += 1;
        }

        let mut out = self.work.clone();
        match self.scale {
            FftScaleFactor::None => {}
            FftScaleFactor::SqrtN => {
                let s = 1.0 / (self.size as f64).sqrt();
                for v in &mut out {
                    v.re *= s;
                    v.im *= s;
                }
            }
            FftScaleFactor::N => {
                let s = 1.0 / self.size as f64;
                for v in &mut out {
                    v.re *= s;
                    v.im *= s;
                }
            }
        }
        Ok(out)
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

        self.stage_offsets.clear();
        self.stage_offsets.reserve(log2n + 1);
        self.stage_offsets.push(0);
        self.twiddle_re.clear();
        self.twiddle_im.clear();

        let sign = match direction {
            FftDirection::Forward => -1.0_f64,
            FftDirection::Inverse => 1.0_f64,
        };
        for stage in 1..=log2n {
            let m = 1usize << stage;
            let half = m >> 1;
            for j in 0..half {
                let theta = sign * 2.0 * std::f64::consts::PI * j as f64 / m as f64;
                self.twiddle_re.push(theta.cos());
                self.twiddle_im.push(theta.sin());
            }
            self.stage_offsets.push(self.twiddle_re.len());
        }

        self.work.resize(size, Complex::new(0.0, 0.0));
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
    use crate::ffts::test_bench_data::{fft_gaussian_32768_golden, fft_gaussian_32768_input};

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
        let input = fft_gaussian_32768_input();
        let expected = fft_gaussian_32768_golden();
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
        dbg!("best_fft::standard execute elapsed", start.elapsed());

        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, 1e-9);
        }
    }

    #[test]
    fn test_best_fft_bit_reversed_ordering() {
        let input = fft_gaussian_32768_input();
        let expected = fft_gaussian_32768_golden();
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
        dbg!("best_fft::bit_reversed execute elapsed", start.elapsed());

        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, 1e-9);
        }
    }
}
