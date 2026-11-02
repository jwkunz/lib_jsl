//! The split-radix FFT (Yavne, 1968; Duhamel and Hollmann, 1984), written for study.
//!
//! A radix-2 step splits the input into even and odd samples. Split radix splits the odd samples
//! once more, into those at `4m + 1` and `4m + 3`: with `U` the FFT of length `n/2` of the even
//! samples and `Z`, `Z'` the FFTs of length `n/4` of the two odd subsequences,
//!
//! `X_k = U_k + ωᵏ Z_k + ω³ᵏ Z'_k`, `X_{k+n/2} = U_k − (ωᵏ Z_k + ω³ᵏ Z'_k)`,
//! `X_{k+n/4} = U_{k+n/4} − i (ωᵏ Z_k − ω³ᵏ Z'_k)`, `X_{k+3n/4} = U_{k+n/4} + i (ωᵏ Z_k − ω³ᵏ Z'_k)`
//!
//! for `0 ≤ k < n/4` and `ω = e^{−2πi/n}` (the signs of `i` swap for the inverse transform). The even
//! half needs no twiddle factors at this level, which is why split radix uses the fewest real
//! operations of the classical power-of-two algorithms: about `4n log₂ n − 6n + 8` real additions
//! and multiplications, against `5n log₂ n` for radix 2.
//!
//! This engine is a plain recursion over strided views of the input, with an exact twiddle table.
//! Replacing an earlier engine of the same name, which (despite its name) was not a split-radix
//! algorithm and returned wrong results; its tests computed the error without checking it.
//!
//! See the book entry [The fast Fourier transform](https://jwkunz.github.io/lib_jsl/book/fft/fft.html#split-radix).
use num_complex::Complex;
use lib_jsl_core::ErrorsJSL;

use lib_jsl_ffts::fft_engine_trait::{FftEngine1D, FftDirection, FftOrdering, FftScaleFactor};

/// A recursive split-radix FFT engine for power-of-two sizes.
pub struct OptimizedSplitRadixFFT {
    size: usize,
    direction: FftDirection,
    scale: FftScaleFactor,
    ordering: FftOrdering,
    /// `ωᵏ` for `0 ≤ k < n`, with `ω = e^{∓2πi/n}` for the forward (inverse) transform.
    twiddles: Vec<Complex<f64>>,
}

impl OptimizedSplitRadixFFT {
    pub fn new() -> Self {
        Self {
            size: 0,
            direction: FftDirection::Forward,
            scale: FftScaleFactor::None,
            ordering: FftOrdering::Standard,
            twiddles: Vec::new(),
        }
    }
}

impl Default for OptimizedSplitRadixFFT {
    fn default() -> Self {
        Self::new()
    }
}

fn bit_reverse(mut n: usize, bits: usize) -> usize {
    let mut reversed = 0usize;
    for _ in 0..bits {
        reversed = (reversed << 1) | (n & 1);
        n >>= 1;
    }
    reversed
}

// ANCHOR: split_radix
/// Writes into `out` (of length `n`) the DFT of `input[start], input[start + stride], …`;
/// `ωᵏ` for this length is `twiddles[k · step]`, and `j` is `−i` (forward) or `+i` (inverse).
fn split_radix(input: &[Complex<f64>], start: usize, stride: usize, out: &mut [Complex<f64>], twiddles: &[Complex<f64>], step: usize, j: Complex<f64>) {
    let n = out.len();
    match n {
        1 => out[0] = input[start],
        2 => {
            let (a, b) = (input[start], input[start + stride]);
            out[0] = a + b;
            out[1] = a - b;
        }
        _ => {
            let (half, quarter) = (n / 2, n / 4);
            {
                let (u, z) = out.split_at_mut(half);
                let (z1, z3) = z.split_at_mut(quarter);
                split_radix(input, start, 2 * stride, u, twiddles, 2 * step, j);
                split_radix(input, start + stride, 4 * stride, z1, twiddles, 4 * step, j);
                split_radix(input, start + 3 * stride, 4 * stride, z3, twiddles, 4 * step, j);
            }
            for k in 0..quarter {
                let zk = twiddles[k * step] * out[half + k];
                let zk3 = twiddles[3 * k * step] * out[half + quarter + k];
                let (sum, difference) = (zk + zk3, j * (zk - zk3));
                let (u0, u1) = (out[k], out[k + quarter]);
                out[k] = u0 + sum;
                out[k + half] = u0 - sum;
                out[k + quarter] = u1 + difference;
                out[k + half + quarter] = u1 - difference;
            }
        }
    }
}
// ANCHOR_END: split_radix

impl FftEngine1D for OptimizedSplitRadixFFT {
    fn execute(&mut self, input: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        if input.len() != self.size {
            return Err(ErrorsJSL::InvalidInputRange("Input size must match the planned size"));
        }
        let n = self.size;
        // Bit-reversed input: sample i of the signal arrives at position bitrev(i).
        let natural: Vec<Complex<f64>> = match self.ordering {
            FftOrdering::Standard => input.to_vec(),
            FftOrdering::BitReversed => {
                let bits = n.trailing_zeros() as usize;
                (0..n).map(|i| input[bit_reverse(i, bits)]).collect()
            }
        };
        let j = match self.direction {
            FftDirection::Forward => Complex::new(0.0, -1.0),
            FftDirection::Inverse => Complex::new(0.0, 1.0),
        };
        let mut out = vec![Complex::new(0.0, 0.0); n];
        split_radix(&natural, 0, 1, &mut out, &self.twiddles, 1, j);
        let factor = match self.scale {
            FftScaleFactor::None => 1.0,
            FftScaleFactor::SqrtN => 1.0 / (n as f64).sqrt(),
            FftScaleFactor::N => 1.0 / n as f64,
        };
        if factor != 1.0 {
            out.iter_mut().for_each(|v| *v *= factor);
        }
        Ok(out)
    }

    fn plan(&mut self, size: usize, scale: FftScaleFactor, direction: FftDirection, ordering: FftOrdering) -> Result<(), ErrorsJSL> {
        if !size.is_power_of_two() {
            return Err(ErrorsJSL::InvalidInputRange("Size must be a power of 2"));
        }
        self.size = size;
        self.scale = scale;
        self.direction = direction;
        self.ordering = ordering;
        let sign = match direction {
            FftDirection::Forward => -1.0,
            FftDirection::Inverse => 1.0,
        };
        self.twiddles = (0..size)
            .map(|k| {
                let angle = sign * 2.0 * std::f64::consts::PI * k as f64 / size as f64;
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();
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
    use crate::test_bench_data::{fft_gaussian_32768_golden, fft_gaussian_32768_input};

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
        // The earlier engine of this name computed this error without checking it, and was wrong.
        assert!(err < 1e-9, "max abs error {err}");
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
        assert!(err < 1e-9, "max abs error {err}");
    }

    #[test]
    fn small_sizes_match_the_definition_and_invert() {
        for n in [1usize, 2, 4, 8, 16, 64] {
            let x: Vec<Complex<f64>> = (0..n).map(|k| Complex::new((k as f64 * 0.7).sin(), (k as f64 * 1.3).cos())).collect();
            let mut fft = OptimizedSplitRadixFFT::new();
            fft.plan(n, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
            let y = fft.execute(&x).unwrap();
            for (k, yk) in y.iter().enumerate() {
                let direct: Complex<f64> = x.iter().enumerate().map(|(m, xm)| {
                    let a = -2.0 * std::f64::consts::PI * ((k * m) % n) as f64 / n as f64;
                    xm * Complex::new(a.cos(), a.sin())
                }).sum();
                assert!((yk - direct).norm() < 1e-12, "n = {n}, k = {k}");
            }
            let mut inverse = OptimizedSplitRadixFFT::new();
            inverse.plan(n, FftScaleFactor::N, FftDirection::Inverse, FftOrdering::Standard).unwrap();
            let back = inverse.execute(&y).unwrap();
            assert!(back.iter().zip(&x).all(|(a, b)| (a - b).norm() < 1e-13), "n = {n}");
        }
    }
}
