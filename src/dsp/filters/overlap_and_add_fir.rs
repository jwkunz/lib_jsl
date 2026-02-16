use std::collections::VecDeque;

use num::Complex;

use crate::{
    dsp::{
        stream_operator::{StreamOperator, StreamOperatorManagement},
    },
    ffts::{
        best_fft::BestFft,
        fft_engine_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
    },
    prelude::{ErrorsJSL, IsAnalytic},
};

/// Streaming FIR filter implemented with overlap-add FFT convolution.
///
/// `block_len` controls the processing chunk size used internally by OLA.
/// Larger blocks generally improve throughput at the cost of latency.
pub struct OverlapAddFir<T: IsAnalytic> {
    taps_len: usize,
    block_len: usize,
    fft_len: usize,
    taps_fft: Vec<Complex<f64>>,
    overlap: Vec<Complex<f64>>,
    pending: VecDeque<T>,
    fwd: BestFft,
    inv: BestFft,
    processed_any: bool,
}

impl<T: IsAnalytic> OverlapAddFir<T> {
    pub fn new(taps: &[T], block_len: usize) -> Result<Self, ErrorsJSL> {
        if taps.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("FIR taps must be non-empty"));
        }
        if block_len == 0 {
            return Err(ErrorsJSL::InvalidInputRange("block_len must be > 0"));
        }

        let taps_len = taps.len();
        let fft_len = (block_len + taps_len - 1).next_power_of_two();
        let mut h_time = vec![Complex::new(0.0, 0.0); fft_len];
        for (i, &tap) in taps.iter().enumerate() {
            h_time[i] = tap.to_complex();
        }

        let mut fwd = BestFft::new();
        fwd.plan(
            fft_len,
            FftScaleFactor::None,
            FftDirection::Forward,
            FftOrdering::Standard,
        )?;
        let taps_fft = fwd.execute(&h_time)?;

        let mut inv = BestFft::new();
        inv.plan(
            fft_len,
            FftScaleFactor::N,
            FftDirection::Inverse,
            FftOrdering::Standard,
        )?;

        Ok(Self {
            taps_len,
            block_len,
            fft_len,
            taps_fft,
            overlap: vec![Complex::new(0.0, 0.0); taps_len.saturating_sub(1)],
            pending: VecDeque::new(),
            fwd,
            inv,
            processed_any: false,
        })
    }

    fn convolve_block_time_domain(&mut self, block: &[T]) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        let mut x_time = vec![Complex::new(0.0, 0.0); self.fft_len];
        for (i, &x) in block.iter().enumerate() {
            x_time[i] = x.to_complex();
        }

        let x_fft = self.fwd.execute(&x_time)?;
        let y_fft = x_fft
            .iter()
            .zip(self.taps_fft.iter())
            .map(|(&x, &h)| x * h)
            .collect::<Vec<_>>();
        self.inv.execute(&y_fft)
    }
}

impl<T: IsAnalytic> StreamOperatorManagement for OverlapAddFir<T> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.pending.clear();
        for v in &mut self.overlap {
            *v = Complex::new(0.0, 0.0);
        }
        self.processed_any = false;
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl<T: IsAnalytic> StreamOperator<T, T> for OverlapAddFir<T> {
    fn process(&mut self, data_in: &[T]) -> Result<Option<Vec<T>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }

        for &x in data_in {
            self.pending.push_back(x);
        }

        let mut out = Vec::<T>::new();
        while self.pending.len() >= self.block_len {
            let block: Vec<T> = self.pending.drain(..self.block_len).collect();
            let mut y = self.convolve_block_time_domain(&block)?;

            for i in 0..self.overlap.len() {
                y[i] += self.overlap[i];
            }

            out.extend(y[..self.block_len].iter().copied().map(T::from_complex));
            if !self.overlap.is_empty() {
                let tail_len = self.overlap.len();
                self.overlap
                    .copy_from_slice(&y[self.block_len..self.block_len + tail_len]);
            }
            self.processed_any = true;
        }

        if out.is_empty() {
            Ok(None)
        } else {
            Ok(Some(out))
        }
    }

    fn flush(&mut self) -> Result<Option<Vec<T>>, ErrorsJSL> {
        if !self.processed_any && self.pending.is_empty() {
            return Ok(None);
        }

        let mut out = Vec::<T>::new();
        if !self.pending.is_empty() {
            let remainder_len = self.pending.len();
            let mut block = vec![T::zero(); self.block_len];
            for dst in block.iter_mut().take(remainder_len) {
                *dst = self.pending.pop_front().ok_or(ErrorsJSL::RuntimeError(
                    "Internal pending underflow while flushing overlap-add FIR",
                ))?;
            }

            let mut y = self.convolve_block_time_domain(&block)?;
            for i in 0..self.overlap.len() {
                y[i] += self.overlap[i];
            }

            let final_len = remainder_len + self.taps_len - 1;
            out.extend(y[..final_len].iter().copied().map(T::from_complex));
            for v in &mut self.overlap {
                *v = Complex::new(0.0, 0.0);
            }
        } else if !self.overlap.is_empty() {
            out.extend(self.overlap.iter().copied().map(T::from_complex));
            for v in &mut self.overlap {
                *v = Complex::new(0.0, 0.0);
            }
        }

        self.processed_any = false;

        if out.is_empty() {
            Ok(None)
        } else {
            Ok(Some(out))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::convolve::{convolve, ConvolveMethod, ConvolveMode};

    fn close_vec(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < tol, "x={x}, y={y}");
        }
    }

    fn close_vec_c(a: &[Complex<f64>], b: &[Complex<f64>], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.re - y.re).abs() < tol && (x.im - y.im).abs() < tol);
        }
    }

    #[test]
    fn test_overlap_add_fir_stream_matches_direct_full_real() {
        let taps = vec![0.2, -0.1, 0.4, 0.05, -0.2];
        let input = vec![0.3, -1.0, 0.2, 0.5, 0.8, -0.7, 1.2, 0.0, -0.3, 0.9, 0.4];

        let mut fir = OverlapAddFir::<f64>::new(&taps, 4).unwrap();
        let mut stream_out = Vec::new();
        stream_out.extend(fir.process(&input[0..3]).unwrap().unwrap_or_default());
        stream_out.extend(fir.process(&input[3..7]).unwrap().unwrap_or_default());
        stream_out.extend(fir.process(&input[7..]).unwrap().unwrap_or_default());
        stream_out.extend(fir.flush().unwrap().unwrap_or_default());

        let golden = convolve(&input, &taps, ConvolveMode::Full, ConvolveMethod::Direct).unwrap();
        close_vec(&stream_out, &golden, 1e-9);
    }

    #[test]
    fn test_overlap_add_fir_stream_matches_direct_full_complex() {
        let taps = vec![
            Complex::new(0.5, 0.0),
            Complex::new(-0.2, 0.3),
            Complex::new(0.1, -0.1),
        ];
        let input = vec![
            Complex::new(1.0, 0.2),
            Complex::new(-0.5, 0.0),
            Complex::new(0.3, -0.8),
            Complex::new(0.7, 0.1),
            Complex::new(-0.1, 0.9),
            Complex::new(0.0, -0.2),
        ];

        let mut fir = OverlapAddFir::<Complex<f64>>::new(&taps, 4).unwrap();
        let mut stream_out = Vec::new();
        stream_out.extend(fir.process(&input[0..2]).unwrap().unwrap_or_default());
        stream_out.extend(fir.process(&input[2..5]).unwrap().unwrap_or_default());
        stream_out.extend(fir.process(&input[5..]).unwrap().unwrap_or_default());
        stream_out.extend(fir.flush().unwrap().unwrap_or_default());

        let golden = convolve(&input, &taps, ConvolveMode::Full, ConvolveMethod::Direct).unwrap();
        close_vec_c(&stream_out, &golden, 1e-9);
    }

    #[test]
    fn test_overlap_add_fir_reset() {
        let taps = vec![1.0, 0.5, -0.25];
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut fir = OverlapAddFir::<f64>::new(&taps, 4).unwrap();

        let _ = fir.process(&input).unwrap();
        let _ = fir.flush().unwrap();
        fir.reset().unwrap();

        let mut out = Vec::new();
        out.extend(fir.process(&input).unwrap().unwrap_or_default());
        out.extend(fir.flush().unwrap().unwrap_or_default());
        let golden = convolve(&input, &taps, ConvolveMode::Full, ConvolveMethod::Direct).unwrap();
        close_vec(&out, &golden, 1e-9);
    }
}
