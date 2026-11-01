//! FFT-based FIR filtering with the overlap-save method.
//!
//! The filter's frequency response is computed once. Each block of `block_len` new input samples is
//! prefixed with the last `taps - 1` input samples, transformed, multiplied by the response and
//! transformed back; because the prefix absorbs the circular wrap-around, the last `block_len`
//! outputs are exactly the linear-convolution result (the first `taps - 1` are discarded). The cost
//! per sample is `O(log N)` regardless of the filter length, which beats direct filtering for
//! longer filters.
//!
//! The streaming contract matches [`OverlapAddFir`](super::overlap_and_add_fir::OverlapAddFir):
//! output appears one whole block at a time, and [`flush`](StreamOperator::flush) emits the rest
//! plus the filter's `taps - 1` sample tail, so the total output is the full convolution
//! (`input length + taps - 1` samples).

use crate::{
    discrete::stream_operator::{StreamOperator, StreamOperatorManagement},
    ffts::{
        Fft,
        fft_engine_trait::{FftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
    },
    prelude::{ErrorsJSL, IsAnalytic},
};
use num_complex::Complex;
use std::marker::PhantomData;

/// A streaming FIR filter that applies its taps in the frequency domain (overlap-save).
pub struct FftFir<T: IsAnalytic> {
    taps_len: usize,
    block_len: usize,
    fft_len: usize,
    taps_fft: Vec<Complex<f64>>,
    history: Vec<Complex<f64>>,
    pending: Vec<Complex<f64>>,
    fwd: Fft,
    inv: Fft,
    any_input: bool,
    _marker: PhantomData<T>,
}

impl<T: IsAnalytic> FftFir<T> {
    /// Builds a filter from `taps`. `min_block_len` is the smallest number of new samples to
    /// gather before transforming; the FFT size is rounded up to a power of two, and the actual
    /// block length ([`block_len`](Self::block_len)) uses all of it.
    pub fn new(taps: &[T], min_block_len: usize) -> Result<Self, ErrorsJSL> {
        if taps.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("FIR taps must be non-empty"));
        }
        if min_block_len == 0 {
            return Err(ErrorsJSL::InvalidInputRange("block length must be > 0"));
        }
        let taps_len = taps.len();
        let fft_len = (min_block_len + taps_len - 1).next_power_of_two();
        let block_len = fft_len - (taps_len - 1);
        let mut h = vec![Complex::new(0.0, 0.0); fft_len];
        for (dst, tap) in h.iter_mut().zip(taps) {
            *dst = tap.to_complex();
        }
        let mut fwd = Fft::new();
        fwd.plan(fft_len, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard)?;
        let taps_fft = fwd.execute(&h)?;
        let mut inv = Fft::new();
        inv.plan(fft_len, FftScaleFactor::N, FftDirection::Inverse, FftOrdering::Standard)?;
        Ok(FftFir {
            taps_len,
            block_len,
            fft_len,
            taps_fft,
            history: vec![Complex::new(0.0, 0.0); taps_len - 1],
            pending: Vec::new(),
            fwd,
            inv,
            any_input: false,
            _marker: PhantomData,
        })
    }

    /// Number of new input samples consumed per transform.
    pub fn block_len(&self) -> usize {
        self.block_len
    }

    /// The FFT size in use.
    pub fn fft_len(&self) -> usize {
        self.fft_len
    }

    /// Filters one block of `new` (at most `block_len` samples, zero-padded) and returns the first
    /// `valid` outputs.
    fn run_block(&mut self, new: &[Complex<f64>], valid: usize) -> Result<Vec<Complex<f64>>, ErrorsJSL> {
        let mut x = vec![Complex::new(0.0, 0.0); self.fft_len];
        x[..self.taps_len - 1].copy_from_slice(&self.history);
        x[self.taps_len - 1..self.taps_len - 1 + new.len()].copy_from_slice(new);
        let mut y = self.fwd.execute(&x)?;
        for (v, h) in y.iter_mut().zip(&self.taps_fft) {
            *v *= *h;
        }
        let y = self.inv.execute(&y)?;
        // The last `taps - 1` samples of [history, block] become the next history.
        if self.taps_len > 1 {
            self.history.copy_from_slice(&x[self.block_len..self.block_len + self.taps_len - 1]);
        }
        Ok(y[self.taps_len - 1..self.taps_len - 1 + valid].to_vec())
    }
}

impl<T: IsAnalytic> StreamOperatorManagement for FftFir<T> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.history.iter_mut().for_each(|v| *v = Complex::new(0.0, 0.0));
        self.pending.clear();
        self.any_input = false;
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl<T: IsAnalytic> StreamOperator<T, T> for FftFir<T> {
    fn process(&mut self, data_in: &[T]) -> Result<Option<Vec<T>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        self.any_input = true;
        self.pending.extend(data_in.iter().map(|x| x.to_complex()));
        let mut out = Vec::new();
        while self.pending.len() >= self.block_len {
            let block: Vec<Complex<f64>> = self.pending.drain(..self.block_len).collect();
            out.extend(self.run_block(&block, self.block_len)?.into_iter().map(T::from_complex));
        }
        Ok(if out.is_empty() { None } else { Some(out) })
    }

    fn flush(&mut self) -> Result<Option<Vec<T>>, ErrorsJSL> {
        if !self.any_input {
            return Ok(None);
        }
        // Drain what is left, then the filter tail, by feeding zeros through partial blocks.
        let mut rest = std::mem::take(&mut self.pending);
        rest.extend(std::iter::repeat_n(Complex::new(0.0, 0.0), self.taps_len - 1));
        let mut out = Vec::new();
        for chunk in rest.chunks(self.block_len) {
            out.extend(self.run_block(chunk, chunk.len())?.into_iter().map(T::from_complex));
        }
        self.reset()?;
        Ok(if out.is_empty() { None } else { Some(out) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::{
        convolve::{convolve, ConvolveMethod, ConvolveMode},
        filters::fir::overlap_and_add_fir::OverlapAddFir,
    };

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    fn run<T: IsAnalytic>(f: &mut impl StreamOperator<T, T>, x: &[T], chunk: usize) -> Vec<T> {
        let mut out = Vec::new();
        for c in x.chunks(chunk) {
            if let Some(y) = f.process(c).unwrap() {
                out.extend(y);
            }
        }
        if let Some(y) = f.flush().unwrap() {
            out.extend(y);
        }
        out
    }

    #[test]
    fn equals_full_convolution_for_any_chunking() {
        let mut seed = 2;
        for (n, m, block) in [(1000usize, 17usize, 64usize), (333, 1, 16), (50, 31, 8), (4096, 129, 500), (7, 40, 4)] {
            let taps: Vec<f64> = (0..m).map(|_| lcg(&mut seed)).collect();
            let x: Vec<f64> = (0..n).map(|_| lcg(&mut seed)).collect();
            let want = convolve(&x, &taps, ConvolveMode::Full, ConvolveMethod::Direct).unwrap();
            for chunk in [1usize, 7, 64, 1000, n] {
                let mut f = FftFir::new(&taps, block).unwrap();
                let got = run(&mut f, &x, chunk.max(1));
                assert_eq!(got.len(), want.len(), "n={n} m={m} block={block} chunk={chunk}");
                assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-10), "n={n} m={m} block={block} chunk={chunk}");
            }
        }
    }

    #[test]
    fn agrees_with_the_overlap_add_filter() {
        let mut seed = 6;
        let taps: Vec<f64> = (0..45).map(|_| lcg(&mut seed)).collect();
        let x: Vec<f64> = (0..2500).map(|_| lcg(&mut seed)).collect();
        let a = run(&mut FftFir::new(&taps, 200).unwrap(), &x, 333);
        let b = run(&mut OverlapAddFir::new(&taps, 200).unwrap(), &x, 333);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-10));
    }

    #[test]
    fn complex_samples_and_complex_taps() {
        let mut seed = 12;
        let taps: Vec<Complex<f64>> = (0..13).map(|_| Complex::new(lcg(&mut seed), lcg(&mut seed))).collect();
        let x: Vec<Complex<f64>> = (0..300).map(|_| Complex::new(lcg(&mut seed), lcg(&mut seed))).collect();
        let want = convolve(&x, &taps, ConvolveMode::Full, ConvolveMethod::Direct).unwrap();
        let got = run(&mut FftFir::new(&taps, 32).unwrap(), &x, 50);
        assert_eq!(got.len(), want.len());
        assert!(got.iter().zip(&want).all(|(g, w)| (g - w).norm() < 1e-10));
    }

    #[test]
    fn output_arrives_in_whole_blocks_and_reset_clears_state() {
        let taps = [0.25, 0.5, 0.25];
        let mut f = FftFir::new(&taps, 6).unwrap();
        let b = f.block_len();
        assert_eq!(f.fft_len(), 8);
        assert_eq!(b, 6);
        assert!(f.process(&vec![1.0; b - 1]).unwrap().is_none(), "nothing until a block fills");
        assert_eq!(f.process(&[1.0]).unwrap().unwrap().len(), b);
        assert!(f.process(&[]).unwrap().is_none());
        // After a reset the filter forgets its history: a unit impulse reproduces the taps.
        StreamOperatorManagement::reset(&mut f).unwrap();
        let mut impulse = vec![0.0; 10];
        impulse[0] = 1.0;
        let y = run(&mut f, &impulse, 10);
        assert_eq!(y.len(), 12);
        for (g, w) in y.iter().zip([0.25, 0.5, 0.25]) {
            assert!((g - w).abs() < 1e-12);
        }
        assert!(f.flush().unwrap().is_none(), "flushing twice yields nothing more");
    }

    #[test]
    fn invalid_arguments() {
        assert!(FftFir::<f64>::new(&[], 8).is_err());
        assert!(FftFir::new(&[1.0], 0).is_err());
    }
}
