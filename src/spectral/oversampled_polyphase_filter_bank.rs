use std::collections::VecDeque;

use ndarray::Array1;
use num::Complex;

use crate::{
    dsp::{
        filters::firwin::{FirwinPassZero, firwin},
        stream_operator::{StreamOperator, StreamOperatorManagement},
        windows::{WindowType, kaiser_beta, kaiser_estimate_numtaps},
    },
    ffts::{
        best_fft::BestFft,
        fft_engine_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
    },
    prelude::{C1D, ErrorsJSL},
};

fn design_default_taps(channels: usize) -> Result<Vec<f64>, ErrorsJSL> {
    let atten_db = 60.0;
    let width = 1.0 / (2.0 * channels as f64);
    let mut numtaps = kaiser_estimate_numtaps(atten_db, width).max(channels);
    let rem = numtaps % channels;
    if rem != 0 {
        numtaps += channels - rem;
    }
    let beta = kaiser_beta(atten_db);
    firwin(
        numtaps,
        &[0.5 / channels as f64],
        None,
        WindowType::Kaiser { beta },
        FirwinPassZero::True,
        true,
        1.0,
    )
}

/// Oversampled uniform DFT analysis polyphase filter bank.
///
/// Oversampling ratio is controlled by `oversampling_factor` (power-of-two),
/// where hop size is `channels / oversampling_factor`.
pub struct OversampledPolyphaseAnalysisFilterBank {
    channels: usize,
    hop: usize,
    taps: Vec<f64>,
    phases: usize,
    history: VecDeque<Complex<f64>>,
    pending: VecDeque<Complex<f64>>,
    fft: BestFft,
}

impl OversampledPolyphaseAnalysisFilterBank {
    pub fn new(
        channels: usize,
        oversampling_factor: usize,
        taps: Option<&[f64]>,
    ) -> Result<Self, ErrorsJSL> {
        if channels < 2 || !channels.is_power_of_two() {
            return Err(ErrorsJSL::InvalidInputRange(
                "channels must be a power of two and >= 2",
            ));
        }
        if oversampling_factor == 0
            || !oversampling_factor.is_power_of_two()
            || oversampling_factor > channels
            || channels % oversampling_factor != 0
        {
            return Err(ErrorsJSL::InvalidInputRange(
                "oversampling_factor must be a power of two and divide channels",
            ));
        }
        let taps = if let Some(t) = taps {
            if t.is_empty() || t.len() % channels != 0 {
                return Err(ErrorsJSL::InvalidInputRange(
                    "taps must be non-empty and length must be a multiple of channels",
                ));
            }
            t.to_vec()
        } else {
            design_default_taps(channels)?
        };
        let phases = taps.len() / channels;
        let hop = channels / oversampling_factor;

        let mut fft = BestFft::new();
        fft.plan(
            channels,
            FftScaleFactor::None,
            FftDirection::Forward,
            FftOrdering::Standard,
        )?;

        Ok(Self {
            channels,
            hop,
            taps,
            phases,
            history: VecDeque::from(vec![Complex::new(0.0, 0.0); phases * channels]),
            pending: VecDeque::new(),
            fft,
        })
    }

    fn push_sample(&mut self, x: Complex<f64>) {
        self.history.pop_front();
        self.history.push_back(x);
    }

    fn compute_frame(&mut self) -> Result<C1D, ErrorsJSL> {
        let mut poly = vec![Complex::new(0.0, 0.0); self.channels];
        let l = self.history.len();
        for (k, pk) in poly.iter_mut().enumerate().take(self.channels) {
            let mut acc = Complex::new(0.0, 0.0);
            for r in 0..self.phases {
                let t_idx = r * self.channels + k;
                let s_off = r * self.channels + (self.channels - 1 - k);
                let s_idx = l - 1 - s_off;
                acc += self.history[s_idx] * self.taps[t_idx];
            }
            *pk = acc;
        }
        Ok(Array1::from_vec(self.fft.execute(&poly)?))
    }
}

impl StreamOperatorManagement for OversampledPolyphaseAnalysisFilterBank {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.pending.clear();
        self.history
            .iter_mut()
            .for_each(|v| *v = Complex::new(0.0, 0.0));
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<Complex<f64>, C1D> for OversampledPolyphaseAnalysisFilterBank {
    fn process(&mut self, data_in: &[Complex<f64>]) -> Result<Option<Vec<C1D>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        self.pending.extend(data_in.iter().copied());

        let mut out = Vec::<C1D>::new();
        while self.pending.len() >= self.hop {
            for _ in 0..self.hop {
                let x = self.pending.pop_front().ok_or(ErrorsJSL::RuntimeError(
                    "internal pending underflow in oversampled analysis filter bank",
                ))?;
                self.push_sample(x);
            }
            out.push(self.compute_frame()?);
        }
        if out.is_empty() {
            Ok(None)
        } else {
            Ok(Some(out))
        }
    }

    fn flush(&mut self) -> Result<Option<Vec<C1D>>, ErrorsJSL> {
        if self.pending.is_empty() {
            return Ok(None);
        }
        while self.pending.len() < self.hop {
            self.pending.push_back(Complex::new(0.0, 0.0));
        }
        let mut out = Vec::<C1D>::new();
        while self.pending.len() >= self.hop {
            for _ in 0..self.hop {
                let x = self.pending.pop_front().ok_or(ErrorsJSL::RuntimeError(
                    "internal pending underflow in oversampled analysis filter bank flush",
                ))?;
                self.push_sample(x);
            }
            out.push(self.compute_frame()?);
        }
        if out.is_empty() {
            Ok(None)
        } else {
            Ok(Some(out))
        }
    }
}

/// Oversampled uniform DFT synthesis polyphase filter bank.
///
/// Input: channelized FFT frames (`C1D`)
/// Output: serial complex samples with hop `channels / oversampling_factor`.
pub struct OversampledPolyphaseSynthesisFilterBank {
    channels: usize,
    oversampling_factor: usize,
    hop: usize,
    taps: Vec<f64>,
    phases: usize,
    frame_history: VecDeque<Vec<Complex<f64>>>,
    overlap: Vec<Complex<f64>>,
    ifft: BestFft,
}

impl OversampledPolyphaseSynthesisFilterBank {
    pub fn new(
        channels: usize,
        oversampling_factor: usize,
        taps: Option<&[f64]>,
    ) -> Result<Self, ErrorsJSL> {
        if channels < 2 || !channels.is_power_of_two() {
            return Err(ErrorsJSL::InvalidInputRange(
                "channels must be a power of two and >= 2",
            ));
        }
        if oversampling_factor == 0
            || !oversampling_factor.is_power_of_two()
            || oversampling_factor > channels
            || channels % oversampling_factor != 0
        {
            return Err(ErrorsJSL::InvalidInputRange(
                "oversampling_factor must be a power of two and divide channels",
            ));
        }
        let taps = if let Some(t) = taps {
            if t.is_empty() || t.len() % channels != 0 {
                return Err(ErrorsJSL::InvalidInputRange(
                    "taps must be non-empty and length must be a multiple of channels",
                ));
            }
            t.to_vec()
        } else {
            design_default_taps(channels)?
        };
        let phases = taps.len() / channels;
        let hop = channels / oversampling_factor;

        let mut ifft = BestFft::new();
        ifft.plan(
            channels,
            FftScaleFactor::N,
            FftDirection::Inverse,
            FftOrdering::Standard,
        )?;

        Ok(Self {
            channels,
            oversampling_factor,
            hop,
            taps,
            phases,
            frame_history: VecDeque::from(vec![vec![Complex::new(0.0, 0.0); channels]; phases]),
            overlap: vec![Complex::new(0.0, 0.0); channels - hop],
            ifft,
        })
    }
}

impl StreamOperatorManagement for OversampledPolyphaseSynthesisFilterBank {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.frame_history.iter_mut().for_each(|f| {
            f.iter_mut().for_each(|v| *v = Complex::new(0.0, 0.0));
        });
        self.overlap
            .iter_mut()
            .for_each(|v| *v = Complex::new(0.0, 0.0));
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<C1D, Complex<f64>> for OversampledPolyphaseSynthesisFilterBank {
    fn process(&mut self, data_in: &[C1D]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        let mut out = Vec::<Complex<f64>>::new();
        for frame in data_in {
            if frame.len() != self.channels {
                return Err(ErrorsJSL::InvalidInputRange(
                    "all synthesis input frames must have length = channels",
                ));
            }
            let time_poly = self.ifft.execute(frame.as_slice().ok_or(ErrorsJSL::RuntimeError(
                "failed to access contiguous oversampled synthesis frame",
            ))?)?;

            if self.phases == 1 {
                out.extend(time_poly[self.channels - self.hop..].iter().copied());
                continue;
            }

            self.frame_history.pop_front();
            self.frame_history.push_back(time_poly);

            let mut block = vec![Complex::new(0.0, 0.0); self.channels];
            for (n, bn) in block.iter_mut().enumerate().take(self.channels) {
                let mut y = Complex::new(0.0, 0.0);
                for r in 0..self.phases {
                    let h_idx = r * self.channels + n;
                    let frame_idx = self.phases - 1 - r;
                    y += self.frame_history[frame_idx][n] * self.taps[h_idx];
                }
                // Normalize overlap gain for rectangular prototype steady-state.
                *bn = y / self.oversampling_factor as f64;
            }

            for i in 0..self.overlap.len() {
                block[i] += self.overlap[i];
            }
            out.extend(block[..self.hop].iter().copied());
            self.overlap.copy_from_slice(&block[self.hop..]);
        }

        if out.is_empty() {
            Ok(None)
        } else {
            Ok(Some(out))
        }
    }

    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        let zero_frame = Array1::from_vec(vec![Complex::new(0.0, 0.0); self.channels]);
        let zeros = vec![zero_frame; self.phases.saturating_sub(1)];
        let mut out = self.process(&zeros)?.unwrap_or_default();
        out.extend(self.overlap.iter().copied());
        self.overlap
            .iter_mut()
            .for_each(|v| *v = Complex::new(0.0, 0.0));
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

    #[test]
    fn test_oversampled_analysis_synthesis_roundtrip_near_inverse() {
        let m = 8usize;
        let os = 2usize;
        let taps = vec![1.0; m]; // simple prototype for deterministic test

        let mut ana = OversampledPolyphaseAnalysisFilterBank::new(m, os, Some(&taps)).unwrap();
        let mut syn = OversampledPolyphaseSynthesisFilterBank::new(m, os, Some(&taps)).unwrap();

        let input = (0..(24 * m))
            .map(|n| Complex::new((0.07 * n as f64).sin(), (0.11 * n as f64).cos()))
            .collect::<Vec<_>>();

        let mut frames = Vec::<C1D>::new();
        frames.extend(ana.process(&input[..17]).unwrap().unwrap_or_default());
        frames.extend(ana.process(&input[17..83]).unwrap().unwrap_or_default());
        frames.extend(ana.process(&input[83..]).unwrap().unwrap_or_default());
        frames.extend(ana.flush().unwrap().unwrap_or_default());

        let mut output = Vec::<Complex<f64>>::new();
        output.extend(syn.process(&frames[..8]).unwrap().unwrap_or_default());
        output.extend(syn.process(&frames[8..16]).unwrap().unwrap_or_default());
        output.extend(syn.process(&frames[16..]).unwrap().unwrap_or_default());
        output.extend(syn.flush().unwrap().unwrap_or_default());

        let n = input.len().min(output.len());
        let trim = 2 * m; // ignore startup/shutdown transient
        let start = trim.min(n);
        let end = n.saturating_sub(trim);
        assert!(end > start);
        let mut mse = 0.0;
        let mut count = 0usize;
        for i in start..end {
            let e = output[i] - input[i];
            mse += e.norm_sqr();
            count += 1;
        }
        mse /= count as f64;
        assert!(mse < 1e-6, "mse={mse}");
    }

    #[test]
    fn test_oversampled_default_taps_smoke() {
        let mut ana = OversampledPolyphaseAnalysisFilterBank::new(8, 2, None).unwrap();
        let x = vec![Complex::new(1.0, 0.0); 16];
        let frames = ana.process(&x).unwrap().unwrap();
        assert_eq!(frames.len(), 4); // hop = 4 for M=8, os=2
        assert!(frames.iter().all(|f| f.len() == 8));
    }
}
