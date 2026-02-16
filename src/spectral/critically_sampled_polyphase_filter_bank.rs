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

/// Critically sampled uniform DFT analysis polyphase filter bank.
///
/// Input: serial complex samples.
/// Output: one complex spectrum (`C1D`) per processed decimated block.
pub struct CriticallySampledPolyphaseAnalysisFilterBank {
    channels: usize,
    taps: Vec<f64>,
    phases: usize,
    history: VecDeque<Complex<f64>>,
    pending: VecDeque<Complex<f64>>,
    fft: BestFft,
}

impl CriticallySampledPolyphaseAnalysisFilterBank {
    pub fn new(channels: usize, taps: Option<&[f64]>) -> Result<Self, ErrorsJSL> {
        if channels < 2 || !channels.is_power_of_two() {
            return Err(ErrorsJSL::InvalidInputRange(
                "channels must be a power of two and >= 2",
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

        let mut fft = BestFft::new();
        fft.plan(
            channels,
            FftScaleFactor::None,
            FftDirection::Forward,
            FftOrdering::Standard,
        )?;

        Ok(Self {
            channels,
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
        let spec = self.fft.execute(&poly)?;
        Ok(Array1::from_vec(spec))
    }
}

impl StreamOperatorManagement for CriticallySampledPolyphaseAnalysisFilterBank {
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

impl StreamOperator<Complex<f64>, C1D> for CriticallySampledPolyphaseAnalysisFilterBank {
    fn process(&mut self, data_in: &[Complex<f64>]) -> Result<Option<Vec<C1D>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        self.pending.extend(data_in.iter().copied());

        let mut out = Vec::<C1D>::new();
        while self.pending.len() >= self.channels {
            for _ in 0..self.channels {
                let x = self.pending.pop_front().ok_or(ErrorsJSL::RuntimeError(
                    "internal pending underflow in analysis filter bank",
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
        while self.pending.len() < self.channels {
            self.pending.push_back(Complex::new(0.0, 0.0));
        }
        let mut out = Vec::<C1D>::new();
        while self.pending.len() >= self.channels {
            for _ in 0..self.channels {
                let x = self.pending.pop_front().ok_or(ErrorsJSL::RuntimeError(
                    "internal pending underflow in analysis filter bank flush",
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

/// Critically sampled uniform DFT synthesis polyphase filter bank.
///
/// Input: sequence of channelized FFT bins (`C1D`), one frame per decimated time step.
/// Output: serial complex samples.
pub struct CriticallySampledPolyphaseSynthesisFilterBank {
    channels: usize,
    taps: Vec<f64>,
    phases: usize,
    frame_history: VecDeque<Vec<Complex<f64>>>,
    ifft: BestFft,
}

impl CriticallySampledPolyphaseSynthesisFilterBank {
    pub fn new(channels: usize, taps: Option<&[f64]>) -> Result<Self, ErrorsJSL> {
        if channels < 2 || !channels.is_power_of_two() {
            return Err(ErrorsJSL::InvalidInputRange(
                "channels must be a power of two and >= 2",
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

        let mut ifft = BestFft::new();
        ifft.plan(
            channels,
            FftScaleFactor::N,
            FftDirection::Inverse,
            FftOrdering::Standard,
        )?;

        Ok(Self {
            channels,
            taps,
            phases,
            frame_history: VecDeque::from(vec![vec![Complex::new(0.0, 0.0); channels]; phases]),
            ifft,
        })
    }
}

impl StreamOperatorManagement for CriticallySampledPolyphaseSynthesisFilterBank {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.frame_history.iter_mut().for_each(|f| {
            f.iter_mut().for_each(|v| *v = Complex::new(0.0, 0.0));
        });
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<C1D, Complex<f64>> for CriticallySampledPolyphaseSynthesisFilterBank {
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
                "failed to access contiguous synthesis frame",
            ))?)?;
            self.frame_history.pop_front();
            self.frame_history.push_back(time_poly);

            for n in 0..self.channels {
                let mut y = Complex::new(0.0, 0.0);
                for r in 0..self.phases {
                    let h_idx = r * self.channels + n;
                    let frame_idx = self.phases - 1 - r;
                    y += self.frame_history[frame_idx][n] * self.taps[h_idx];
                }
                out.push(y);
            }
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
        self.process(&zeros)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_synthesis_roundtrip_delta_taps() {
        let m = 8usize;
        let taps = vec![1.0; m]; // K=1, critically sampled test path

        let mut ana = CriticallySampledPolyphaseAnalysisFilterBank::new(m, Some(&taps)).unwrap();
        let mut syn = CriticallySampledPolyphaseSynthesisFilterBank::new(m, Some(&taps)).unwrap();

        let input = (0..(5 * m))
            .map(|n| Complex::new((n as f64).sin(), (n as f64 * 0.5).cos()))
            .collect::<Vec<_>>();

        let mut frames = Vec::<C1D>::new();
        frames.extend(ana.process(&input[..13]).unwrap().unwrap_or_default());
        frames.extend(ana.process(&input[13..31]).unwrap().unwrap_or_default());
        frames.extend(ana.process(&input[31..]).unwrap().unwrap_or_default());
        frames.extend(ana.flush().unwrap().unwrap_or_default());

        let mut output = Vec::<Complex<f64>>::new();
        output.extend(syn.process(&frames[..2]).unwrap().unwrap_or_default());
        output.extend(syn.process(&frames[2..4]).unwrap().unwrap_or_default());
        output.extend(syn.process(&frames[4..]).unwrap().unwrap_or_default());

        assert_eq!(output.len(), input.len());
        for (y, x) in output.iter().zip(input.iter()) {
            assert!((y.re - x.re).abs() < 1e-9 && (y.im - x.im).abs() < 1e-9);
        }
    }

    #[test]
    fn test_default_taps_smoke() {
        let m = 8usize;
        let mut ana = CriticallySampledPolyphaseAnalysisFilterBank::new(m, None).unwrap();
        let x = vec![Complex::new(1.0, 0.0); 4 * m];
        let frames = ana.process(&x).unwrap().unwrap();
        assert_eq!(frames.len(), 4);
        assert!(frames.iter().all(|f| f.len() == m));
    }
}
