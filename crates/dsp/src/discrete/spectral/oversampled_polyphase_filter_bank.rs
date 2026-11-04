//! Oversampled uniform DFT polyphase analysis and synthesis filter banks.
//!
//! See the book entry [Polyphase filter banks](https://jwkunz.github.io/lib_jsl/book/spectral/filter-banks.html#critical-sampling-and-oversampling).

use std::collections::VecDeque;

use ndarray::Array1;
use num_complex::Complex;

use super::polyphase_synthesis::{DftSynthesis, default_synthesis_filter};
use crate::{
    dsp::discrete::{
        filters::firwin::{FirwinPassZero, firwin},
        stream_operator::{StreamOperator, StreamOperatorManagement},
        windows::{WindowType, kaiser_beta, kaiser_estimate_numtaps},
    },
    ffts::{
        Fft,
        fft_engine_trait::{FftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
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
    fft: Fft,
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
            || !channels.is_multiple_of(oversampling_factor)
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

        let mut fft = Fft::new();
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
                let s_off = r * self.channels + (self.channels - 1 - k);
                // The prototype tap for the sample s_off places before the newest one.
                let t_idx = s_off;
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
/// Input: channelized FFT frames (`C1D`) from the matching analysis bank.
/// Output: serial complex samples, `channels / oversampling_factor` per frame.
///
/// Each frame is inverted by an inverse DFT and overlap-added through a synthesis filter `f`
/// (see `polyphase_synthesis` for the derivation). The default `f` depends on the prototype:
///
/// - one tap per phase (`taps.len() == channels`): a rectangle, which reconstructs exactly when the
///   prototype overlap-adds to a constant at the hop (a rectangular prototype does), with no delay;
/// - longer prototypes, including the default: a 60 dB Kaiser lowpass of the prototype's length
///   with cutoff `1/(2·hop)`. With the default prototype (whose shifted copies add to one) the bank
///   reconstructs its input to within about 60 dB, delayed by `taps.len() − channels` samples,
///   for oversampling factors of 2 or more.
///
/// A different filter can be set with [`with_synthesis_filter`](Self::with_synthesis_filter);
/// reconstruction then needs `Σ_q H(ν − q/M) F(ν − q/M)` constant, no overlap between `F` and the
/// prototype's images at multiples of `1/hop`, and a combined delay of `h` and `f` that is a
/// multiple of `channels`. Scale it so that `Σ f = hop / Σ h` for unit gain.
pub struct OversampledPolyphaseSynthesisFilterBank {
    engine: DftSynthesis,
}

impl OversampledPolyphaseSynthesisFilterBank {
    /// A synthesis bank matching the analysis bank built with the same arguments.
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
            || !channels.is_multiple_of(oversampling_factor)
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
        let hop = channels / oversampling_factor;
        let filter = default_synthesis_filter(&taps, channels, hop)?;
        Ok(Self { engine: DftSynthesis::new(channels, hop, filter)? })
    }

    /// Replaces the synthesis filter.
    pub fn with_synthesis_filter(mut self, filter: &[f64]) -> Result<Self, ErrorsJSL> {
        self.engine.set_filter(filter.to_vec())?;
        Ok(self)
    }

    /// The synthesis filter in use.
    pub fn synthesis_filter(&self) -> &[f64] {
        self.engine.filter()
    }
}

impl StreamOperatorManagement for OversampledPolyphaseSynthesisFilterBank {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.engine.reset();
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl StreamOperator<C1D, Complex<f64>> for OversampledPolyphaseSynthesisFilterBank {
    fn process(&mut self, data_in: &[C1D]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        let mut out = Vec::<Complex<f64>>::new();
        for frame in data_in {
            let slice = frame.as_slice().ok_or(ErrorsJSL::RuntimeError("failed to access contiguous oversampled synthesis frame"))?;
            self.engine.push_frame(slice, &mut out)?;
        }
        Ok(if out.is_empty() { None } else { Some(out) })
    }

    /// Emits the tail of the synthesis filter after the last frame; the stream ends here.
    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        let out = self.engine.flush();
        Ok(if out.is_empty() { None } else { Some(out) })
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

    /// Runs `input` through analysis and synthesis and returns the output.
    fn round_trip(m: usize, os: usize, taps: Option<&[f64]>, input: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let mut ana = OversampledPolyphaseAnalysisFilterBank::new(m, os, taps).unwrap();
        let mut syn = OversampledPolyphaseSynthesisFilterBank::new(m, os, taps).unwrap();
        let mut frames = ana.process(input).unwrap().unwrap_or_default();
        frames.extend(ana.flush().unwrap().unwrap_or_default());
        let mut out = syn.process(&frames).unwrap().unwrap_or_default();
        out.extend(syn.flush().unwrap().unwrap_or_default());
        out
    }

    #[test]
    fn the_default_prototype_reconstructs_to_60_db() {
        // White-ish deterministic input; the output must be the input delayed by
        // taps.len() - channels, with unit gain, to about the prototype's 60 dB stopband.
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let input: Vec<Complex<f64>> = (0..4096).map(|_| Complex::new(rnd(), rnd())).collect();
        for (m, os) in [(16usize, 2usize), (16, 4), (32, 2), (8, 8)] {
            let delay = design_default_taps(m).unwrap().len() - m;
            let out = round_trip(m, os, None, &input);
            let (mut err, mut sig) = (0.0, 0.0);
            for i in 1000..3000 {
                err += (out[i + delay] - input[i]).norm_sqr();
                sig += input[i].norm_sqr();
            }
            let rel = (err / sig).sqrt();
            assert!(rel < 2e-3, "M = {m}, oversampling {os}: relative error {rel:.2e}");
        }
    }

    #[test]
    fn a_custom_synthesis_filter_is_used() {
        let syn = OversampledPolyphaseSynthesisFilterBank::new(8, 2, None).unwrap().with_synthesis_filter(&[0.5; 16]).unwrap();
        assert_eq!(syn.synthesis_filter(), &[0.5; 16]);
        assert!(OversampledPolyphaseSynthesisFilterBank::new(8, 2, None).unwrap().with_synthesis_filter(&[]).is_err());
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
