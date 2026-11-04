//! The synthesis stage shared by the uniform DFT filter banks.
//!
//! The analysis banks emit, every `D` input samples, the spectrum
//!
//! `X_q(t) = Σ_s h[s] w^{q(s+1)} x[m_t − s]`, with `w = e^{2πi/M}`,
//!
//! where `h` is the prototype and `m_t = (t + 1)D − 1` is the newest input sample of frame `t`.
//! The synthesis bank inverts each frame with an unnormalised inverse DFT,
//! `z_t[r] = Σ_q X_q(t) w^{qr}`, and overlap-adds it through a synthesis filter `f`:
//!
//! `y[(t + 1)D − M + v] += f[v] · z_t[v mod M]`,  `v = 0, …, len(f) − 1`.
//!
//! Substituting gives `y[n] = M Σ_t f[u_t − u_0] Σ_{s ≡ −u_t (mod M)} h[s] x[n − u_t − s]` with
//! `u_t = n − m_t`. In the frequency domain the output is
//! `Y(ν) = (1/D) Σ_a X(ν − a/D) Σ_q H(ν − q/M − a/D) F(ν − q/M)` (up to a linear phase), so the bank
//! reconstructs its input when
//!
//! 1. `Σ_q H(ν − q/M) F(ν − q/M)` is constant (the channels add up flat), and
//! 2. `H(ν − a/D) F(ν) = 0` for `a ≠ 0 mod D` (no aliasing),
//!
//! with the combined delay of `h` and `f` a multiple of `M`. For a prototype whose shifted copies add
//! to one (a lowpass with its −6 dB point at `1/(2M)`, as the default is) and `D ≤ M/2`, a lowpass
//! `f` with cutoff `1/(2D)` meets both conditions to within the filters' stopband attenuation.
//! For a critically sampled bank (`D = M`) condition 2 cannot be met by any FIR `f` unless each
//! polyphase component of `h` is a single tap, which is a theorem about DFT filter banks, not a
//! limitation of this code.

use std::collections::VecDeque;

use num_complex::Complex;

use crate::{
    dsp::discrete::{
        filters::firwin::{FirwinPassZero, firwin},
        windows::{WindowType, kaiser_beta},
    },
    ffts::{
        Fft,
        fft_engine_trait::{FftDirection, FftEngine1D, FftOrdering, FftScaleFactor},
    },
    prelude::ErrorsJSL,
};

/// The default synthesis filter for a prototype `h` with `channels` channels and hop `hop`,
/// scaled so that the bank has unit gain: `Σ f = hop / Σ h`.
///
/// - One tap per phase (`h.len() == channels`): a rectangle of length `channels`, which
///   reconstructs exactly when `h` overlap-adds to a constant at the hop (a rectangular `h` does).
/// - Critically sampled (`hop == channels`): the prototype's own shape, the best simple choice,
///   though aliasing then limits the reconstruction.
/// - Oversampled: a Kaiser-window lowpass of the prototype's length with cutoff `1/(2·hop)`
///   (60 dB stopband), passing every channel's band and rejecting the spectral images that
///   decimation by `hop` creates.
pub(crate) fn default_synthesis_filter(prototype: &[f64], channels: usize, hop: usize) -> Result<Vec<f64>, ErrorsJSL> {
    let h0: f64 = prototype.iter().sum();
    if h0.abs() < 1e-12 {
        return Err(ErrorsJSL::InvalidInputRange("The prototype filter must have a nonzero DC gain."));
    }
    let shape = if prototype.len() == channels {
        vec![1.0; channels]
    } else if hop == channels {
        prototype.to_vec()
    } else {
        let cutoff = (0.5 / hop as f64).min(0.5 - 0.25 / channels as f64);
        firwin(prototype.len(), &[cutoff], None, WindowType::Kaiser { beta: kaiser_beta(60.0) }, FirwinPassZero::True, true, 1.0)?
    };
    let sum: f64 = shape.iter().sum();
    let target = hop as f64 / h0;
    Ok(shape.iter().map(|v| v * target / sum).collect())
}

/// Inverse DFT and overlap-add through a synthesis filter, one frame per `hop` output samples.
pub(crate) struct DftSynthesis {
    channels: usize,
    hop: usize,
    filter: Vec<f64>,
    ifft: Fft,
    /// Frames processed so far (`t` of the next frame).
    frames: usize,
    /// Partial output sums, `acc[0]` being output sample `next_out`.
    acc: VecDeque<Complex<f64>>,
    next_out: usize,
}

impl DftSynthesis {
    pub(crate) fn new(channels: usize, hop: usize, filter: Vec<f64>) -> Result<Self, ErrorsJSL> {
        if filter.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("The synthesis filter must not be empty."));
        }
        let mut ifft = Fft::new();
        ifft.plan(channels, FftScaleFactor::None, FftDirection::Inverse, FftOrdering::Standard)?;
        Ok(DftSynthesis { channels, hop, filter, ifft, frames: 0, acc: VecDeque::new(), next_out: 0 })
    }

    pub(crate) fn filter(&self) -> &[f64] {
        &self.filter
    }

    pub(crate) fn set_filter(&mut self, filter: Vec<f64>) -> Result<(), ErrorsJSL> {
        if filter.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("The synthesis filter must not be empty."));
        }
        self.filter = filter;
        self.reset();
        Ok(())
    }

    pub(crate) fn reset(&mut self) {
        self.frames = 0;
        self.acc.clear();
        self.next_out = 0;
    }

    /// Adds one frame and appends the output samples it completes to `out`.
    pub(crate) fn push_frame(&mut self, frame: &[Complex<f64>], out: &mut Vec<Complex<f64>>) -> Result<(), ErrorsJSL> {
        if frame.len() != self.channels {
            return Err(ErrorsJSL::InvalidInputRange("all synthesis input frames must have length = channels"));
        }
        // ANCHOR: overlap_add
        let z = self.ifft.execute(frame)?;
        // Output index of filter tap 0 for this frame: (t + 1)·hop − M (may be negative at the start).
        let start = ((self.frames + 1) * self.hop) as isize - self.channels as isize;
        for (v, &fv) in self.filter.iter().enumerate() {
            let n = start + v as isize;
            if n < self.next_out as isize {
                continue;
            }
            let i = n as usize - self.next_out;
            if i >= self.acc.len() {
                self.acc.resize(i + 1, Complex::new(0.0, 0.0));
            }
            self.acc[i] += z[v % self.channels] * fv;
        }
        self.frames += 1;
        // Later frames only touch outputs from (t + 2)·hop − M on, so everything before is final.
        let complete = ((self.frames + 1) * self.hop).saturating_sub(self.channels);
        while self.next_out < complete {
            out.push(self.acc.pop_front().unwrap_or(Complex::new(0.0, 0.0)));
            self.next_out += 1;
        }
        // ANCHOR_END: overlap_add
        Ok(())
    }

    /// Emits the remaining partial sums (the tail of the last frames' filters).
    pub(crate) fn flush(&mut self) -> Vec<Complex<f64>> {
        let tail: Vec<Complex<f64>> = self.acc.drain(..).collect();
        self.next_out += tail.len();
        tail
    }
}
