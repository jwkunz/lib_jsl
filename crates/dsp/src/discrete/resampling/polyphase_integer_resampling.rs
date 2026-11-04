use std::collections::VecDeque;

use crate::{
    dsp::discrete::{
        filters::firwin::{FirwinPassZero, firwin},
        stream_operator::{StreamOperator, StreamOperatorManagement},
        windows::{WindowType, kaiser_beta, kaiser_estimate_numtaps},
    },
    number_theory::greatest_common_divisor::gcd,
    prelude::{ErrorsJSL, IsAnalytic},
};

/// The streaming polyphase engine shared by the integer and arbitrary-ratio resamplers.
///
/// It computes `y[m] = Σ_j h[p_m − j·up] x[j]` with `p_m = ⌊m · down⌋`: the input upsampled by
/// `up` (zeros between samples), filtered by `h`, and sampled every `down` positions, exactly as
/// SciPy's `upfirdn` but without forming the zero-stuffed signal. Only the taps that meet real
/// samples are multiplied, so each output costs about `len(h) / up` multiply-adds.
#[derive(Debug, Clone)]
pub(crate) struct PolyphaseEngine<T: IsAnalytic> {
    up: usize,
    down: f64,
    pub(crate) taps: Vec<f64>,
    /// The most recent inputs, oldest first; `history[0]` is input number `first`.
    history: VecDeque<T>,
    first: usize,
    /// Inputs received so far.
    received: usize,
    /// Index of the next output.
    next: usize,
}

impl<T: IsAnalytic> PolyphaseEngine<T> {
    pub(crate) fn new(up: usize, down: f64, taps: Vec<f64>) -> Self {
        PolyphaseEngine { up, down, taps, history: VecDeque::new(), first: 0, received: 0, next: 0 }
    }

    /// The upsampled-domain position of output `m`.
    fn position(&self, m: usize) -> usize {
        (m as f64 * self.down + 1e-9).floor() as usize
    }

    /// Emits every output whose inputs are all available, considering only positions below `end`.
    fn drain(&mut self, end: usize, out: &mut Vec<T>) {
        let len = self.taps.len();
        loop {
            let p = self.position(self.next);
            if p >= end || self.received == 0 || p / self.up > self.received - 1 {
                break;
            }
            // Inputs j with 0 <= p − j·up < len.
            let hi = p / self.up;
            let lo = if p + 1 > len { (p + 1 - len).div_ceil(self.up) } else { 0 };
            let mut acc = T::zero();
            for j in lo..=hi {
                acc += self.history[j - self.first].f_scale(self.taps[p - j * self.up]);
            }
            out.push(acc);
            self.next += 1;
        }
        // Forget inputs that no future output can reach.
        let p = self.position(self.next);
        let needed_from = if p + 1 > len { (p + 1 - len).div_ceil(self.up) } else { 0 };
        while self.first < needed_from && !self.history.is_empty() {
            self.history.pop_front();
            self.first += 1;
        }
    }

    pub(crate) fn process(&mut self, data_in: &[T]) -> Vec<T> {
        let mut out = Vec::new();
        for &x in data_in {
            self.history.push_back(x);
            self.received += 1;
            self.drain(usize::MAX, &mut out);
        }
        out
    }

    /// The remaining outputs of the full filtered signal (up to position `(N − 1)·up + len(h)`),
    /// after which the engine is reset.
    pub(crate) fn flush(&mut self) -> Vec<T> {
        let mut out = Vec::new();
        if self.received > 0 {
            let end = (self.received - 1) * self.up + self.taps.len();
            while self.position(self.next) < end {
                self.history.push_back(T::zero());
                self.received += 1;
                self.drain(end, &mut out);
            }
        }
        self.reset();
        out
    }

    pub(crate) fn reset(&mut self) {
        self.history.clear();
        self.first = 0;
        self.received = 0;
        self.next = 0;
    }
}

/// Streaming rational-ratio polyphase resampler: upsample by `up_rate`, low-pass filter, keep every
/// `down_rate`-th sample (`up_rate / down_rate` is reduced to lowest terms). The output equals
/// SciPy's `upfirdn(up · h, x, up, down)`, the core of `resample_poly`, and arrives as soon as the
/// inputs it needs have; `flush` emits the rest of the filter's tail. Without explicit taps a
/// Kaiser-window low-pass filter with cutoff `1 / max(up, down)` of Nyquist and 60 dB attenuation
/// is designed.
pub struct PolyphaseIntegerResampler<T: IsAnalytic> {
    engine: PolyphaseEngine<T>,
}

impl<T: IsAnalytic> PolyphaseIntegerResampler<T> {
    /// The `new` method initializes a `PolyphaseIntegerResampler` instance with the specified upsampling rate, downsampling rate, and optional FIR filter taps.
    /// If FIR filter taps are not provided, it designs a low-pass FIR filter using the Kaiser window method, with a cutoff frequency determined by the maximum of the upsampling and downsampling rates.
    /// The number of taps is estimated based on the desired attenuation and the width of the transition band,
    pub fn new(
        up_rate: usize,
        down_rate: usize,
        fir_taps: Option<&[f64]>,
    ) -> Result<Self, ErrorsJSL> {
        if up_rate == 0 || down_rate == 0 {
            return Err(ErrorsJSL::InvalidInputRange(
                "up_rate and down_rate must both be > 0",
            ));
        }

        let g = gcd(up_rate as i128, down_rate as i128) as usize;
        let up = up_rate / g;
        let down = down_rate / g;
        let max_rate = up.max(down);
        let mut taps = if let Some(t) = fir_taps {
            if t.is_empty() {
                return Err(ErrorsJSL::InvalidInputRange("fir_taps must be non-empty"));
            }
            t.to_vec()
        } else {
            let cut_off = 0.5 / max_rate as f64;
            let mut numtaps = kaiser_estimate_numtaps(60.0, cut_off).max(max_rate);
            numtaps = ((numtaps as f64) / up as f64).ceil() as usize * up; // Round up to a multiple of up_rate for better efficiency
            let beta = kaiser_beta(60.0);
            firwin(
                numtaps,
                &[cut_off],
                None,
                WindowType::Kaiser { beta },
                FirwinPassZero::True,
                true,
                1.0,
            )?
        };

        // Match resample_poly: scale the taps by the upsampling factor to keep unit gain.
        for h in &mut taps {
            *h *= up as f64;
        }
        Ok(Self { engine: PolyphaseEngine::new(up, down as f64, taps) })
    }

}

impl<T: IsAnalytic> StreamOperatorManagement for PolyphaseIntegerResampler<T> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.engine.reset();
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl<T: IsAnalytic> StreamOperator<T, T> for PolyphaseIntegerResampler<T> {
    fn process(&mut self, data_in: &[T]) -> Result<Option<Vec<T>>, ErrorsJSL> {
        let out = self.engine.process(data_in);
        Ok(if out.is_empty() { None } else { Some(out) })
    }

    fn flush(&mut self) -> Result<Option<Vec<T>>, ErrorsJSL> {
        let out = self.engine.flush();
        Ok(if out.is_empty() { None } else { Some(out) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SciPy's upfirdn, written out directly.
    fn upfirdn(h: &[f64], x: &[f64], up: usize, down: usize) -> Vec<f64> {
        let total = (x.len() - 1) * up + h.len();
        (0..total.div_ceil(down))
            .map(|m| {
                let p = m * down;
                (0..x.len()).filter(|&j| j * up <= p && p - j * up < h.len()).map(|j| h[p - j * up] * x[j]).sum()
            })
            .collect()
    }

    #[test]
    fn decimation_with_a_unit_filter_keeps_every_third_sample() {
        let mut resampler = PolyphaseIntegerResampler::new(3, 9, Some(&[1.0])).unwrap();
        let input = (0..10).map(|x| x as f64).collect::<Vec<f64>>();
        assert_eq!(resampler.process(&input).unwrap().unwrap(), vec![0.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn interpolates_between_samples_and_matches_upfirdn_in_any_chunking() {
        // Linear interpolation by 3 (triangular taps; the resampler multiplies them by 3).
        let taps = [1.0 / 9.0, 2.0 / 9.0, 3.0 / 9.0, 2.0 / 9.0, 1.0 / 9.0];
        let mut r = PolyphaseIntegerResampler::new(3, 1, Some(&taps)).unwrap();
        let y = r.process(&[0.0, 3.0, 6.0, 9.0]).unwrap().unwrap();
        // After the two-sample delay of the filter, the ramp 0, 1, 2, … appears.
        assert_eq!(y.len(), 12);
        assert!(y[2..].iter().enumerate().all(|(i, v)| (v - i as f64).abs() < 1e-12), "{y:?}");
        // 3/2 resampling of a longer signal, fed in uneven chunks, equals upfirdn exactly.
        let x: Vec<f64> = (0..50).map(|n| (0.3 * n as f64).sin() + 0.01 * (n * n) as f64).collect();
        let h: Vec<f64> = (0..17).map(|k| 1.0 / (1.0 + (k as f64 - 8.0).powi(2))).collect();
        let mut r = PolyphaseIntegerResampler::new(3, 2, Some(&h)).unwrap();
        let mut got = Vec::new();
        for chunk in x.chunks(7) {
            got.extend(r.process(chunk).unwrap().unwrap_or_default());
        }
        got.extend(r.flush().unwrap().unwrap_or_default());
        let scaled: Vec<f64> = h.iter().map(|v| 3.0 * v).collect();
        let want = upfirdn(&scaled, &x, 3, 2);
        assert_eq!(got.len(), want.len());
        assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-12));
    }
}
