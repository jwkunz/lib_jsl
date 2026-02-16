use std::collections::VecDeque;

use crate::{
    dsp::{
        filters::firwin::{FirwinPassZero, firwin},
        stream_operator::{StreamOperator, StreamOperatorManagement},
        windows::{WindowType, kaiser_beta, kaiser_estimate_numtaps},
    },
    number_theory::greatest_common_divisor::gcd,
    prelude::{ErrorsJSL, IsAnalytic},
};

/// Streaming integer-ratio polyphase resampler.
///
/// This implements upsample -> FIR -> downsample without explicitly creating
/// the zero-stuffed sequence, and supports chunked streaming input.
pub struct PolyphaseIntegerResampler<T: IsAnalytic> {
    up_rate: usize,
    down_rate: usize,
    taps: Vec<f64>,
    history: VecDeque<Option<T>>,
    counter: usize,
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

        let mut taps = if let Some(t) = fir_taps {
            if t.is_empty() {
                return Err(ErrorsJSL::InvalidInputRange("fir_taps must be non-empty"));
            }
            t.to_vec()
        } else {
            let max_rate = up.max(down);
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

        while taps.len() < up {
            taps.push(0.0); // Pad taps to at least the up_rate for proper polyphase decomposition
        }
        let history = VecDeque::from_iter(vec![None; taps.len()]); // Start with empty history

        // Match common resample_poly behavior: scale taps by upsampling factor.
        let up_scale = up as f64;
        for h in &mut taps {
            *h *= up_scale;
        }

        Ok(Self {
            up_rate: up,
            down_rate: down,
            taps,
            history,
            counter: 0,
        })
    }

    /// This step function processes one input sample at a time, updating the history and counter, and producing output samples whenever enough input has been accumulated according to the resampling ratio.
    /// It applies the FIR filter taps to the current history of input samples to compute each output sample.
    fn step(&mut self, input: T) -> Option<Vec<T>> {
        // Shift in the new sample and update the history and counter.
        // The history is a sliding window of the most recent input samples, and the counter keeps track of how many samples have been processed since the last output sample.
        self.history.pop_front();
        self.history.push_back(Some(input));
        self.counter += 1;
        // Pad the history with None for the up_rate - 1 samples that would be inserted in a zero-stuffed sequence, and update the counter accordingly.
        // This simulates the effect of upsampling by inserting zeros between input samples without actually creating a larger intermediate buffer.
        for _ in 1..self.up_rate {
            self.history.pop_front();
            self.history.push_back(None);
            self.counter += 1;
        }
        let mut result = Vec::new();
        // Whenever the counter indicates that enough input has been accumulated to produce an output sample (i.e., counter >= down_rate), we compute the output sample by applying the FIR filter taps to the current history of input samples.
        // We then decrement the counter by the down_rate to account for the output sample that was just produced, and repeat this process until the counter is less than the down_rate, which means we need to wait for more input samples before producing the next output sample.
        while self.counter >= self.down_rate {
            self.counter -= self.down_rate;
            let mut acc = T::zero();
            for (h, x) in self.taps.iter().zip(self.history.iter()) {
                if let Some(x) = x {
                    acc += x.f_scale(*h);
                }
            }
            result.push(acc);
        }
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }
}

impl<T: IsAnalytic> StreamOperatorManagement for PolyphaseIntegerResampler<T> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.history.iter_mut().for_each(|x| *x = None);
        self.counter = 0;
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl<T: IsAnalytic> StreamOperator<T, T> for PolyphaseIntegerResampler<T> {
    fn process(&mut self, data_in: &[T]) -> Result<Option<Vec<T>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        let result = data_in
            .iter()
            .filter_map(|&x| self.step(x))
            .flatten()
            .collect();
        Ok(Some(result))
    }

    fn flush(&mut self) -> Result<Option<Vec<T>>, ErrorsJSL> {
        let data_in = vec![T::zero(); self.taps.len()]; // Flush with zeros to clear the history
        self.process(&data_in)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_polyphase_integer_resampler() {
        let mut resampler = PolyphaseIntegerResampler::new(3, 9, Some(&vec![1.0])).unwrap();
        let input = (0..10).map(|x| x as f64).collect::<Vec<f64>>();
        let output = resampler.process(&input).unwrap().unwrap();
        assert_eq!(output, vec![2.0, 5.0, 8.0]);
    }
}
