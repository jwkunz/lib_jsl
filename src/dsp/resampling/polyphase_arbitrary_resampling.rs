use std::collections::VecDeque;

use crate::{
    dsp::{
        filters::firwin::{FirwinPassZero, firwin},
        stream_operator::{StreamOperator, StreamOperatorManagement},
        windows::{WindowType, kaiser_beta, kaiser_estimate_numtaps},
    },
    prelude::{ErrorsJSL, IsAnalytic},
};

/// Streaming arbitrary-ratio polyphase resampler approximation.
///
/// The upsampling rate is integer, and the downsampling rate is floating-point.
/// Internally this uses a floating sample counter, so output timing is an
/// approximation that improves as `up_rate` increases.
pub struct PolyphaseArbitraryResampler<T: IsAnalytic> {
    up_rate: usize,
    down_rate: f64,
    taps: Vec<f64>,
    history: VecDeque<Option<T>>,
    counter: f64,
}

impl<T: IsAnalytic> PolyphaseArbitraryResampler<T> {
    pub fn new(
        up_rate: usize,
        down_rate: f64,
        fir_taps: Option<&[f64]>,
    ) -> Result<Self, ErrorsJSL> {
        if up_rate == 0 {
            return Err(ErrorsJSL::InvalidInputRange("up_rate must be > 0"));
        }
        if !down_rate.is_finite() || down_rate <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange(
                "down_rate must be finite and > 0",
            ));
        }
        let max_rate = up_rate.max(down_rate.ceil() as usize);
        let mut taps = if let Some(t) = fir_taps {
            if t.is_empty() {
                return Err(ErrorsJSL::InvalidInputRange("fir_taps must be non-empty"));
            }
            t.to_vec()
        } else {
            let cut_off = 0.5 / max_rate as f64;
            let mut numtaps = kaiser_estimate_numtaps(60.0, cut_off).max(max_rate);
            numtaps = ((numtaps as f64) / up_rate as f64).ceil() as usize * up_rate; // Round up to a multiple of up_rate for better efficiency
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

        while taps.len() < max_rate {
            taps.push(0.0);
        }

        for h in &mut taps {
            *h *= up_rate as f64;
        }

        let history = VecDeque::from_iter(vec![None; taps.len()]);
        Ok(Self {
            up_rate,
            down_rate,
            taps,
            history,
            counter: 0.0,
        })
    }

    fn step(&mut self, input: T) -> Option<Vec<T>> {
        // Shift in the new sample and update the history and counter.
        // The history is a sliding window of the most recent input samples, and the counter keeps track of how many samples have been processed since the last output sample.
        self.history.pop_front();
        self.history.push_back(Some(input));
        self.counter += 1.0;
        // Pad the history with None for the up_rate - 1 samples that would be inserted in a zero-stuffed sequence, and update the counter accordingly.
        // This simulates the effect of upsampling by inserting zeros between input samples without actually creating a larger intermediate buffer.
        for _ in 1..self.up_rate {
            self.history.pop_front();
            self.history.push_back(None);
            self.counter += 1.0;
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

impl<T: IsAnalytic> StreamOperatorManagement for PolyphaseArbitraryResampler<T> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.history.iter_mut().for_each(|x| *x = None);
        self.counter = 0.0;
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl<T: IsAnalytic> StreamOperator<T, T> for PolyphaseArbitraryResampler<T> {
    fn process(&mut self, data_in: &[T]) -> Result<Option<Vec<T>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        let out: Vec<T> = data_in
            .iter()
            .filter_map(|&x| self.step(x))
            .flatten()
            .collect();
        if out.is_empty() {
            Ok(None)
        } else {
            Ok(Some(out))
        }
    }

    fn flush(&mut self) -> Result<Option<Vec<T>>, ErrorsJSL> {
        let zeros = vec![T::zero(); self.taps.len()];
        self.process(&zeros)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polyphase_arbitrary_resampler() {
        let mut resampler = PolyphaseArbitraryResampler::new(1, 3.0, Some(&vec![1.0])).unwrap();
        let input = (0..10).map(|x| x as f64).collect::<Vec<f64>>();
        let output = resampler.process(&input).unwrap().unwrap();
        assert_eq!(output, vec![0.0, 3.0, 6.0]);
    }
}
