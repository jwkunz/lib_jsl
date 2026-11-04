
use crate::{
    dsp::discrete::{
        filters::firwin::{FirwinPassZero, firwin},
        resampling::polyphase_integer_resampling::PolyphaseEngine,
        stream_operator::{StreamOperator, StreamOperatorManagement},
        windows::{WindowType, kaiser_beta, kaiser_estimate_numtaps},
    },
    prelude::{ErrorsJSL, IsAnalytic},
};

/// Streaming resampler by an integer `up_rate` and a real `down_rate`.
///
/// Output `m` is taken from the upsampled, filtered signal at position `⌊m · down_rate⌋`, so the
/// output times are rounded to the nearest-below point of the `up_rate` grid: the timing error is
/// at most one upsampled sample, and shrinks as `up_rate` grows. With an integer `down_rate` the
/// result equals [`PolyphaseIntegerResampler`](super::polyphase_integer_resampling::PolyphaseIntegerResampler)'s.
pub struct PolyphaseArbitraryResampler<T: IsAnalytic> {
    engine: PolyphaseEngine<T>,
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
        Ok(Self { engine: PolyphaseEngine::new(up_rate, down_rate, taps) })
    }
}

impl<T: IsAnalytic> StreamOperatorManagement for PolyphaseArbitraryResampler<T> {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.engine.reset();
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        Ok(())
    }
}

impl<T: IsAnalytic> StreamOperator<T, T> for PolyphaseArbitraryResampler<T> {
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
    use crate::discrete::resampling::polyphase_integer_resampling::PolyphaseIntegerResampler;

    #[test]
    fn decimation_with_a_unit_filter_keeps_every_third_sample() {
        let mut resampler = PolyphaseArbitraryResampler::new(1, 3.0, Some(&[1.0])).unwrap();
        let input = (0..10).map(|x| x as f64).collect::<Vec<f64>>();
        assert_eq!(resampler.process(&input).unwrap().unwrap(), vec![0.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn integer_ratios_agree_with_the_integer_resampler_and_fractional_ones_track_a_sine() {
        let x: Vec<f64> = (0..400).map(|n| (0.05 * n as f64).sin()).collect();
        let mut a = PolyphaseArbitraryResampler::new(4, 3.0, None).unwrap();
        let mut b = PolyphaseIntegerResampler::new(4, 3, None).unwrap();
        assert_eq!(a.process(&x).unwrap(), b.process(&x).unwrap());
        // Resample by 32/45.7: output m sits at input time ⌊45.7 m⌋ / 32, minus the filter delay.
        let up = 32;
        let down = 45.7;
        let mut r = PolyphaseArbitraryResampler::new(up, down, None).unwrap();
        let y = r.process(&x).unwrap().unwrap();
        let delay = (r.engine.taps.len() - 1) as f64 / 2.0;
        let worst = y.iter().enumerate().skip(20).take(y.len() - 40).map(|(m, v)| {
            let t = ((m as f64 * down).floor() - delay) / up as f64;
            (v - (0.05 * t).sin()).abs()
        }).fold(0.0, f64::max);
        assert!(worst < 2e-3, "{worst}");
    }
}
