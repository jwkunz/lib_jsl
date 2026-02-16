/// Receiver front-end that mixes a channel to baseband and then resamples.

use num::Complex;

use crate::{
    dsp::{
        resampling::polyphase_arbitrary_resampling::PolyphaseArbitraryResampler,
        stream_operator::{StreamOperator, StreamOperatorManagement},
        transformations::frequency_mixer::FrequencyMixer,
    },
    prelude::ErrorsJSL,
};

/// Receiver front-end that mixes a channel to baseband and then resamples.
///
/// Processing order:
/// 1) Frequency shift (`FrequencyMixer`)
/// 2) Arbitrary polyphase resample (`PolyphaseArbitraryResampler`)
pub struct ChannelReceiver {
    mixer: FrequencyMixer,
    resampler_complex: PolyphaseArbitraryResampler<Complex<f64>>,
}

impl ChannelReceiver {
    pub fn new(
        frequency_shift_hz: f64,
        sample_rate_hz: f64,
        up_rate: usize,
        down_rate: f64,
        fir_taps: Option<&[f64]>,
    ) -> Result<Self, ErrorsJSL> {
        if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange(
                "sample_rate_hz must be finite and > 0",
            ));
        }

        Ok(Self {
            mixer: FrequencyMixer::new(frequency_shift_hz, sample_rate_hz, None),
            resampler_complex: PolyphaseArbitraryResampler::new(up_rate, down_rate, fir_taps)?,
        })
    }
}

impl StreamOperatorManagement for ChannelReceiver {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.mixer.reset()?;
        self.resampler_complex.reset()?;
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        self.mixer.finalize()?;
        self.resampler_complex.finalize()?;
        Ok(())
    }
}

impl StreamOperator<f64, Complex<f64>> for ChannelReceiver {
    fn process(&mut self, data_in: &[f64]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        let mixed = match <FrequencyMixer as StreamOperator<f64, Complex<f64>>>::process(
            &mut self.mixer,
            data_in,
        )? {
            Some(v) => v,
            None => return Ok(None),
        };
        self.resampler_complex.process(&mixed)
    }

    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        self.resampler_complex.flush()
    }
}

impl StreamOperator<Complex<f64>, Complex<f64>> for ChannelReceiver {
    fn process(&mut self, data_in: &[Complex<f64>]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        let mixed = match self.mixer.process(data_in)? {
            Some(v) => v,
            None => return Ok(None),
        };
        self.resampler_complex.process(&mixed)
    }

    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        self.resampler_complex.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_receiver_f64_to_complex() {
        let taps = [1.0];
        let mut dut = ChannelReceiver::new(0.0, 48_000.0, 1, 2.0, Some(&taps)).unwrap();
        let x = (0..8).map(|v| v as f64).collect::<Vec<_>>();
        let mut y =
            <ChannelReceiver as StreamOperator<f64, Complex<f64>>>::process(&mut dut, &x)
                .unwrap()
                .unwrap_or_default();
        y.extend(
            <ChannelReceiver as StreamOperator<f64, Complex<f64>>>::flush(&mut dut)
                .unwrap()
                .unwrap_or_default(),
        );
        let expected = vec![
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        assert_eq!(y, expected);
    }

    #[test]
    fn test_channel_receiver_complex_to_complex() {
        let taps = [1.0];
        let mut dut = ChannelReceiver::new(0.0, 48_000.0, 1, 2.0, Some(&taps)).unwrap();
        let x = (0..8)
            .map(|v| Complex::new(v as f64, -(v as f64)))
            .collect::<Vec<_>>();
        let mut y =
            <ChannelReceiver as StreamOperator<Complex<f64>, Complex<f64>>>::process(&mut dut, &x)
                .unwrap()
                .unwrap_or_default();
        y.extend(
            <ChannelReceiver as StreamOperator<Complex<f64>, Complex<f64>>>::flush(&mut dut)
                .unwrap()
                .unwrap_or_default(),
        );
        let expected = vec![
            Complex::new(0.0, -0.0),
            Complex::new(2.0, -2.0),
            Complex::new(4.0, -4.0),
            Complex::new(6.0, -6.0),
            Complex::new(0.0, 0.0),
        ];
        assert_eq!(y, expected);
    }
}
