/// Transmitter front-end that resamples then frequency shifts.
///
/// Processing order:
/// 1) Arbitrary polyphase resample (`PolyphaseArbitraryResampler`)
/// 2) Frequency shift (`FrequencyMixer`)
///
/// The internal path is always complex and output is always complex.
/// 
use num::Complex;

use crate::{
    dsp::{
        resampling::polyphase_arbitrary_resampling::PolyphaseArbitraryResampler,
        stream_operator::{StreamOperator, StreamOperatorManagement},
        transformations::frequency_mixer::FrequencyMixer,
    },
    prelude::ErrorsJSL,
};


pub struct ChannelTransmitter {
    resampler_complex: PolyphaseArbitraryResampler<Complex<f64>>,
    mixer: FrequencyMixer,
}

impl ChannelTransmitter {
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
            resampler_complex: PolyphaseArbitraryResampler::new(up_rate, down_rate, fir_taps)?,
            mixer: FrequencyMixer::new(frequency_shift_hz, sample_rate_hz, None),
        })
    }
}

impl StreamOperatorManagement for ChannelTransmitter {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        self.resampler_complex.reset()?;
        self.mixer.reset()?;
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        self.resampler_complex.finalize()?;
        self.mixer.finalize()?;
        Ok(())
    }
}

impl StreamOperator<f64, Complex<f64>> for ChannelTransmitter {
    fn process(&mut self, data_in: &[f64]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        let x = data_in
            .iter()
            .map(|&v| Complex::new(v, 0.0))
            .collect::<Vec<_>>();
        let y = match self.resampler_complex.process(&x)? {
            Some(v) => v,
            None => return Ok(None),
        };
        self.mixer.process(&y)
    }

    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        let y = match self.resampler_complex.flush()? {
            Some(v) => v,
            None => return Ok(None),
        };
        self.mixer.process(&y)
    }
}

impl StreamOperator<Complex<f64>, Complex<f64>> for ChannelTransmitter {
    fn process(&mut self, data_in: &[Complex<f64>]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }
        let y = match self.resampler_complex.process(data_in)? {
            Some(v) => v,
            None => return Ok(None),
        };
        self.mixer.process(&y)
    }

    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        let y = match self.resampler_complex.flush()? {
            Some(v) => v,
            None => return Ok(None),
        };
        self.mixer.process(&y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_transmitter_f64_to_complex() {
        let taps = [1.0];
        let mut dut = ChannelTransmitter::new(0.0, 48_000.0, 1, 2.0, Some(&taps)).unwrap();
        let x = (0..8).map(|v| v as f64).collect::<Vec<_>>();
        let mut y =
            <ChannelTransmitter as StreamOperator<f64, Complex<f64>>>::process(&mut dut, &x)
                .unwrap()
                .unwrap_or_default();
        y.extend(
            <ChannelTransmitter as StreamOperator<f64, Complex<f64>>>::flush(&mut dut)
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
    fn test_channel_transmitter_complex_to_complex() {
        let taps = [1.0];
        let mut dut = ChannelTransmitter::new(0.0, 48_000.0, 1, 2.0, Some(&taps)).unwrap();
        let x = (0..8)
            .map(|v| Complex::new(v as f64, -(v as f64)))
            .collect::<Vec<_>>();
        let mut y = <ChannelTransmitter as StreamOperator<Complex<f64>, Complex<f64>>>::process(
            &mut dut, &x,
        )
        .unwrap()
        .unwrap_or_default();
        y.extend(
            <ChannelTransmitter as StreamOperator<Complex<f64>, Complex<f64>>>::flush(&mut dut)
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

