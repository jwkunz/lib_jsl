use num::Complex;

use crate::{
    dsp::{
        filters::overlap_and_add_fir::OverlapAddFir,
        stream_operator::{StreamOperator, StreamOperatorManagement},
        transformations::{
            frequency_mixer::FrequencyMixer, non_linearity_transformers::NonLinearityTransformer,
        },
    },
    prelude::ErrorsJSL,
    random::{
        distributions::guassian_distribution_box_muller,
        uniform_generator::UniformRNG,
        xoshiro256plusplus::Xoshiro256PlusPlus,
    },
};

/// Streaming channel impairment model for complex baseband samples.
///
/// Optional stages are applied in this order:
/// 1) Channel convolution
/// 2) Doppler shift
/// 3) Non-linearity
/// 4) Additive white Gaussian noise (AWGN)
pub struct ChannelImpairment {
    sample_rate_hz: Option<f64>,
    channel_fir: Option<OverlapAddFir<Complex<f64>>>,
    noise_power_db_watts_per_hz: Option<f64>,
    doppler_mixer: Option<FrequencyMixer>,
    non_linear_fn: Option<NonLinearityTransformer<Complex<f64>>>,
    rng: UniformRNG,
}

impl ChannelImpairment {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        // Sample rate in Hz, used for interpreting noise power and Doppler shift. Optional if those features are not used.
        sample_rate_hz: Option<f64>,
        // FIR taps for channel convolution. If None, no convolution is applied.
        channel_model: Option<&[f64]>,
        // Power spectral density of AWGN in dB Watts/Hz. If None, no noise is added.
        noise_power_db_watts_per_hz: Option<f64>,
        // Doppler shift in Hz. If None, no Doppler shift is applied.
        doppler_shift_hz: Option<f64>,
        // Optional non-linear function to apply to the signal after Doppler shift and before noise. If None, no non-linearity is applied.
        non_linear_function: Option<NonLinearityTransformer<Complex<f64>>>,
    ) -> Result<Self, ErrorsJSL> {
        if let Some(fs) = sample_rate_hz {
            if !fs.is_finite() || fs <= 0.0 {
                return Err(ErrorsJSL::InvalidInputRange(
                    "sample_rate must be finite and > 0",
                ));
            }
        }
        if let Some(db) = noise_power_db_watts_per_hz {
            if !db.is_finite() {
                return Err(ErrorsJSL::InvalidInputRange(
                    "noise_power_db_watts_per_hz must be finite",
                ));
            }
            if sample_rate_hz.is_none() {
                return Err(ErrorsJSL::Misconfiguration(
                    "sample_rate must be provided when noise_power_db_watts_per_hz is specified",
                ));
            }
        }

        let channel_fir = if let Some(h) = channel_model {
            if h.is_empty() {
                return Err(ErrorsJSL::InvalidInputRange(
                    "channel_model must be non-empty when provided",
                ));
            }
            let taps: Vec<Complex<f64>> = h.iter().map(|&x| Complex::new(x, 0.0)).collect();
            // Modest default block size for streaming throughput.
            Some(OverlapAddFir::new(&taps, 256)?)
        } else {
            None
        };

        let doppler_mixer = if let Some(fd) = doppler_shift_hz {
            let fs = sample_rate_hz.ok_or(ErrorsJSL::Misconfiguration(
                "sample_rate must be provided when doppler_shift_hz is specified",
            ))?;
            Some(FrequencyMixer::new(fd, fs, None))
        } else {
            None
        };

        Ok(Self {
            sample_rate_hz,
            channel_fir,
            noise_power_db_watts_per_hz,
            doppler_mixer,
            non_linear_fn: non_linear_function,
            rng: Xoshiro256PlusPlus::from_random_seed(),
        })
    }

    #[inline]
    fn apply_awgn(&mut self, data: &mut [Complex<f64>]) {
        let Some(noise_db_psd) = self.noise_power_db_watts_per_hz else {
            return;
        };
        // PSD in W/Hz -> linear.
        let n0 = 10.0_f64.powf(noise_db_psd / 10.0);
        let fs = self.sample_rate_hz.unwrap_or(1.0);
        // Complex baseband: total noise power ~= N0*Fs, split equally across I/Q.
        let sigma = (n0 * fs / 2.0).sqrt();

        for x in data.iter_mut() {
            let (n_i, n_q) = guassian_distribution_box_muller(&mut self.rng, 0.0, sigma);
            *x += Complex::new(n_i, n_q);
        }
    }
}

impl StreamOperatorManagement for ChannelImpairment {
    fn reset(&mut self) -> Result<(), ErrorsJSL> {
        if let Some(fir) = &mut self.channel_fir {
            fir.reset()?;
        }
        if let Some(mix) = &mut self.doppler_mixer {
            mix.reset()?;
        }
        if let Some(nl) = &mut self.non_linear_fn {
            nl.reset()?;
        }
        self.rng = Xoshiro256PlusPlus::from_random_seed();
        Ok(())
    }

    fn finalize(&mut self) -> Result<(), ErrorsJSL> {
        if let Some(fir) = &mut self.channel_fir {
            fir.finalize()?;
        }
        if let Some(mix) = &mut self.doppler_mixer {
            mix.finalize()?;
        }
        if let Some(nl) = &mut self.non_linear_fn {
            nl.finalize()?;
        }
        Ok(())
    }
}

impl StreamOperator<Complex<f64>, Complex<f64>> for ChannelImpairment {
    fn process(&mut self, data_in: &[Complex<f64>]) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        if data_in.is_empty() {
            return Ok(None);
        }

        let mut y = data_in.to_vec();

        if let Some(fir) = &mut self.channel_fir {
            y = match fir.process(&y)? {
                Some(v) => v,
                None => return Ok(None),
            };
        }
        if let Some(mix) = &mut self.doppler_mixer {
            y = match mix.process(&y)? {
                Some(v) => v,
                None => return Ok(None),
            };
        }
        if let Some(nl) = &mut self.non_linear_fn {
            y = match nl.process(&y)? {
                Some(v) => v,
                None => return Ok(None),
            };
        }

        self.apply_awgn(&mut y);
        Ok(Some(y))
    }

    fn flush(&mut self) -> Result<Option<Vec<Complex<f64>>>, ErrorsJSL> {
        let Some(fir) = &mut self.channel_fir else {
            return Ok(None);
        };

        let mut y = match fir.flush()? {
            Some(v) => v,
            None => return Ok(None),
        };

        if let Some(mix) = &mut self.doppler_mixer {
            y = match mix.process(&y)? {
                Some(v) => v,
                None => return Ok(None),
            };
        }
        if let Some(nl) = &mut self.non_linear_fn {
            y = match nl.process(&y)? {
                Some(v) => v,
                None => return Ok(None),
            };
        }

        self.apply_awgn(&mut y);
        Ok(Some(y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::{
        convolve::{convolve, ConvolveMethod, ConvolveMode},
        transformations::non_linearity_transformers::hard_clip_complex,
    };

    fn close(a: &[Complex<f64>], b: &[Complex<f64>], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.re - y.re).abs() < tol && (x.im - y.im).abs() < tol);
        }
    }

    #[test]
    fn test_channel_impairment_passthrough_when_all_none() {
        let mut dut = ChannelImpairment::new(None, None, None, None, None).unwrap();
        let x = vec![Complex::new(1.0, 0.5), Complex::new(-0.2, 0.3), Complex::new(0.0, -1.0)];
        let y = dut.process(&x).unwrap().unwrap();
        close(&x, &y, 1e-12);
        assert!(dut.flush().unwrap().is_none());
    }

    #[test]
    fn test_channel_impairment_convolution_matches_direct() {
        let taps = [0.5, -0.2, 0.1];
        let mut dut = ChannelImpairment::new(None, Some(&taps), None, None, None).unwrap();
        let x = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, -1.0),
            Complex::new(-0.5, 0.25),
            Complex::new(0.1, 0.2),
        ];
        let mut y = dut.process(&x).unwrap().unwrap_or_default();
        y.extend(dut.flush().unwrap().unwrap_or_default());

        let h = taps.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>();
        let golden = convolve(&x, &h, ConvolveMode::Full, ConvolveMethod::Direct).unwrap();
        close(&y, &golden, 1e-9);
    }

    #[test]
    fn test_channel_impairment_doppler_and_nonlinearity() {
        fn clip1(x: Complex<f64>) -> Complex<f64> {
            hard_clip_complex(x, 1.0)
        }
        let nl = NonLinearityTransformer::<Complex<f64>>::new(Some(clip1));
        let mut dut =
            ChannelImpairment::new(Some(8_000.0), None, None, Some(1_000.0), Some(nl)).unwrap();
        let x = vec![Complex::new(2.0, 0.0); 8];
        let y = dut.process(&x).unwrap().unwrap();
        assert_eq!(y.len(), x.len());
        for s in y {
            assert!(s.norm() <= 1.0000001);
        }
    }

    #[test]
    fn test_channel_impairment_noise_changes_signal() {
        let mut dut = ChannelImpairment::new(
            Some(8_000.0),
            None,
            Some(-100.0),
            None,
            None,
        )
        .unwrap();
        let x = vec![Complex::new(0.0, 0.0); 128];
        let y = dut.process(&x).unwrap().unwrap();
        assert_eq!(y.len(), x.len());
        let energy: f64 = y.iter().map(|v| v.norm_sqr()).sum();
        assert!(energy > 0.0);
    }
}
