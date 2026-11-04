//! Changing the sampling rate: polyphase resamplers, a half-band decimator and a CIC filter.

use lib_jsl_dsp::discrete::resampling::cic::CicDecimator;
use lib_jsl_dsp::discrete::resampling::half_band::{HalfBandDecimator, halfband_lowpass};
use lib_jsl_dsp::discrete::resampling::polyphase_arbitrary_resampling::PolyphaseArbitraryResampler;
use lib_jsl_dsp::discrete::resampling::polyphase_integer_resampling::PolyphaseIntegerResampler;
use lib_jsl_dsp::discrete::stream_operator::StreamOperator;

fn main() {
    // 44.1 kHz to 48 kHz is 160/147.
    let x: Vec<f64> = (0..4410).map(|n| (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / 44100.0).sin()).collect();
    let mut r = PolyphaseIntegerResampler::new(160, 147, None).unwrap();
    let mut y = r.process(&x).unwrap().unwrap_or_default();
    y.extend(r.flush().unwrap().unwrap_or_default());
    let rms = |v: &[f64]| (v.iter().map(|s| s * s).sum::<f64>() / v.len() as f64).sqrt();
    println!("44.1 → 48 kHz: {} samples in, {} out; RMS {:.4} (sine: 0.7071)", x.len(), y.len(), rms(&y[1000..4000]));
    let mut a = PolyphaseArbitraryResampler::new(64, 64.0 * 1.25, None).unwrap();
    println!("arbitrary 1/1.25: {} outputs from {}", a.process(&x).unwrap().unwrap().len(), x.len());
    let taps = halfband_lowpass(23, 7.0).unwrap();
    let mut hb = HalfBandDecimator::new(&taps).unwrap();
    println!("half-band: {} taps, {} zero; {} outputs", taps.len(), taps.iter().filter(|t| **t == 0.0).count(), hb.process(&x).len());
    let mut cic = CicDecimator::new(16, 4, 1).unwrap();
    let out = cic.process(&vec![100; 640]);
    println!("CIC R = 16, N = 4: steady output {} = 100 × gain {}", out.last().unwrap(), cic.gain());
}
