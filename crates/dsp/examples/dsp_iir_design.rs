//! Digital IIR design: choose the order, design, check the response and filter a signal.

use lib_jsl_dsp::continuous::filter::design::IirFilterBandType;
use lib_jsl_dsp::discrete::filters::butterworth::ButterOutput;
use lib_jsl_dsp::discrete::filters::discrete_linear_filter::DiscreteLinearFilter;
use lib_jsl_dsp::discrete::filters::elliptic::{ellip, ellipord};
use lib_jsl_dsp::discrete::spectral::freqz::{FreqzWorN, freqz};

fn main() {
    // Sampling at 8 kHz: pass up to 1 kHz with 0.5 dB ripple, reject above 1.3 kHz by 60 dB.
    let fs = 8000.0;
    let spec = ellipord(&[1000.0], &[1300.0], 0.5, 60.0, IirFilterBandType::Lowpass, Some(fs)).unwrap();
    println!("elliptic order {} (critical frequency {:.1} Hz)", spec.order, spec.wn[0]);
    let ButterOutput::Ba(ba) = ellip(spec.order, 0.5, 60.0, &spec.wn, IirFilterBandType::Lowpass, None, Some(fs)).unwrap() else { unreachable!() };
    let (b, a): (Vec<f64>, Vec<f64>) = (ba.beta.iter().map(|c| c.re).collect(), ba.alpha.iter().map(|c| c.re).collect());
    let r = freqz(&b, Some(&a), Some(FreqzWorN::Frequencies(vec![500.0, 1000.0, 1300.0, 2000.0])), None, Some(fs), None).unwrap();
    for (f, h) in r.w.iter().zip(&r.h) {
        println!("{f:6.0} Hz: {:7.2} dB", 20.0 * h.norm().log10());
    }
    // Filter 400 Hz + 2 kHz: the 2 kHz tone should disappear.
    let x: Vec<f64> = (0..800).map(|n| (2.0 * std::f64::consts::PI * 400.0 * n as f64 / fs).sin() + (2.0 * std::f64::consts::PI * 2000.0 * n as f64 / fs).sin()).collect();
    let mut filter = DiscreteLinearFilter::new(&b, &a);
    let y: Vec<f64> = x.iter().map(|v| filter.step(v)).collect();
    let rms = |v: &[f64]| (v.iter().map(|s| s * s).sum::<f64>() / v.len() as f64).sqrt();
    println!("RMS in {:.3} (two unit tones), out {:.3} (one tone: 0.707)", rms(&x[400..]), rms(&y[400..]));
}
