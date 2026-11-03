//! FIR design three ways: Kaiser window, Parks-McClellan and frequency sampling.

use lib_jsl_dsp::discrete::filters::firwin::{FirwinPassZero, firwin};
use lib_jsl_dsp::discrete::filters::firwin2::firwin2;
use lib_jsl_dsp::discrete::filters::remez::{RemezFilterType, remez};
use lib_jsl_dsp::discrete::spectral::freqz::{FreqzWorN, freqz};
use lib_jsl_dsp::discrete::windows::{WindowType, kaiser_beta, kaiser_estimate_numtaps};

/// The smallest stopband attenuation (dB) from `from` to Nyquist (frequencies × Nyquist).
fn stopband(h: &[f64], from: f64) -> f64 {
    let w: Vec<f64> = (0..2000).map(|i| std::f64::consts::PI * (from + (1.0 - from) * i as f64 / 1999.0)).collect();
    let r = freqz(h, None, Some(FreqzWorN::Frequencies(w)), None, None, None).unwrap();
    r.h.iter().map(|v| -20.0 * v.norm().log10()).fold(f64::INFINITY, f64::min)
}

fn main() {
    // Lowpass: pass below 0.2, stop above 0.3 (× Nyquist), 60 dB.
    let n = kaiser_estimate_numtaps(60.0, 0.1) | 1;
    let kaiser = firwin(n, &[0.25], None, WindowType::Kaiser { beta: kaiser_beta(60.0) }, FirwinPassZero::True, true, 2.0).unwrap();
    println!("Kaiser window: {n} taps, stopband {:.1} dB", stopband(&kaiser, 0.3));
    for taps in [41, 51, 61] {
        let pm = remez(taps, &[0.0, 0.2, 0.3, 1.0], &[1.0, 0.0], Some(&[1.0, 10.0]), RemezFilterType::Bandpass, 25, 16, 2.0).unwrap();
        println!("Parks-McClellan: {taps} taps, stopband {:.1} dB", stopband(&pm, 0.3));
    }
    // An arbitrary response: a gentle high-frequency boost.
    let boost = firwin2(41, &[0.0, 0.5, 1.0], &[1.0, 1.0, 2.0], None, Some(WindowType::Hamming), false, 2.0).unwrap();
    let r = freqz(&boost, None, Some(FreqzWorN::Frequencies(vec![0.0, 0.5 * std::f64::consts::PI, 0.9 * std::f64::consts::PI])), None, None, None).unwrap();
    println!("firwin2 boost: gains {:.3} at 0, 0.5 and 0.9 × Nyquist", r.h.mapv(|v| v.norm()));
}
