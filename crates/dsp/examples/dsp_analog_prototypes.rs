//! Analogue prototypes, frequency transformations and the analogue frequency response.

use lib_jsl_dsp::continuous::filter::design::{buttap, cheb1ap, ellipap, lp2bp_zpk, zpk_to_tf};
use lib_jsl_dsp::continuous::spectral::freqs::{FreqsWorN, freqs};

fn main() {
    let butter = buttap(3).unwrap();
    println!("Butterworth poles: {:.4}", butter.p);
    let tf = zpk_to_tf(&butter).unwrap();
    println!("Butterworth denominator: {:.4}", tf.a.mapv(|c| c.re));
    let w = vec![0.5, 1.0, 2.0];
    for (name, zpk) in [("Butterworth", buttap(3).unwrap()), ("Chebyshev I, 1 dB", cheb1ap(3, 1.0).unwrap()), ("elliptic, 1/40 dB", ellipap(3, 1.0, 40.0).unwrap())] {
        let tf = zpk_to_tf(&zpk).unwrap();
        let r = freqs(tf.b.as_slice().unwrap(), tf.a.as_slice().unwrap(), Some(FreqsWorN::Frequencies(w.clone()))).unwrap();
        let gains: Vec<String> = r.h.iter().map(|h| format!("{:7.2}", 20.0 * h.norm().log10())).collect();
        println!("{name:<18} gain at ω = 0.5, 1, 2: {} dB", gains.join(" "));
    }
    // A bandpass centred on 1000 rad/s with 200 rad/s bandwidth: twice as many poles.
    let bp = lp2bp_zpk(&buttap(3).unwrap(), 1000.0, 200.0).unwrap();
    println!("bandpass: {} poles, {} zeros", bp.p.len(), bp.z.len());
}
