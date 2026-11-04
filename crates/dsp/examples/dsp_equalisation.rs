//! Channel impairments and a decision-feedback equalizer.

use lib_jsl_dsp::discrete::digital_communications::constellations::Constellation;
use lib_jsl_dsp::discrete::digital_communications::decision_feedback_equalizer::decision_feedback_equalizer;
use lib_jsl_dsp::discrete::stream_operator::StreamOperator;
use lib_jsl_dsp::discrete::transformations::channel_impairment::ChannelImpairment;
use lib_jsl_dsp::discrete::transformations::non_linearity_transformers::{hard_clip_real, quantize_real};
use num_complex::Complex;

fn main() {
    println!("hard clip of 1.7 at 1.0: {}; 8-level quantisation of 0.33: {:.4}", hard_clip_real(1.7, 1.0), quantize_real(0.33, 8));
    let qpsk = Constellation::qpsk();
    let symbols: Vec<Complex<f64>> = (0..3000).map(|k| qpsk.points()[(k * 13 + k / 7) % 4]).collect();
    // A three-path channel with a little noise (the generator is seeded randomly).
    let mut channel = ChannelImpairment::new(Some(1.0), Some(&[1.0, 0.9, 0.6]), Some(-30.0), None, None).unwrap();
    let mut rx = channel.process(&symbols).unwrap().unwrap_or_default();
    rx.extend(channel.flush().unwrap().unwrap_or_default());
    let rx = &rx[..symbols.len()];
    let raw_errors = rx.iter().zip(&symbols).filter(|(r, s)| (qpsk.slice(**r) - **s).norm() > 1e-9).count();
    let eq = decision_feedback_equalizer(rx, &symbols[..200], qpsk.clone(), 7, 3, 0.05).unwrap();
    let eq_errors = eq.iter().zip(&symbols).skip(500).filter(|(r, s)| (qpsk.slice(**r) - **s).norm() > 1e-9).count();
    println!("symbol errors: {raw_errors} of 3000 without equalisation, {eq_errors} of 2500 after the DFE");
}
