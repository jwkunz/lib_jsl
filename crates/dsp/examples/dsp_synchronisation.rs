//! Carrier and timing synchronisation: PLL, Costas loop, Mueller-Müller timing recovery and Farrow interpolation.

use lib_jsl_dsp::discrete::digital_communications::constellations::Constellation;
use lib_jsl_dsp::discrete::digital_communications::farrow_filter::farrow_filter;
use lib_jsl_dsp::discrete::digital_communications::mueller_timing_detector::MuellerTimingRecovery;
use lib_jsl_dsp::discrete::stream_operator::StreamOperator;
use lib_jsl_dsp::discrete::synchronization::costas_loop::{CostasLoop, CostasModulation};
use lib_jsl_dsp::discrete::synchronization::phase_locked_loop::PhaseLockedLoop;
use num_complex::Complex;
use std::f64::consts::PI;

fn main() {
    let mut pll = PhaseLockedLoop::new(0.02, 0.707).unwrap();
    for n in 0..2000 {
        pll.step(Complex::from_polar(1.0, 2.0 * PI * 0.01 * n as f64 + 1.0));
    }
    println!("PLL frequency {:.5} cycles/sample (true 0.01000)", pll.frequency() / (2.0 * PI));
    // BPSK symbols with a phase offset and a slow frequency offset.
    let mut costas = CostasLoop::new(CostasModulation::Bpsk, 0.02, 0.707).unwrap();
    let mut state = 1u64;
    let out: Vec<Complex<f64>> = (0..3000)
        .map(|n| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let s = if state >> 63 == 1 { 1.0 } else { -1.0 };
            costas.step(Complex::from_polar(s, 0.5 + 0.002 * n as f64)).0
        })
        .collect();
    println!("Costas: residual imaginary part {:.1e}, phase ambiguity ±1 on the real axis: {:.3}", out[2999].im.abs(), out[2999].re);
    // A fractional delay of 0.25 samples with cubic Lagrange interpolation.
    let x: Vec<f64> = (0..50).map(|n| (0.2 * n as f64).sin()).collect();
    let y = farrow_filter(&x, 0.25, 3).unwrap();
    println!("Farrow delay 0.25: y[20] = {:.5}, sin(0.2 × 19.75) = {:.5}", y[20], (0.2f64 * 19.75).sin());
    // Timing recovery on an oversampled QPSK stream with a timing offset.
    let qpsk = Constellation::qpsk();
    let symbols: Vec<Complex<f64>> = (0..800).map(|k| qpsk.points()[(k * 7 + k / 3) % 4]).collect();
    let mut stream = Vec::new();
    for (k, &s) in symbols.iter().enumerate() {
        let next = symbols.get(k + 1).copied().unwrap_or(s);
        for i in 0..4 {
            let t = (i as f64 + 0.3) / 4.0;
            stream.push(s * (1.0 - t) + next * t); // a linear "pulse": sampled 0.3 samples late
        }
    }
    let mut mm = MuellerTimingRecovery::new(4.0, qpsk.clone()).unwrap();
    let rec = mm.process(&stream).unwrap().unwrap();
    let late: f64 = rec[600..].iter().map(|y| (y - qpsk.slice(*y)).norm()).fold(0.0, f64::max);
    println!("Mueller-Müller: {} symbols out, worst error after settling {late:.3}", rec.len());
}
