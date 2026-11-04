//! Signal sources: numerically controlled oscillators, CORDIC and periodic waveforms.

use lib_jsl_dsp::discrete::sources::cordic::{Cordic32, cordic_sin_cos};
use lib_jsl_dsp::discrete::sources::nco::{Nco, NcoSource};
use lib_jsl_dsp::discrete::sources::quadrature::{Polarity, QuadratureWave};
use lib_jsl_dsp::discrete::sources::sinusoid::SineWave;
use lib_jsl_dsp::discrete::sources::square::SquareWave;

fn main() {
    // A 1 kHz tone at 48 kHz: the 64-bit phase accumulator steps by f/fs of a turn.
    let mut nco = Nco::new(1000.0, 48000.0).unwrap();
    for _ in 0..12 {
        nco.next_phase();
    }
    println!("phase after 12 samples: {:.6} turns (exact 0.25)", nco.phase().turns());
    let mut sine = NcoSource::<SineWave>::sinusoid(1000.0, 48000.0).unwrap();
    println!("sine: {:.4?}", sine.generate(4));
    let mut iq = NcoSource::<QuadratureWave>::quadrature(1000.0, 48000.0, Polarity::Positive).unwrap();
    println!("quadrature: {}", iq.generate(2).iter().map(|c| format!("{c:.4}")).collect::<Vec<_>>().join(", "));
    let mut square = NcoSource::<SquareWave>::square_with_duty(12000.0, 48000.0, 0.25).unwrap();
    println!("square, duty 0.25: {:?}", square.generate(8));
    let (s, c) = cordic_sin_cos(0.5);
    println!("CORDIC sin/cos(0.5) = {s:.9}, {c:.9} (exact {:.9}, {:.9})", 0.5f64.sin(), 0.5f64.cos());
    let coarse = Cordic32::new(10).unwrap();
    println!("10-iteration CORDIC: sin(0.5) = {:.5}, gain constant {:.6}", coarse.sin_cos(0.5).0, coarse.gain());
}
