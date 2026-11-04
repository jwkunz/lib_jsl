//! Pulse compression with a linear-FM chirp, Barker codes, and pulse tracking.

use lib_jsl_dsp::discrete::digital_communications::barker_codes::{barker_autocorrelation, barker_code};
use lib_jsl_dsp::discrete::digital_communications::pulse_correlation_tracker::locate_pulse;
use lib_jsl_dsp::discrete::radar::pulse_compression::{linear_chirp, peak_sidelobe_level_db, pulse_compress};
use num_complex::Complex;

fn main() {
    let pulse = linear_chirp(5e6, 20e-6, 20e6).unwrap();
    println!("chirp: {} samples, time-bandwidth product {}", pulse.len(), 5e6 * 20e-6);
    let mut rx = vec![Complex::new(0.0, 0.0); 2000];
    for (k, p) in pulse.iter().enumerate() {
        rx[700 + k] += p;
    }
    let c = pulse_compress(&rx, &pulse, None).unwrap();
    let peak = (0..c.len()).max_by(|&i, &j| c[i].norm().total_cmp(&c[j].norm())).unwrap();
    println!("compressed peak at sample {peak}, gain {:.0} (pulse energy), sidelobes {:.1} dB", c[peak].norm(), peak_sidelobe_level_db(&c));
    let code = barker_code(13).unwrap();
    println!("Barker 13 {code:?}");
    println!("autocorrelation {:?}", barker_autocorrelation(&code));
    let frame: Vec<Complex<f64>> = (0..1000).map(|n| if n >= 300 && n < 300 + pulse.len() { pulse[n - 300] } else { Complex::new(0.0, 0.0) }).collect();
    let found = locate_pulse(&frame, &pulse).unwrap();
    println!("locate_pulse: delay {:.2}, normalised correlation {:.3}", found.delay, found.normalized);
}
