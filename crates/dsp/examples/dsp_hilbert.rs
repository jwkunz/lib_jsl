//! The analytic signal: envelope and instantaneous frequency.

use lib_jsl_dsp::discrete::spectral::hilbert::{analytic_signal, instantaneous_attributes};
use std::f64::consts::PI;

fn main() {
    let fs = 1000.0;
    let x: Vec<f64> = (0..1000).map(|n| (2.0 * PI * 50.0 * n as f64 / fs).cos()).collect();
    let z = analytic_signal(&x).unwrap();
    println!("cos → analytic signal at n = 0, 5: {:.3}, {:.3}", z[0], z[5]);
    let am: Vec<f64> = (0..1000).map(|n| {
        let t = n as f64 / fs;
        (1.0 + 0.5 * (2.0 * PI * 2.0 * t).sin()) * (2.0 * PI * 100.0 * t).cos()
    }).collect();
    let attr = instantaneous_attributes(&am, fs).unwrap();
    let mid = 500;
    println!("envelope at t = 0.5 s: {:.4} (true {:.4})", attr.amplitude[mid], 1.0 + 0.5 * (2.0 * PI * 2.0 * 0.5f64).sin());
    let f = &attr.frequency[200..800];
    println!("instantaneous frequency: mean {:.3} Hz", f.iter().sum::<f64>() / f.len() as f64);
}
