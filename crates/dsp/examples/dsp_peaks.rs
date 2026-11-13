//! Finding the lines of a noisy spectrum by prominence, and measuring their widths.

use lib_jsl_dsp::discrete::peaks::{Bounds, PeakOptions, find_peaks};

fn main() {
    // Three Gaussian lines on a sloping baseline, plus a little deterministic "noise".
    let x: Vec<f64> = (0..400)
        .map(|i| {
            let t = i as f64;
            let line = |c: f64, h: f64, w: f64| h * (-0.5 * ((t - c) / w).powi(2)).exp();
            line(80.0, 1.0, 4.0) + line(200.0, 0.4, 10.0) + line(300.0, 0.7, 3.0) + 0.001 * t + 0.03 * (1.7 * t).sin() * (0.31 * t).cos()
        })
        .collect();
    let every = find_peaks(&x, &PeakOptions::default()).unwrap();
    println!("local maxima: {}", every.indices.len());
    let options = PeakOptions { prominence: Some(Bounds::at_least(0.2)), width: Some(Bounds::at_least(0.0)), ..Default::default() };
    let peaks = find_peaks(&x, &options).unwrap();
    let p = peaks.prominences.as_ref().unwrap();
    let w = peaks.widths.as_ref().unwrap();
    println!("peaks with prominence at least 0.2:");
    for (k, &i) in peaks.indices.iter().enumerate() {
        println!(
            "  at {i:3}: height {:.3}, prominence {:.3}, width at half prominence {:.1} samples",
            x[i], p.prominences[k], w.widths[k]
        );
    }
}
