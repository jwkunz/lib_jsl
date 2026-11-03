//! This module provides implementations of the sinc function and its normalized version, as well as functions to generate centered sinc windows. The sinc function is defined as sin(pi * x) / (pi * x) for x != 0 and is equal to 1 at x = 0. The normalized sinc function is defined similarly but with a factor of 2 in the argument. The centered sinc window functions generate a vector of sinc values that are centered around zero, which can be used in various signal processing applications, such as interpolation and filter design.
//!
//! See the book entry [FIR design: windows, frequency sampling and Parks-McClellan](https://jwkunz.github.io/lib_jsl/book/filters/fir-design.html#the-window-method).

use std::f64::consts::PI;
pub fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-15 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

pub fn sinc_normalized(x: f64) -> f64 {
    if x.abs() < 1e-15 {
        1.0
    } else {
        (2.0 * PI * x).sin() / (2.0 * PI * x)
    }
}

/// `n` samples of `sinc(cut_off · t)` (or of the normalised sinc for
/// [`generate_centered_sinc_normalized`]) at points `t` spaced `1/n` apart and centred on zero,
/// so `t` spans about `[−1/2, 1/2]`. The result is a smooth taper whose width depends on `cut_off`
/// (for `cut_off ≤ 2` every value is positive), used as a smoothing kernel; it is not the impulse
/// response of a low-pass filter with that cutoff, for which see `firwin`.
pub fn generate_centered_sinc(n : usize, cut_off: f64) -> Vec<f64> {
    let m = n as f64;
    let mut sinc_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 - (m - 1.0) / 2.0) / m;
        sinc_vec.push(sinc(x * cut_off));
    }
    sinc_vec
}

pub fn generate_centered_sinc_normalized(n : usize, cut_off: f64) -> Vec<f64> {
    let m = n as f64;
    let mut sinc_vec = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 - (m - 1.0) / 2.0) / m;
        sinc_vec.push(sinc_normalized(x * cut_off ));
    }
    sinc_vec
}