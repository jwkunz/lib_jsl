/// This module provides implementations of the sinc function and its normalized version, as well as functions to generate centered sinc windows. The sinc function is defined as sin(pi * x) / (pi * x) for x != 0 and is equal to 1 at x = 0. The normalized sinc function is defined similarly but with a factor of 2 in the argument. The centered sinc window functions generate a vector of sinc values that are centered around zero, which can be used in various signal processing applications, such as interpolation and filter design.
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

/// The `generate_centered_sinc` and `generate_centered_sinc_normalized` functions create vectors of sinc values that are centered around zero. 
/// The input parameter `n` specifies the number of points in the output vector, and the sinc values are computed based on their respective definitions. 
/// These functions can be used to create sinc-based filters or interpolation kernels for various signal processing tasks.
/// The `cut_off` parameter allows you to control the frequency response of the generated sinc function, which can be useful for designing low-pass filters or other types of filters based on the sinc function. 
/// It has a valid range of (0, 1], where 1 is the Nyquist frequency. Values closer to 1 will result in a wider main lobe and less attenuation in the stopband, while values closer to 0 will result in a narrower main lobe and more attenuation in the stopband.
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