//! Spectral methods: Chebyshev collocation and Fourier differentiation.
//!
//! Run with `cargo run --release -p lib_jsl_integration --example pde_spectral`.

use lib_jsl_integration::pde::spectral::{chebyshev_bvp, fourier_derivative, periodic_heat};
use lib_jsl_special_functions::airy_functions::airy_ai;
use std::f64::consts::PI;

fn main() {
    // Airy's equation u'' = x u on [−8, 2] with u(−8) = Ai(−8), u(2) = Ai(2): the error against
    // the Airy function at every collocation point.
    for n in [16, 24, 32, 40, 48] {
        let (xs, u) = chebyshev_bvp(|_| 0.0, |x| -x, |_| 0.0, -8.0, 2.0, airy_ai(-8.0), airy_ai(2.0), n).unwrap();
        let error = xs.iter().zip(&u).map(|(&x, &v)| (v - airy_ai(x)).abs()).fold(0.0, f64::max);
        println!("Airy equation with {:2} Chebyshev points: max error {error:.1e}", n + 1);
    }

    // The derivative of a smooth periodic function from 32 samples.
    let n = 32;
    let x: Vec<f64> = (0..n).map(|i| 2.0 * PI * i as f64 / n as f64).collect();
    let u: Vec<f64> = x.iter().map(|x| (x.cos()).exp()).collect();
    let du = fourier_derivative(&u, 1, 2.0 * PI).unwrap();
    let error = x.iter().zip(&du).map(|(x, d)| (d + x.sin() * x.cos().exp()).abs()).fold(0.0, f64::max);
    println!("Fourier derivative of e^(cos x) with {n} points: max error {error:.1e}");

    // The periodic heat equation, solved exactly in Fourier space: a square pulse smooths out.
    let pulse: Vec<f64> = x.iter().map(|&x| if (x - PI).abs() < 1.0 { 1.0 } else { 0.0 }).collect();
    let later = periodic_heat(&pulse, 1.0, 0.5, 2.0 * PI).unwrap();
    let mean_before = pulse.iter().sum::<f64>() / n as f64;
    let mean_after = later.iter().sum::<f64>() / n as f64;
    println!("heat: peak {:.4} → {:.4}, mean {mean_before:.4} → {mean_after:.4} (conserved)", 1.0, later.iter().cloned().fold(f64::MIN, f64::max));
}
