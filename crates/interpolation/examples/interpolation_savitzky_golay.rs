//! Savitzky-Golay smoothing and differentiation of noisy samples.
//!
//! Run with `cargo run -p lib_jsl_interpolation --example interpolation_savitzky_golay`.

use lib_jsl_core::R1D;
use lib_jsl_interpolation::one_d::savitzky_golay::{savitzky_golay, savitzky_golay_coefficients};

fn main() {
    // The classic 5-point quadratic smoothing filter: (−3, 12, 17, 12, −3) / 35.
    let c = savitzky_golay_coefficients(5, 2, 0, 1.0).unwrap();
    println!("5-point quadratic filter × 35: {:?}", c.iter().map(|v| (v * 35.0).round()).collect::<Vec<_>>());

    // A position measured every 10 ms with noise: estimate the velocity.
    let dt = 0.01;
    let position = |t: f64| 2.0 * t + 0.5 * (3.0 * t).sin();
    let mut seed = 7u64;
    let mut noise = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.002 };
    let samples = R1D::from_vec((0..300).map(|k| position(k as f64 * dt) + noise()).collect());
    let velocity = savitzky_golay(samples.view(), 31, 3, 1, dt).unwrap();
    let k = 150;
    let t = k as f64 * dt;
    let naive = (samples[k + 1] - samples[k - 1]) / (2.0 * dt);
    println!("velocity at t = {t}: true {:.4}, central difference {naive:.4}, Savitzky-Golay {:.4}", 2.0 + 1.5 * (3.0 * t).cos(), velocity[k]);
}
