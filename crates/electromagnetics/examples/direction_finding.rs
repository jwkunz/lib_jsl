//! Direction-of-arrival estimation: Bartlett, Capon and MUSIC spectra and ESPRIT on a uniform
//! linear array.
//!
//! Run with `cargo run --release -p lib_jsl_electromagnetics --example direction_finding`.

use lib_jsl_core::constants::em::C0;
use lib_jsl_electromagnetics::algorithms::{bartlett_spectrum, capon_spectrum, esprit_ula, music_spectrum, spectrum_peaks};
use lib_jsl_electromagnetics::antenna_models::HertzianDipole;
use lib_jsl_electromagnetics::arrays::SensorArray;
use lib_jsl_electromagnetics::conventions::{C64, TimeConvention};
use lib_jsl_electromagnetics::coordinates::RigidTransform;
use lib_jsl_electromagnetics::signal_models::{sample_covariance, synthesize_snapshots};
use ndarray::Array2;
use std::f64::consts::{FRAC_PI_2, TAU};
use std::sync::Arc;

fn main() {
    let f = 1e9;
    let lambda = C0 / f;
    // Ten short vertical dipoles along x, half a wavelength apart; sources in the horizontal plane.
    let positions = SensorArray::linear_positions(10, lambda / 2.0, [1.0, 0.0, 0.0]);
    let array = SensorArray::from_antenna(&positions, Arc::new(HertzianDipole::new(0.01)), &RigidTransform::identity(), f, TimeConvention::PlusJ, 1.0).unwrap();
    let pol = [C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
    // Two uncorrelated sources at azimuths 70° and 78°, each at 5 dB SNR per element.
    let mut r = Array2::<C64>::eye(10) * C64::new(10f64.powf(-0.5), 0.0);
    for az in [70.0f64, 78.0] {
        let a = array.steering_vector(FRAC_PI_2, az.to_radians(), pol).unwrap();
        let scale = a.iter().map(|v| v.norm_sqr()).sum::<f64>() / 10.0;
        for i in 0..10 {
            for j in 0..10 {
                r[[i, j]] += a[i] * a[j].conj() / scale;
            }
        }
    }
    // Estimate the covariance from 300 snapshots, as a receiver would.
    let r_hat = sample_covariance(&synthesize_snapshots(&r, 300, 1).unwrap()).unwrap();
    let grid: Vec<f64> = (0..=1200).map(|i| 30.0 + 120.0 * i as f64 / 1200.0).collect();
    let dirs: Vec<(f64, f64)> = grid.iter().map(|&a| (FRAC_PI_2, a.to_radians())).collect();
    for (name, s) in [
        ("Bartlett", bartlett_spectrum(&array, &r_hat, &dirs, pol).unwrap()),
        ("Capon", capon_spectrum(&array, &r_hat, &dirs, pol).unwrap()),
        ("MUSIC", music_spectrum(&array, &r_hat, 2, &dirs, pol).unwrap()),
    ] {
        let mut peaks: Vec<f64> = spectrum_peaks(&s, 2).iter().map(|&i| grid[i]).collect();
        peaks.sort_by(f64::total_cmp);
        println!("{name:>8}: peaks at {:.1}° and {:.1}°", peaks[0], peaks[1]);
    }
    // ESPRIT returns direction cosines u = cos φ along the array axis.
    let u = esprit_ula(&r_hat, 2, lambda / 2.0, TAU / lambda, TimeConvention::PlusJ).unwrap();
    let mut phi: Vec<f64> = u.iter().map(|v| v.acos().to_degrees()).collect();
    phi.sort_by(f64::total_cmp);
    println!("  ESPRIT: {:.2}° and {:.2}°", phi[0], phi[1]);
}
