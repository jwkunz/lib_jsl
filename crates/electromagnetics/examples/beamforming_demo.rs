//! Beamforming in a realistic scene: a weak desired signal, two strong jammers, a warm ground and
//! a cold sky, and receiver noise. CFM, max-SNR, MVDR and LCMV are compared.
//!
//! Run with `cargo run --release -p lib_jsl_electromagnetics --example beamforming_demo`.

use lib_jsl_core::constants::em::C0;
use lib_jsl_electromagnetics::algorithms::{cfm_for_direction, lcmv_beamformer, max_snr_beamformer, mvdr_beamformer, response, sinr};
use lib_jsl_electromagnetics::antenna_models::HalfWaveDipole;
use lib_jsl_electromagnetics::arrays::SensorArray;
use lib_jsl_electromagnetics::conventions::{C64, TimeConvention, amplitude_to_db, power_to_db};
use lib_jsl_electromagnetics::coordinates::RigidTransform;
use lib_jsl_electromagnetics::fields::Domain;
use lib_jsl_electromagnetics::signal_models::{BrightnessScene, Coherency, PointSource, Scene, SkyPolarization};
use lib_jsl_utils::terminal_plot::{PlotOptions, PlotSeries, render_terminal_plot};
use ndarray::{Array1, Array2};
use std::f64::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

fn main() {
    let f = 1.5e9;
    let lambda = C0 / f;
    // 16 vertical half-wave dipoles along x, half a wavelength apart.
    let array = SensorArray::from_antenna(&SensorArray::linear_positions(16, lambda / 2.0, [1.0, 0.0, 0.0]), Arc::new(HalfWaveDipole::new(f)), &RigidTransform::identity(), f, TimeConvention::PlusJ, 1.0).unwrap();
    let theta_pol = [C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
    // Directions in the xz plane, measured as θ from +z; the array axis is x.
    let look = 1.2;
    let jammers = [0.75, 2.4];
    let bw = 1e6;
    let desired = Scene::new(bw).unwrap().with_point_source(PointSource::new(look, 0.0, 3e-13, Coherency::theta()).unwrap());
    let environment = BrightnessScene::from_fn(Arc::new(Domain::far_field_gauss(40, 80).unwrap()), f, SkyPolarization::Unpolarized, |t, _| if t > FRAC_PI_2 { 290.0 } else { 10.0 }).unwrap();
    let mut interference = Scene::new(bw).unwrap().with_distributed(environment).with_receiver_temperature(75.0);
    for &j in &jammers {
        interference = interference.with_point_source(PointSource::new(j, 0.0, 3e-8, Coherency::theta()).unwrap());
    }
    let rs = desired.covariance(&array).unwrap();
    let rn = interference.covariance(&array).unwrap();
    let r = &rs + &rn;
    let a_look = array.steering_vector(look, 0.0, theta_pol).unwrap();

    let cfm = cfm_for_direction(&array, look, 0.0, theta_pol).unwrap();
    let (max_snr, _) = max_snr_beamformer(&rs, &rn).unwrap();
    let mvdr = mvdr_beamformer(&r, &a_look).unwrap();
    // LCMV: pass the look direction, and force exact nulls on both jammers.
    let dirs = [look, jammers[0], jammers[1]];
    let c = Array2::from_shape_fn((16, 3), |(i, k)| array.steering_vector(dirs[k], 0.0, theta_pol).unwrap()[i]);
    let lcmv = lcmv_beamformer(&r, &c, &Array1::from(vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)])).unwrap();

    // Per-element levels measured from the model itself.
    let noise_only = Scene::new(bw).unwrap().with_distributed(BrightnessScene::from_fn(Arc::new(Domain::far_field_gauss(40, 80).unwrap()), f, SkyPolarization::Unpolarized, |t, _| if t > FRAC_PI_2 { 290.0 } else { 10.0 }).unwrap()).with_receiver_temperature(75.0).covariance(&array).unwrap();
    let noise_power = noise_only[[0, 0]].re;
    let jammer_power = PointSource::new(jammers[0], 0.0, 3e-8, Coherency::theta()).unwrap().covariance(&array).unwrap()[[0, 0]].re;
    println!("== 16-element half-wave dipole array at 1.5 GHz ==");
    println!("  desired at θ = {:.0}°, jammers at {:.0}° and {:.0}°;", look.to_degrees(), jammers[0].to_degrees(), jammers[1].to_degrees());
    println!("  290 K ground below the horizon, 10 K sky above, 75 K receivers (system noise {:.0} K per element).", noise_power / (lib_jsl_core::constants::physics::K_B * bw));
    println!("  per element: desired SNR {:.1} dB, each jammer {:.1} dB above the noise.\n", power_to_db(rs[[0, 0]].re / noise_power), power_to_db(jammer_power / noise_power));
    println!("  {:<10} {:>12} {:>22} {:>22}", "beamformer", "SINR (dB)", "jammer 1 response (dB)", "jammer 2 response (dB)");
    let mut weights_for_plot = Vec::new();
    for (name, w) in [("CFM", &cfm), ("max-SNR", &max_snr), ("MVDR", &mvdr), ("LCMV", &lcmv)] {
        let gain = response(w, &a_look).unwrap().norm();
        let rel = |j: f64| amplitude_to_db(response(w, &array.steering_vector(j, 0.0, theta_pol).unwrap()).unwrap().norm().max(1e-12) / gain);
        println!("  {:<10} {:>12.2} {:>22.1} {:>22.1}", name, power_to_db(sinr(w, &rs, &rn).unwrap()), rel(jammers[0]), rel(jammers[1]));
        weights_for_plot.push((name, w.clone()));
    }

    // Pattern cuts in the xz plane, normalised to the look direction.
    let angles: Vec<f64> = (1..360).map(|k| PI * k as f64 / 360.0).collect();
    let curves: Vec<(String, Vec<f64>)> = weights_for_plot
        .iter()
        .filter(|(n, _)| *n == "CFM" || *n == "MVDR")
        .map(|(name, w)| {
            let g = response(w, &a_look).unwrap().norm();
            let db = angles.iter().map(|&t| amplitude_to_db((array.pattern_at(w, t, 0.0).unwrap()[0].norm() / g).max(1e-4))).collect();
            (name.to_string(), db)
        })
        .collect();
    let deg: Vec<f64> = angles.iter().map(|a| a.to_degrees()).collect();
    let series: Vec<PlotSeries> = curves.iter().map(|(n, d)| PlotSeries { label: n, x: &deg, y: d }).collect();
    let opts = PlotOptions { width: 72, height: 18, title: Some("array pattern in the xz plane (dB re look direction) vs θ (degrees)".into()), color: true, y_range: Some((-70.0, 5.0)), ..Default::default() };
    println!("\n{}", render_terminal_plot(&series, &opts));
    println!("\nThe CFM (matched filter) points a clean beam but leaves the jammers in its sidelobes; the");
    println!("adaptive solutions place deep nulls on them and recover nearly the full array gain. (A linear");
    println!("array along x only sees sin θ: a jammer at 180° - 69° = 111° would be indistinguishable.)");
}
