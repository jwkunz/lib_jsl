//! Antenna models compared: directivity, radiation resistance and pattern cuts.
//!
//! Run with `cargo run --release -p lib_jsl_electromagnetics --example antenna_patterns`.

use lib_jsl_core::constants::em::C0;
use lib_jsl_electromagnetics::antenna_models::{Antenna, Dipole, Dish, HalfWaveDipole, HertzianDipole, MagneticLoop, PatchAntenna};
use lib_jsl_electromagnetics::conventions::power_to_db;
use lib_jsl_utils::terminal_plot::{PolarSeries, render_polar_plot};
use std::f64::consts::PI;

fn cut(antenna: &dyn Antenna, phi: f64, f: f64) -> (Vec<f64>, Vec<f64>) {
    // θ from 0 to 360° in the plane φ (θ > 180° continues on the far side, φ + 180°): plotted with
    // the boresight (+z) pointing up.
    let peak = antenna.peak_directivity(f);
    (0..=720)
        .map(|k| {
            let t = PI * k as f64 / 360.0;
            let (theta, ph) = if t <= PI { (t, phi) } else { (2.0 * PI - t, phi + PI) };
            let d = antenna.directivity(theta, ph, f) / peak;
            (PI / 2.0 - t, power_to_db(d.max(1e-6)))
        })
        .unzip()
}

fn main() {
    let f = 10e9;
    let lambda = C0 / f;
    let patch = PatchAntenna::design(f, 2.2, 0.1588e-2).unwrap();
    let models: Vec<(&str, Box<dyn Antenna>)> = vec![
        ("Hertzian dipole (λ/50)", Box::new(HertzianDipole::new(lambda / 50.0))),
        ("small loop (a = λ/30)", Box::new(MagneticLoop::new(lambda / 30.0, 1))),
        ("half-wave dipole", Box::new(HalfWaveDipole::new(f))),
        ("full-wave dipole", Box::new(Dipole::new(lambda))),
        ("microstrip patch (ε_r 2.2)", Box::new(patch)),
        ("dish, D = 10λ, uniform", Box::new(Dish::new(10.0 * lambda))),
        ("dish, D = 10λ, tapered (p=2)", Box::new(Dish::with_taper(10.0 * lambda, 2))),
    ];
    println!("== antenna models at 10 GHz (λ = {:.1} mm) ==", lambda * 1e3);
    println!("  {:<30} {:>10} {:>8} {:>16}", "model", "D (lin)", "D (dBi)", "R_rad (Ω)");
    for (name, a) in &models {
        let d = a.peak_directivity(f);
        let r = a.radiation_resistance(f).map_or("-".to_string(), |r| format!("{r:.4}"));
        println!("  {:<30} {:>10.2} {:>8.2} {:>16}", name, d, power_to_db(d), r);
    }
    println!(
        "\n  patch: W = {:.2} mm, L = {:.2} mm, edge resistance {:.0} Ω, 50 Ω inset {:.2} mm",
        patch.width * 1e3,
        patch.length * 1e3,
        patch.edge_resistance(f),
        patch.inset_for_resistance(f, 50.0).unwrap() * 1e3
    );
    let dish = Dish::new(10.0 * lambda);
    println!("  dish: half-power beamwidth {:.2}° (uniform circular aperture: 1.029 λ/D = {:.2}°)\n", dish.half_power_beamwidth(f).to_degrees(), (1.029 / 10.0f64).to_degrees());

    for (title, antenna, phi) in [
        ("half-wave dipole, E-plane (relative dB, floor -30)", &models[2].1, 0.0),
        ("microstrip patch, E-plane", &models[4].1, 0.0),
        ("microstrip patch, H-plane", &models[4].1, PI / 2.0),
        ("tapered 10λ dish, principal cut", &models[6].1, 0.0),
    ] {
        let (angles, db) = cut(antenna.as_ref(), phi, f);
        println!("{}\n", render_polar_plot(&[PolarSeries { label: "pattern", angles: &angles, values_db: &db }], -30.0, 10.0, 48, Some(title), true));
    }
}
