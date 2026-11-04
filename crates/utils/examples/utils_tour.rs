//! Constants, floating-point spacing, unit conversion and a terminal plot.

use lib_jsl_core::constants::{em, math, physics};
use lib_jsl_utils::machine_epsilon::{approx_eq_ulps, ulp, ulps_between};
use lib_jsl_utils::terminal_plot::{PlotOptions, PlotSeries, render_terminal_plot};
use lib_jsl_utils::unit_conversion::{unit_conversion, watts_to_dbm};

fn main() {
    // Constants are plain f64 values grouped by field.
    println!("wavelength of 2.4 GHz: {:.4} m", physics::C / 2.4e9);
    println!("free-space impedance: {:.6} ohm, Euler's gamma {}", em::Z_0, math::EULER_GAMMA);

    // Floating-point spacing: 0.1 + 0.2 is one step away from 0.3.
    println!("ulp(1) = {:e}, ulp(1e6) = {:e}", ulp(1.0), ulp(1e6));
    println!("0.1 + 0.2 is {:?} representable steps from 0.3; equal within 1 ulp: {}", ulps_between(0.1 + 0.2, 0.3), approx_eq_ulps(0.1 + 0.2, 0.3, 1));

    // Unit conversion checks dimensions and respects case where prefixes need it.
    println!("55 mph = {:.2} km/h; 20 C = {:.1} F", unit_conversion(55.0, "mph", "km/h").unwrap(), unit_conversion(20.0, "C", "F").unwrap());
    println!("25 mW = {:.2} dBm; converting kg to m is refused: {}", watts_to_dbm(unit_conversion(25.0, "mW", "W").unwrap()), unit_conversion(1.0, "kg", "m").is_err());

    // A damped oscillation drawn with Braille dots.
    let t: Vec<f64> = (0..400).map(|i| i as f64 * 0.05).collect();
    let y: Vec<f64> = t.iter().map(|&t| (-0.15 * t).exp() * (2.0 * t).cos()).collect();
    let envelope: Vec<f64> = t.iter().map(|&t| (-0.15 * t).exp()).collect();
    let options = PlotOptions { width: 56, height: 10, title: Some("damped oscillation".into()), ..Default::default() };
    println!("{}", render_terminal_plot(&[PlotSeries { label: "signal", x: &t, y: &y }, PlotSeries { label: "envelope", x: &t, y: &envelope }], &options));
}
