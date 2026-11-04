//! Transmission lines, two-port networks and matching.
//!
//! Run with `cargo run --release -p lib_jsl_electromagnetics --example transmission_lines`.

use lib_jsl_electromagnetics::conventions::C64;
use lib_jsl_electromagnetics::networks::s_parameters::{abcd_to_s, cascade, insertion_loss_db, is_lossless};
use lib_jsl_electromagnetics::networks::transmission_lines::{TransmissionLine, quarter_wave_transformer, reflection_coefficient, return_loss_db, single_stub_match, vswr};

fn main() {
    let f = 1e9;
    // A 50 Ω coaxial line with velocity factor 0.66, terminated in 100 Ω.
    let coax = TransmissionLine::lossless(50.0, 0.66).unwrap();
    let lambda = coax.wavelength(f);
    let load = C64::new(100.0, 0.0);
    let gamma = reflection_coefficient(load, C64::new(50.0, 0.0));
    println!("guided wavelength {:.1} mm; Γ = {:.3}, VSWR {:.2}, return loss {:.2} dB", 1e3 * lambda, gamma.re, vswr(gamma), return_loss_db(gamma));
    for frac in [0.125, 0.25, 0.5] {
        let z = coax.input_impedance(load, frac * lambda, f);
        println!("  input impedance {frac}λ from the load: {:.1} {:+.1}j Ω", z.re, z.im);
    }
    // Match 100 Ω with a quarter-wave section, and 25 + 50j Ω with a shorted stub.
    println!("quarter-wave transformer: {:.2} Ω", quarter_wave_transformer(50.0, 100.0));
    for (d, s) in single_stub_match(C64::new(25.0, 50.0), 50.0).unwrap() {
        println!("  stub {s:.4}λ long, {d:.4}λ from the load");
    }
    // A lossy line as a two-port: R = 2 Ω/m, L = 250 nH/m, G = 10 µS/m, C = 100 pF/m, 3 m long.
    let lossy = TransmissionLine::distributed(2.0, 250e-9, 1e-5, 100e-12).unwrap();
    println!("lossy line: Z₀ = {:.2} {:+.3}j Ω, α = {:.4} Np/m", lossy.characteristic_impedance(f).re, lossy.characteristic_impedance(f).im, lossy.propagation_constant(f).re);
    let s = abcd_to_s(&lossy.abcd(3.0, f), 50.0).unwrap();
    let two = cascade(&s, &s, 50.0).unwrap();
    println!("  3 m: insertion loss {:.3} dB; 6 m in cascade {:.3} dB; lossless {}", insertion_loss_db(&s), insertion_loss_db(&two), is_lossless(&s, 1e-6));
}
