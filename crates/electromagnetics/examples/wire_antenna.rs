//! A thin-wire dipole solved by the method of moments (Hallén's equation).
//!
//! Run with `cargo run --release -p lib_jsl_electromagnetics --example wire_antenna`.

use lib_jsl_electromagnetics::algorithms::WireDipole;

fn main() {
    let f = 300e6; // λ = 1 m
    let radius = 1e-4;
    for length in [0.47, 0.5, 1.0] {
        let s = WireDipole::new(length, radius).unwrap().solve(f, 81).unwrap();
        let mid = s.currents.len() / 2;
        println!("L = {length}λ: Z_in = {:.2} {:+.2}j Ω, |I(feed)| = {:.2} mA, radiated {:.3} mW for 1 V",
            s.input_impedance.re, s.input_impedance.im, 1e3 * s.currents[mid].norm(), 1e3 * s.radiated_power());
    }
}
