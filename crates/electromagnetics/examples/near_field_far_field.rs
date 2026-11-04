//! Spherical wave expansions and near-field to far-field transformation.
//!
//! Run with `cargo run --release -p lib_jsl_electromagnetics --example near_field_far_field`.

use lib_jsl_electromagnetics::algorithms::spherical_near_to_far;
use lib_jsl_electromagnetics::antenna_models::{Antenna, Dipole, HertzianDipole};
use lib_jsl_electromagnetics::conventions::TimeConvention;
use lib_jsl_electromagnetics::coordinates::Position;
use lib_jsl_electromagnetics::fields::SphericalWaveExpansion;

fn main() {
    let f = 3e8; // λ = 1 m
    let conv = TimeConvention::PlusJ;
    // Expand a 2λ dipole's far-field pattern in spherical waves: the power sits in the low degrees.
    let dipole = Dipole::new(2.0);
    let swe = SphericalWaveExpansion::from_far_field(|t, p| dipole.far_field(t, p, f, conv), f, 15, conv).unwrap();
    let by_degree = swe.power_by_degree();
    let total: f64 = by_degree.iter().sum();
    let tail: f64 = by_degree.iter().skip(10).sum();
    println!("2λ dipole: radiated power {:.4} W (direct integration {:.4} W); share above degree 10: {:.1e}", swe.radiated_power(), dipole.radiated_power(f), tail / total);
    // Measure a Hertzian dipole's exact near field on a sphere of 0.3λ and recover its far field.
    let h = HertzianDipole::new(0.01);
    let r0 = 0.3;
    let probe = |t: f64, p: f64| {
        let (e, _) = h.near_field(&Position::spherical(r0, t, p), f, conv);
        [e[1], e[2]]
    };
    let recovered = spherical_near_to_far(probe, r0, f, 5, conv).unwrap();
    let (a, b) = (recovered.far_field(1.0, 0.0, conv)[0], h.far_field(1.0, 0.0, f, conv)[0]);
    println!("Hertzian dipole from a 0.3λ sphere: far field at θ = 1 rad {:.6} {:+.6}j V (exact {:.6} {:+.6}j V)", a.re, a.im, b.re, b.im);
}
