//! Four-vectors, Lorentz boosts and the Christoffel symbols of curved space-time.

use lib_jsl_geometry::four_d::{ChristoffelSymbols4D, SpaceTimeVector4D};

fn main() {
    // A flash 5 light-seconds away, 4 seconds later: spacelike, so no signal links the two.
    let event = SpaceTimeVector4D::new(4.0, 5.0, 0.0, 0.0);
    println!("interval {}, {:?}", event.interval_squared(), event.causality(1e-12));
    for beta in [0.5, 0.8] {
        let seen = event.boosted([beta, 0.0, 0.0]).unwrap();
        println!("seen from a frame moving at {beta}c: t = {:.4}, x = {:.4}, interval {:.4}", if seen.t.abs() < 1e-12 { 0.0 } else { seen.t }, seen.x, seen.interval_squared());
    }
    println!("0.8c + 0.8c = {:.4}c, gamma(0.8c) = {:.4}", SpaceTimeVector4D::add_collinear_velocities(0.8, 0.8), SpaceTimeVector4D::gamma([0.8, 0.0, 0.0]).unwrap());

    // Around a mass M = 1, at r = 10 in the equatorial plane: a particle at rest falls inwards.
    let gamma = ChristoffelSymbols4D::schwarzschild(1.0, 10.0, std::f64::consts::FRAC_PI_2).unwrap();
    let at_rest = [1.0 / (1.0f64 - 0.2).sqrt(), 0.0, 0.0, 0.0];
    println!("radial acceleration of a particle at rest: {:.5} (Newton: -1/r² = -0.01)", gamma.geodesic_acceleration(at_rest)[1]);
    println!("torsion free: {}", gamma.is_torsion_free(1e-15));
}
