//! Elliptic integrals and Jacobian elliptic functions: the period and motion of a pendulum.
//!
//! Run with `cargo run -p lib_jsl_special_functions --example special_elliptic`.

use lib_jsl_special_functions::{carlson_rf, elliptic_e, elliptic_integral, elliptic_integral_second, jacobian_elliptic};

fn main() {
    // A pendulum of length L released from angle θ₀ has period 4 √(L/g) K(sin(θ₀/2)).
    let (l, g) = (1.0f64, 9.81f64);
    for degrees in [5.0f64, 45.0, 90.0, 170.0] {
        let k = (degrees.to_radians() / 2.0).sin();
        let period = 4.0 * (l / g).sqrt() * elliptic_integral(k);
        println!("θ₀ = {degrees:>5}°: period {period:.6} s (small-angle value {:.6} s)", 2.0 * std::f64::consts::PI * (l / g).sqrt());
    }
    // Its angle at time t: sin(θ/2) = k sn(K − t √(g/L) | k²).
    let k = (90f64.to_radians() / 2.0).sin();
    let quarter = elliptic_integral(k);
    for t in [0.0, 0.25, 0.5] {
        let (sn, _, _) = jacobian_elliptic(quarter - t * (g / l).sqrt(), k * k);
        println!("t = {t}: θ = {:.4}°", (2.0 * (k * sn).asin()).to_degrees());
    }
    // The perimeter of an ellipse with semi-axes a ≥ b is 4a E(e), e the eccentricity.
    let (a, b) = (2.0f64, 1.0f64);
    let e = (1.0 - (b / a).powi(2)).sqrt();
    println!("perimeter of the 2 × 1 ellipse: {:.12}", 4.0 * a * elliptic_integral_second(e));
    println!("E(π/3, 0.5) = {:.15}, R_F(1, 2, 0) = {:.15}", elliptic_e(std::f64::consts::FRAC_PI_3, 0.5), carlson_rf(1.0, 2.0, 0.0));
}
