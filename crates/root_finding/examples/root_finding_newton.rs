//! Newton's method and the secant method.
//!
//! Run with `cargo run -p lib_jsl_root_finding --example root_finding_newton`.

use lib_jsl_root_finding::{brent_method, newton_raphson, secant_method};

fn main() {
    // Kepler's equation M = E − e sin E: the eccentric anomaly E of an orbit at mean anomaly M.
    let (e, m) = (0.6, 1.2);
    let f = |x: f64| x - e * x.sin() - m;
    let df = |x: f64| 1.0 - e * x.cos();
    let newton = newton_raphson(f, df, m, 1e-15).unwrap();
    let secant = secant_method(f, m, m + 0.1, 1e-15).unwrap();
    println!("Kepler: E = {newton:.15} (Newton), {secant:.15} (secant)");

    // Newton's method needs a good start: on arctan it diverges from |x₀| > 1.39.
    let atan_derivative = |x: f64| 1.0 / (1.0 + x * x);
    println!("arctan from 1.3: {:?}", newton_raphson(f64::atan, atan_derivative, 1.3, 1e-15));
    println!("arctan from 1.5: {:?}", newton_raphson(f64::atan, atan_derivative, 1.5, 1e-15));

    // A safe combination: bracket the root, then let a bracketing method finish.
    println!("arctan by Brent on [-5, 1.5]: {:?}", brent_method(f64::atan, -5.0, 1.5, 1e-15));
}
