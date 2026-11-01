//! Bracketing root finders: find every root of a function in an interval.
//!
//! Run with `cargo run -p lib_jsl_root_finding --example root_finding_bracketing`.

use lib_jsl_root_finding::{bisection, bracket_root, bracket_roots_scan, brent_method, false_position, ridders_method};

fn main() {
    // The roots of cos x · cosh x + 1 = 0 in [0, 12], which give the vibration frequencies of a
    // cantilever beam: scan for sign changes, then refine each bracket.
    let f = |x: f64| x.cos() * x.cosh() + 1.0;
    for (a, b) in bracket_roots_scan(f, 0.0, 12.0, 120) {
        let root = brent_method(f, a, b, 1e-14).unwrap();
        println!("beam mode at x = {root:.12}");
    }

    // When only a starting interval is known, expand it until it brackets a root.
    let g = |x: f64| x.exp() - 50.0;
    let (lo, hi) = bracket_root(g, 0.0, 1.0).unwrap();
    println!("bracket for eˣ = 50: [{lo:.3}, {hi:.3}]");

    // All four bracketing methods find the same root.
    for (name, r) in [
        ("bisection", bisection(g, lo, hi, 1e-13)),
        ("false position", false_position(g, lo, hi, 1e-13)),
        ("Ridders", ridders_method(g, lo, hi, 1e-13)),
        ("Brent", brent_method(g, lo, hi, 1e-13)),
    ] {
        println!("{name:>15}: {:.13} (ln 50 = {:.13})", r.unwrap(), 50f64.ln());
    }
}
