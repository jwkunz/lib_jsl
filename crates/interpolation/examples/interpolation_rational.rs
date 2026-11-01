//! Rational interpolation and approximation: Floater-Hormann, Bulirsch-Stoer, AAA and Padé.
//!
//! Run with `cargo run -p lib_jsl_interpolation --example interpolation_rational`.

use lib_jsl_core::R1D;
use lib_jsl_interpolation::interpolation_trait::InterpolationTrait;
use lib_jsl_interpolation::one_d::aaa::AaaApproximation;
use lib_jsl_interpolation::one_d::bulirsch_stoer::BulirschStoer;
use lib_jsl_interpolation::one_d::floater_hormann::FloaterHormann;
use lib_jsl_interpolation::one_d::pade_approximants::PadeApproximant;

fn main() {
    // tan x on [0, 1.5] has a pole just beyond the interval, at π/2 ≈ 1.5708.
    let n = 31;
    let xs: Vec<f64> = (0..n).map(|i| 1.5 * i as f64 / (n - 1) as f64).collect();
    let (x, y) = (R1D::from_vec(xs.clone()), R1D::from_vec(xs.iter().map(|v| v.tan()).collect()));
    let max_err = |f: &mut dyn FnMut(f64) -> f64| (0..=1500).map(|i| { let t = i as f64 / 1000.0; (f(t) - t.tan()).abs() / t.tan().abs().max(1.0) }).fold(0.0, f64::max);
    let mut fh = FloaterHormann::new(x.view(), y.view(), 4).unwrap();
    let mut bs = BulirschStoer::new(x.view(), y.view()).unwrap();
    let aaa = AaaApproximation::new(x.view(), y.view(), 1e-13, 30).unwrap();
    println!("tan x from {n} equally spaced samples on [0, 1.5], maximum relative error:");
    println!("  Floater-Hormann (d = 4): {:.1e}", max_err(&mut |t| fh.interpolate_at(t).unwrap()));
    println!("  Bulirsch-Stoer:          {:.1e}", max_err(&mut |t| bs.interpolate_at(t).unwrap()));
    println!("  AAA ({} terms):          {:.1e}", aaa.terms(), max_err(&mut |t| aaa.evaluate(t)));

    // A Bulirsch-Stoer value with its error estimate.
    let (v, e) = bs.interpolate_with_error(1.2).unwrap();
    println!("Bulirsch-Stoer at 1.2: {v:.14} ± {e:.1e} (tan 1.2 = {:.14})", 1.2f64.tan());

    // Padé: e^x from its Taylor series, [4/4] against the degree-8 Taylor polynomial at x = 3.
    let taylor: Vec<f64> = (0..=8).scan(1.0, |f, k| { let c = 1.0 / *f; *f *= (k + 1) as f64; Some(c) }).collect();
    let pade = PadeApproximant::new(&taylor, 4, 4).unwrap();
    let polynomial: f64 = taylor.iter().enumerate().map(|(k, c)| c * 3f64.powi(k as i32)).sum();
    println!("e³ = {:.6}: Taylor {:.6}, Padé [4/4] {:.6}", 3f64.exp(), polynomial, pade.evaluate(3.0));
}
