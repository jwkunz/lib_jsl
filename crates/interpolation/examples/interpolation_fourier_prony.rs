//! Trigonometric interpolation of periodic samples, and Prony's method for damped oscillations.
//!
//! Run with `cargo run -p lib_jsl_interpolation --example interpolation_fourier_prony`.

use lib_jsl_core::R1D;
use lib_jsl_interpolation::one_d::fourier_1d::FourierInterpolator1D;
use lib_jsl_interpolation::one_d::pronys_method::PronyModel;
use std::f64::consts::PI;

fn main() {
    // A periodic signal sampled 24 times per period, evaluated between the samples.
    let f = |x: f64| (2.0 * PI * x).sin().exp() + 0.3 * (6.0 * PI * x).cos();
    let n = 24;
    let samples = R1D::from_vec((0..n).map(|k| f(k as f64 / n as f64)).collect());
    let interp = FourierInterpolator1D::new(samples.view(), 0.0, 1.0).unwrap();
    let error = (0..1000).map(|i| { let x = i as f64 / 1000.0; (interp.evaluate(x) - f(x)).abs() }).fold(0.0, f64::max);
    println!("trigonometric interpolation from {n} samples: max error {error:.1e}");

    // A decaying chord: 220 Hz and 330 Hz with different damping, sampled at 2 kHz for 25 ms.
    let dt = 1.0 / 2000.0;
    let chord = |t: f64| (-40.0 * t).exp() * (2.0 * PI * 220.0 * t).sin() + 0.6 * (-90.0 * t).exp() * (2.0 * PI * 330.0 * t).cos();
    let data = R1D::from_vec((0..50).map(|k| chord(k as f64 * dt)).collect());
    let model = PronyModel::new(data.view(), 0.0, dt, 4).unwrap();
    for s in model.exponents().into_iter().filter(|s| s.im > 0.0) {
        println!("component: decay rate {:6.2} /s, frequency {:7.3} Hz", -s.re, s.im / (2.0 * PI));
    }
    println!("extrapolated to 40 ms: model {:.6}, true {:.6}", model.evaluate(0.04), chord(0.04));
}
