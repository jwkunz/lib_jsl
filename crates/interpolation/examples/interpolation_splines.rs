//! Piecewise interpolation: linear, natural cubic spline, PCHIP, Akima and Catmull-Rom.
//!
//! Run with `cargo run -p lib_jsl_interpolation --example interpolation_splines`.

use lib_jsl_core::R1D;
use lib_jsl_interpolation::interpolation_trait::InterpolationTrait;
use lib_jsl_interpolation::one_d::akima::akima;
use lib_jsl_interpolation::one_d::cubic_basis::cubic_basis;
use lib_jsl_interpolation::one_d::linear_interpolator::LinearInterpolator;
use lib_jsl_interpolation::one_d::natural_cubic_spline_interpolator::NaturalCubicSplineInterpolator;
use lib_jsl_interpolation::one_d::pchip::pchip;

fn main() {
    // A temperature profile measured at uneven depths, rising steeply across a thermocline.
    let depth = R1D::from_vec(vec![0.0, 5.0, 10.0, 15.0, 18.0, 20.0, 22.0, 25.0, 35.0, 50.0]);
    let temperature = R1D::from_vec(vec![20.0, 20.0, 19.9, 19.8, 17.0, 12.0, 8.5, 7.0, 6.0, 5.5]);
    let mut linear = LinearInterpolator::new(depth.view(), temperature.view()).unwrap();
    let mut spline = NaturalCubicSplineInterpolator::new(depth.view(), temperature.view()).unwrap();
    let mut monotone = pchip(depth.view(), temperature.view()).unwrap();
    let mut local = akima(depth.view(), temperature.view()).unwrap();
    let mut cardinal = cubic_basis(depth.view(), temperature.view(), 0.0).unwrap();
    println!("depth   linear   spline    PCHIP    Akima  Catmull-Rom");
    for z in [2.5, 12.5, 16.5, 19.0, 21.0, 30.0, 42.0] {
        println!(
            "{z:5.1} {:8.3} {:8.3} {:8.3} {:8.3} {:8.3}",
            linear.interpolate_at(z).unwrap(),
            spline.interpolate_at(z).unwrap(),
            monotone.interpolate_at(z).unwrap(),
            local.interpolate_at(z).unwrap(),
            cardinal.interpolate_at(z).unwrap()
        );
    }
    // The spline overshoots the 20 °C surface value; PCHIP cannot leave the range of its data.
    let max = |f: &mut dyn FnMut(f64) -> f64| (0..=500).map(|i| f(i as f64 * 0.1)).fold(f64::MIN, f64::max);
    println!("maximum on [0, 50]: spline {:.3}, PCHIP {:.3}", max(&mut |z| spline.interpolate_at(z).unwrap()), max(&mut |z| monotone.interpolate_at(z).unwrap()));
}
