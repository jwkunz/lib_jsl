//! Interpolating scattered data: radial basis functions and Shepard's method.
//!
//! Run with `cargo run -p lib_jsl_interpolation --example interpolation_scattered`.

use lib_jsl_core::R1D;
use lib_jsl_interpolation::three_d::Interpolation3D;
use lib_jsl_interpolation::three_d::shepards::Shepards;
use lib_jsl_interpolation::two_d::Interpolation2D;
use lib_jsl_interpolation::two_d::radial_basis_functions::{RadialBasisInterpolator, RadialKernel};

fn main() {
    // Rainfall (mm) measured at eight weather stations (positions in km).
    let east = R1D::from_vec(vec![2.0, 15.0, 8.0, 22.0, 30.0, 5.0, 18.0, 27.0]);
    let north = R1D::from_vec(vec![3.0, 4.0, 12.0, 10.0, 6.0, 25.0, 20.0, 28.0]);
    let rain = R1D::from_vec(vec![31.0, 27.0, 40.0, 33.0, 22.0, 48.0, 45.0, 38.0]);
    let tps = RadialBasisInterpolator::new(east.view(), north.view(), rain.view(), RadialKernel::ThinPlateSpline).unwrap();
    let gauss = RadialBasisInterpolator::new(east.view(), north.view(), rain.view(), RadialKernel::Gaussian(0.1)).unwrap();
    // Shepard's method in 3D works on 2D data with a constant third coordinate.
    let flat = R1D::zeros(8);
    let shepard = Shepards::new(east.view(), north.view(), flat.view(), rain.view(), 2.0).unwrap();
    println!("estimated rainfall (mm):   thin-plate   Gaussian   Shepard");
    for (e, n) in [(10.0, 10.0), (20.0, 15.0), (12.0, 22.0), (35.0, 30.0)] {
        println!(
            "  at ({e:4.1}, {n:4.1}) km:  {:10.2} {:10.2} {:9.2}",
            tps.interpolate_at(e, n).unwrap(),
            gauss.interpolate_at(e, n).unwrap(),
            shepard.interpolate_at(e, n, 0.0).unwrap()
        );
    }
    // Every method reproduces the measurements exactly.
    println!("at station 3: thin-plate {:.6}, Shepard {:.6} (measured 40)", tps.interpolate_at(8.0, 12.0).unwrap(), shepard.interpolate_at(8.0, 12.0, 0.0).unwrap());
}
