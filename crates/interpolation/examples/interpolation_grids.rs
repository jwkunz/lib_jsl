//! Interpolation on grids in two, three and N dimensions.
//!
//! Run with `cargo run --release -p lib_jsl_interpolation --example interpolation_grids`.

use lib_jsl_core::{R1D, R2D};
use lib_jsl_interpolation::nd::InterpolationND;
use lib_jsl_interpolation::nd::grid_data::GridData;
use lib_jsl_interpolation::three_d::Interpolation3D;
use lib_jsl_interpolation::three_d::b_spline::BSpline3D;
use lib_jsl_interpolation::two_d::Interpolation2D;
use lib_jsl_interpolation::two_d::{bicubic::Bicubic, bilinear::Bilinear, lanczos::Lanczos2D};
use ndarray::{Array3, ArrayD, Dimension, IxDyn};

fn main() {
    // A height map sampled every 10 m over 200 m × 150 m.
    let x = R1D::from_vec((0..=20).map(|i| 10.0 * i as f64).collect());
    let y = R1D::from_vec((0..=15).map(|j| 10.0 * j as f64).collect());
    let terrain = |x: f64, y: f64| 50.0 + 20.0 * (x / 60.0).sin() * (y / 45.0).cos() + 0.05 * x;
    let z = R2D::from_shape_fn((21, 16), |(i, j)| terrain(x[i], y[j]));
    let (bl, bc, lz) = (Bilinear::new(x.view(), y.view(), z.view()).unwrap(), Bicubic::new(x.view(), y.view(), z.view()).unwrap(), Lanczos2D::new(x.view(), y.view(), z.view(), 3).unwrap());
    let (px, py) = (123.4, 67.8);
    println!("height at ({px}, {py}): exact {:.4}", terrain(px, py));
    println!("  bilinear {:.4}, bicubic {:.4}, Lanczos-3 {:.4}", bl.interpolate_at(px, py).unwrap(), bc.interpolate_at(px, py).unwrap(), lz.interpolate_at(px, py).unwrap());

    // A temperature field on a 3D grid, with a smooth tricubic B-spline.
    let a = R1D::from_vec((0..11).map(|i| i as f64 / 10.0).collect());
    let field = |x: f64, y: f64, z: f64| (x * y).exp() + z.sin();
    let v = Array3::from_shape_fn((11, 11, 11), |(i, j, k)| field(a[i], a[j], a[k]));
    let spline = BSpline3D::new(a.view(), a.view(), a.view(), &v).unwrap();
    println!("3D B-spline at (0.33, 0.47, 0.71): {:.6} (exact {:.6})", spline.interpolate_at(0.33, 0.47, 0.71).unwrap(), field(0.33, 0.47, 0.71));

    // Multilinear interpolation in 4 dimensions (5 points per axis).
    let axis: Vec<f64> = (0..5).map(|i| i as f64 * 0.25).collect();
    let axes = [axis.as_slice(); 4];
    let values = ArrayD::from_shape_fn(IxDyn(&[5, 5, 5, 5]), |i| i.as_array_view().iter().map(|&k| k as f64 * 0.25).sum::<f64>());
    let g = GridData::new(&axes, &values).unwrap();
    println!("4D multilinear at (0.1, 0.2, 0.3, 0.4): {:.6} (exact 1.0 for this linear function)", g.interpolate_at(&[0.1, 0.2, 0.3, 0.4]).unwrap());
}
