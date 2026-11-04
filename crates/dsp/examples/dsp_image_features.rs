//! Blob detection, SIFT and SURF keypoints, and descriptor matching.

use lib_jsl_dsp::discrete::image::feature_extraction::blob_detection::blob_detection;
use lib_jsl_dsp::discrete::image::feature_extraction::match_descriptors;
use lib_jsl_dsp::discrete::image::feature_extraction::sift::{SiftParams, sift};
use lib_jsl_dsp::discrete::image::feature_extraction::surf::{SurfParams, surf};
use ndarray::Array2;

fn main() {
    // Two Gaussian spots of standard deviation 3 and 6.
    let spots = Array2::from_shape_fn((80, 80), |(r, c)| {
        let g = |r0: f64, c0: f64, s: f64| (-((r as f64 - r0).powi(2) + (c as f64 - c0).powi(2)) / (2.0 * s * s)).exp();
        g(20.0, 20.0, 3.0) + g(50.0, 55.0, 6.0)
    });
    for b in blob_detection(&spots, 1.0, 12.0, 12, 0.1, 0.5).unwrap() {
        println!("blob at ({}, {}), σ = {:.2}, radius {:.1}", b.row, b.col, b.sigma, b.radius());
    }
    // A textured image and the same shifted by (5, 3) pixels with its contrast halved.
    let texture = |r: f64, c: f64| (r / 4.0).sin() * (c / 5.0).cos() + 0.5 * ((r + c) / 7.0).sin() + 0.3 * ((r * c).abs().sqrt() / 2.0).cos();
    let a = Array2::from_shape_fn((96, 96), |(r, c)| texture(r as f64, c as f64));
    let b = Array2::from_shape_fn((96, 96), |(r, c)| 0.5 * texture(r as f64 - 5.0, c as f64 - 3.0) + 0.2);
    let (ka, kb) = (sift(&a, &SiftParams::default()).unwrap(), sift(&b, &SiftParams::default()).unwrap());
    let da: Vec<Vec<f64>> = ka.iter().map(|k| k.descriptor.clone()).collect();
    let db: Vec<Vec<f64>> = kb.iter().map(|k| k.descriptor.clone()).collect();
    let m = match_descriptors(&da, &db, 0.8, true).unwrap();
    let shifts: Vec<(f64, f64)> = m.iter().map(|mm| (kb[mm.index_b].y - ka[mm.index_a].y, kb[mm.index_b].x - ka[mm.index_a].x)).collect();
    let good = shifts.iter().filter(|(dy, dx)| (dy - 5.0).abs() < 1.0 && (dx - 3.0).abs() < 1.0).count();
    println!("SIFT: {} and {} keypoints, {} matches, {} with the true shift (5, 3)", ka.len(), kb.len(), m.len(), good);
    let (sa, sb) = (surf(&a, &SurfParams::default()).unwrap(), surf(&b, &SurfParams::default()).unwrap());
    println!("SURF: {} and {} keypoints, descriptors of length {}", sa.len(), sb.len(), sa.first().map_or(0, |k| k.descriptor.len()));
}
