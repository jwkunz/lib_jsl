//! The Hough transform for lines and circles.

use lib_jsl_dsp::discrete::image::feature_extraction::hough_transform::{hough_circles, hough_lines};
use ndarray::Array2;
use std::f64::consts::PI;

fn main() {
    let mut edges = Array2::from_elem((64, 64), false);
    // The line y = x / 2 + 10 (x the column, y the row), with a gap, and a circle of radius 9.
    for x in 0..64 {
        if !(20..30).contains(&x) {
            edges[(10 + x / 2, x)] = true;
        }
    }
    for k in 0..360 {
        let a = k as f64 * PI / 180.0;
        edges[((45.0 + 9.0 * a.sin()).round() as usize, (20.0 + 9.0 * a.cos()).round() as usize)] = true;
    }
    for line in hough_lines(&edges, 20, 1).unwrap() {
        println!("line: θ = {:.0}°, ρ = {:.1}, {} votes", line.theta.to_degrees(), line.rho, line.votes);
    }
    // The line's direction is (2, 1), so its normal is (−1, 2)/√5: θ = atan2(2, −1), ρ = 20/√5.
    println!("exact: θ = {:.1}°, ρ = {:.2}", 2f64.atan2(-1.0).to_degrees(), 20.0 / 5f64.sqrt());
    for c in hough_circles(&edges, &[7, 9, 11], 0.6, 5).unwrap().iter().take(1) {
        println!("circle: centre (row {}, column {}), radius {}, support {:.2}", c.row, c.col, c.radius, c.support);
    }
}
