//! Smoothing, decimating and slicing a triangle mesh.

use lib_jsl_geometry::algorithms::{decimate, laplacian_smoothing, slice_mesh, taubin_smoothing};
use std::f64::consts::PI;

fn main() {
    // A torus of radii 1 and 0.3, as a 48 × 24 grid of quads, with a little bumpiness.
    let (m, k) = (48, 24);
    let mut vertices = Vec::new();
    for i in 0..m {
        for j in 0..k {
            let (u, w) = (2.0 * PI * i as f64 / m as f64, 2.0 * PI * j as f64 / k as f64);
            let r = 0.3 + 0.01 * ((7 * i + 3 * j) % 5) as f64;
            vertices.push([(1.0 + r * w.cos()) * u.cos(), (1.0 + r * w.cos()) * u.sin(), r * w.sin()]);
        }
    }
    let id = |i: usize, j: usize| (i % m) * k + (j % k);
    let triangles: Vec<[usize; 3]> = (0..m).flat_map(|i| (0..k).flat_map(move |j| [[id(i, j), id(i + 1, j), id(i + 1, j + 1)], [id(i, j), id(i + 1, j + 1), id(i, j + 1)]])).collect();
    let tube = |v: &[[f64; 3]]| v.iter().map(|p| ((p[0] * p[0] + p[1] * p[1]).sqrt() - 1.0).hypot(p[2])).sum::<f64>() / v.len() as f64;
    println!("{} triangles, mean tube radius {:.4}", triangles.len(), tube(&vertices));
    println!("Laplacian, 10 steps: mean tube radius {:.4}", tube(&laplacian_smoothing(&vertices, &triangles, 10, 0.5, true)));
    println!("Taubin, 10 steps:    mean tube radius {:.4}", tube(&taubin_smoothing(&vertices, &triangles, 10, 0.5, -0.53, true)));

    let (v, t) = decimate(&vertices, &triangles, 400);
    println!("decimated to {} triangles on {} vertices, mean tube radius {:.4}", t.len(), v.len(), tube(&v));

    for contour in slice_mesh(&vertices, &triangles, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1e-12) {
        println!("slice at z = 0: {} points, closed {}, length {:.4}", contour.points.len(), contour.closed, contour.length());
    }
}
