//! Convex hulls in two and three dimensions, and boolean operations on polygons.

use lib_jsl_geometry::algorithms::{BooleanOperation, convex_hull_2d, convex_hull_3d, polygon_area_2d, polygon_boolean, polygon_signed_area};

fn main() {
    let points = [[0.0, 0.0], [2.0, 0.5], [4.0, 0.0], [3.0, 2.0], [4.0, 4.0], [2.0, 3.0], [0.0, 4.0], [1.0, 2.0], [2.0, 2.0]];
    let hull = convex_hull_2d(&points);
    println!("2D hull {:?}, area {}", hull, polygon_area_2d(&points, &hull));

    // The unit cube's corners plus its centre and a point in a face.
    let mut cube: Vec<[f64; 3]> = (0..8).map(|i| [(i & 1) as f64, (i >> 1 & 1) as f64, (i >> 2 & 1) as f64]).collect();
    cube.extend([[0.5, 0.5, 0.5], [0.5, 0.5, 1.0]]);
    let h = convex_hull_3d(&cube).unwrap();
    println!("3D hull: {} faces, vertices {:?}, volume {}, area {}", h.faces.len(), h.vertices(), h.volume(&cube), h.surface_area(&cube));
    println!("contains (0.2, 0.9, 0.5): {}", h.contains(&cube, [0.2, 0.9, 0.5], 1e-12));

    let square = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
    let diamond = [[1.0, -0.5], [2.5, 1.0], [1.0, 2.5], [-0.5, 1.0]];
    for op in [BooleanOperation::Intersection, BooleanOperation::Union, BooleanOperation::Difference] {
        let pieces = polygon_boolean(&square, &diamond, op);
        let area: f64 = pieces.iter().map(|p| polygon_signed_area(p)).sum();
        println!("{op:?}: {} piece(s), area {area:.4}", pieces.len());
    }
}
