//! Delaunay triangulation and the Voronoi diagram of a few points.

use lib_jsl_geometry::algorithms::{delaunay_triangulation, polygon_signed_area, voronoi_diagram};

fn main() {
    let points = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0], [1.5, 1.2], [2.8, 1.9], [2.0, 2.6]];
    let t = delaunay_triangulation(&points).unwrap();
    println!("{} triangles: {:?}", t.triangles.len(), t.triangles);
    println!("area {} (the convex hull's), Delaunay: {}", t.area(&points), t.is_delaunay(&points, 1e-12));
    println!("{} edges; the point (3, 0.5) is in triangle {:?}", t.edges().len(), t.locate(&points, [3.0, 0.5]));

    let v = voronoi_diagram(&points, [-1.0, -1.0], [5.0, 4.0]).unwrap();
    for (i, cell) in v.cells.iter().enumerate() {
        println!("cell of {:?}: {} corners, area {:.3}", points[i], cell.len(), polygon_signed_area(cell));
    }
    println!("total {:.3} = the 6 × 5 box", v.cells.iter().map(|c| polygon_signed_area(c)).sum::<f64>());
}
