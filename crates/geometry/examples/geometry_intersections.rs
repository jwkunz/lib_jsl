//! Intersection tests and closest points.

use lib_jsl_geometry::algorithms::*;

fn main() {
    println!("segments: {:?}", segment_segment_2d([0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]));
    println!("collinear overlap: {:?}", segment_segment_2d([0.0, 0.0], [3.0, 0.0], [1.0, 0.0], [5.0, 0.0]));
    println!("circles: {:?}", circle_circle([0.0, 0.0], 1.0, [1.0, 0.0], 1.0));

    let (a, b, c) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let hit = ray_triangle([0.2, 0.3, 1.0], [0.0, 0.0, -1.0], a, b, c).unwrap();
    println!("ray hits the triangle at t = {}, barycentric {:?}", hit.t, hit.barycentric);
    println!("ray-sphere (entry, exit): {:?}", ray_sphere([-3.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0));
    println!("ray-box (entry, exit): {:?}", ray_aabb([-3.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]));
    println!("planes z = 0 and x = 1 meet along {:?}", plane_plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]));

    let (closest, bary) = closest_point_on_triangle([2.0, 2.0, 1.0], a, b, c);
    println!("closest point of the triangle to (2, 2, 1): {:?} at distance {:.4}, barycentric {:?}", closest.point, closest.distance, bary);
    let (p, q, d) = closest_points_between_segments([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 1.0], [0.5, -1.0, 1.0]);
    println!("skew segments come closest at {p:?} and {q:?}, distance {d}");
}
