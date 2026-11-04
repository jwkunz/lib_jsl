//! Closest-point queries: the point of a primitive nearest to a query point.
//!
//! Points and vectors are plain `[f64; D]` arrays. Every query returns the closest point together
//! with whatever parameterisation is natural for the primitive (a segment parameter, barycentric
//! coordinates, ...), and distances are Euclidean. For nearest neighbours *within a large point
//! set*, build a [`KdTree`](super::kd_trees::KdTree) instead.
//!
//! See the book entry [Intersections and closest points](https://jwkunz.github.io/lib_jsl/book/geometry/intersections.html#closest-points).

fn sub<const D: usize>(a: [f64; D], b: [f64; D]) -> [f64; D] {
    std::array::from_fn(|k| a[k] - b[k])
}
fn add_scaled<const D: usize>(a: [f64; D], b: [f64; D], s: f64) -> [f64; D] {
    std::array::from_fn(|k| a[k] + s * b[k])
}
fn dot<const D: usize>(a: [f64; D], b: [f64; D]) -> f64 {
    (0..D).map(|k| a[k] * b[k]).sum()
}
fn distance<const D: usize>(a: [f64; D], b: [f64; D]) -> f64 {
    let d = sub(a, b);
    dot(d, d).sqrt()
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// A closest point and its distance from the query.
#[derive(Debug, Clone, Copy, PartialEq)]

pub struct Closest<const D: usize> {
    pub point: [f64; D],
    pub distance: f64,
}

impl<const D: usize> Closest<D> {
    fn new(query: [f64; D], point: [f64; D]) -> Self {
        Closest { point, distance: distance(query, point) }
    }
}

/// The closest point on segment `a`-`b`, and its parameter `t` in `[0, 1]` (`a + t (b - a)`).
/// A degenerate segment (`a == b`) returns `a` with `t = 0`.
pub fn closest_point_on_segment<const D: usize>(p: [f64; D], a: [f64; D], b: [f64; D]) -> (Closest<D>, f64) {
    let ab = sub(b, a);
    let len2 = dot(ab, ab);
    let t = if len2 == 0.0 { 0.0 } else { (dot(sub(p, a), ab) / len2).clamp(0.0, 1.0) };
    (Closest::new(p, add_scaled(a, ab, t)), t)
}

/// The closest point on the infinite line through `origin` with direction `dir`, and its
/// (unclamped) parameter. A zero direction returns `origin`.
pub fn closest_point_on_line<const D: usize>(p: [f64; D], origin: [f64; D], dir: [f64; D]) -> (Closest<D>, f64) {
    let len2 = dot(dir, dir);
    let t = if len2 == 0.0 { 0.0 } else { dot(sub(p, origin), dir) / len2 };
    (Closest::new(p, add_scaled(origin, dir, t)), t)
}

/// The closest point on the ray from `origin` along `dir` (parameter `t >= 0`).
pub fn closest_point_on_ray<const D: usize>(p: [f64; D], origin: [f64; D], dir: [f64; D]) -> (Closest<D>, f64) {
    let (c, t) = closest_point_on_line(p, origin, dir);
    if t >= 0.0 { (c, t) } else { (Closest::new(p, origin), 0.0) }
}

/// The closest point on a plane given by a point on it and a normal (need not be unit); also the
/// signed distance (positive on the side the normal points to). `None` for a zero normal.
pub fn closest_point_on_plane(p: [f64; 3], plane_point: [f64; 3], normal: [f64; 3]) -> Option<(Closest<3>, f64)> {
    let n2 = dot(normal, normal);
    if n2 == 0.0 {
        return None;
    }
    let signed = dot(sub(p, plane_point), normal) / n2.sqrt();
    let point = add_scaled(p, normal, -signed / n2.sqrt());
    Some((Closest { point, distance: signed.abs() }, signed))
}

/// The closest point on the (filled) triangle `abc` in 3D, with its barycentric coordinates
/// `[wa, wb, wc]` (non-negative, summing to 1). Works for degenerate triangles too.
// ANCHOR: closest_triangle
pub fn closest_point_on_triangle(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> (Closest<3>, [f64; 3]) {
    // Ericson, Real-Time Collision Detection, 5.1.5: classify p by Voronoi region.
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (Closest::new(p, a), [1.0, 0.0, 0.0]);
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (Closest::new(p, b), [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (Closest::new(p, add_scaled(a, ab, v)), [1.0 - v, v, 0.0]);
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (Closest::new(p, c), [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (Closest::new(p, add_scaled(a, ac, w)), [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (Closest::new(p, add_scaled(b, sub(c, b), w)), [0.0, 1.0 - w, w]);
    }
    let denom = va + vb + vc;
    if denom == 0.0 {
        // Degenerate (collinear) triangle: fall back to the nearest of its three edges.
        let best = [(a, b), (b, c), (c, a)]
            .into_iter()
            .map(|(u, v)| closest_point_on_segment(p, u, v).0)
            .min_by(|x, y| x.distance.total_cmp(&y.distance))
            .expect("three edges");
        return (best, [f64::NAN; 3]);
    }
    let (v, w) = (vb / denom, vc / denom);
    (Closest::new(p, add_scaled(add_scaled(a, ab, v), ac, w)), [1.0 - v - w, v, w])
}
// ANCHOR_END: closest_triangle

/// The closest point in the axis-aligned box `[min, max]` (the point itself if inside).
pub fn closest_point_on_aabb<const D: usize>(p: [f64; D], min: [f64; D], max: [f64; D]) -> Closest<D> {
    Closest::new(p, std::array::from_fn(|k| p[k].clamp(min[k], max[k])))
}

/// The closest point on the surface of a sphere (the centre direction is arbitrary if `p` is at
/// the centre: the +x pole is returned).
pub fn closest_point_on_sphere<const D: usize>(p: [f64; D], centre: [f64; D], radius: f64) -> Closest<D> {
    let d = sub(p, centre);
    let len = dot(d, d).sqrt();
    let dir: [f64; D] = if len == 0.0 { std::array::from_fn(|k| if k == 0 { 1.0 } else { 0.0 }) } else { std::array::from_fn(|k| d[k] / len) };
    Closest::new(p, add_scaled(centre, dir, radius))
}

/// The closest point on a polyline's segments, with the index of the segment it lies on and the
/// parameter along it. `None` if there are fewer than two vertices.
pub fn closest_point_on_polyline<const D: usize>(p: [f64; D], vertices: &[[f64; D]]) -> Option<(Closest<D>, usize, f64)> {
    vertices
        .windows(2)
        .enumerate()
        .map(|(i, w)| {
            let (c, t) = closest_point_on_segment(p, w[0], w[1]);
            (c, i, t)
        })
        .min_by(|x, y| x.0.distance.total_cmp(&y.0.distance))
}

/// The closest point on the boundary of a closed polygon (its last vertex joins the first).
pub fn closest_point_on_polygon_boundary<const D: usize>(p: [f64; D], vertices: &[[f64; D]]) -> Option<(Closest<D>, usize, f64)> {
    if vertices.len() < 2 {
        return None;
    }
    let mut closed = vertices.to_vec();
    closed.push(vertices[0]);
    closest_point_on_polyline(p, &closed)
}

/// The closest point on a triangle mesh (brute force over triangles) with the triangle index.
/// Triangles index into `vertices`; `None` for an empty mesh or out-of-range indices.
pub fn closest_point_on_mesh(p: [f64; 3], vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> Option<(Closest<3>, usize)> {
    triangles
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let [a, b, c] = [*vertices.get(t[0])?, *vertices.get(t[1])?, *vertices.get(t[2])?];
            Some((closest_point_on_triangle(p, a, b, c).0, i))
        })
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .min_by(|x, y| x.0.distance.total_cmp(&y.0.distance))
}

/// The closest points between segments `p1`-`q1` and `p2`-`q2`: `(point on first, point on
/// second, distance)`; the segments may be degenerate (points). Ericson 5.1.9.
pub fn closest_points_between_segments<const D: usize>(p1: [f64; D], q1: [f64; D], p2: [f64; D], q2: [f64; D]) -> ([f64; D], [f64; D], f64) {
    let (d1, d2, r) = (sub(q1, p1), sub(q2, p2), sub(p1, p2));
    let (a, e, f) = (dot(d1, d1), dot(d2, d2), dot(d2, r));
    let (s, t);
    if a <= 1e-300 && e <= 1e-300 {
        (s, t) = (0.0, 0.0);
    } else if a <= 1e-300 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if e <= 1e-300 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            let mut s_ = if denom > 1e-14 * a * e { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
            let mut t_ = (b * s_ + f) / e;
            if t_ < 0.0 {
                t_ = 0.0;
                s_ = (-c / a).clamp(0.0, 1.0);
            } else if t_ > 1.0 {
                t_ = 1.0;
                s_ = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t) = (s_, t_);
        }
    }
    let (c1, c2) = (add_scaled(p1, d1, s), add_scaled(p2, d2, t));
    (c1, c2, distance(c1, c2))
}

/// The distance from `p` to the infinite 3D line through `a` and `b` (the point distance to `a`
/// if they coincide).
pub fn distance_point_to_line_3d(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let len = dot(ab, ab).sqrt();
    if len == 0.0 {
        return distance(p, a);
    }
    let c = cross3(sub(p, a), ab);
    dot(c, c).sqrt() / len
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close<const D: usize>(a: [f64; D], b: [f64; D]) -> bool {
        (0..D).all(|k| (a[k] - b[k]).abs() < 1e-12)
    }

    #[test]
    fn segment_line_ray() {
        let (c, t) = closest_point_on_segment([1.0, 2.0], [0.0, 0.0], [4.0, 0.0]);
        assert!(close(c.point, [1.0, 0.0]) && (c.distance - 2.0).abs() < 1e-12 && (t - 0.25).abs() < 1e-12);
        let (c, t) = closest_point_on_segment([9.0, 3.0], [0.0, 0.0], [4.0, 0.0]);
        assert!(close(c.point, [4.0, 0.0]) && t == 1.0);
        assert_eq!(closest_point_on_segment([1.0, 1.0], [2.0, 2.0], [2.0, 2.0]).1, 0.0);
        let (c, t) = closest_point_on_line([9.0, 3.0], [0.0, 0.0], [1.0, 0.0]);
        assert!(close(c.point, [9.0, 0.0]) && t == 9.0);
        let (c, t) = closest_point_on_ray([-5.0, 1.0], [0.0, 0.0], [1.0, 0.0]);
        assert!(close(c.point, [0.0, 0.0]) && t == 0.0);
        assert!((distance_point_to_line_3d([0.0, 3.0, 4.0], [0.0; 3], [1.0, 0.0, 0.0]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn plane_box_sphere() {
        let (c, signed) = closest_point_on_plane([1.0, 2.0, 5.0], [0.0; 3], [0.0, 0.0, 2.0]).unwrap();
        assert!(close(c.point, [1.0, 2.0, 0.0]) && signed == 5.0 && c.distance == 5.0);
        assert_eq!(closest_point_on_plane([0.0; 3], [0.0; 3], [0.0; 3]), None);
        assert!(close(closest_point_on_aabb([5.0, -1.0], [0.0, 0.0], [2.0, 2.0]).point, [2.0, 0.0]));
        let inside = closest_point_on_aabb([1.0, 1.0], [0.0, 0.0], [2.0, 2.0]);
        assert!(inside.distance == 0.0);
        let s = closest_point_on_sphere([3.0, 0.0, 4.0], [0.0; 3], 1.0);
        assert!(close(s.point, [0.6, 0.0, 0.8]) && (s.distance - 4.0).abs() < 1e-12);
        assert!(close(closest_point_on_sphere([0.0; 3], [0.0; 3], 2.0).point, [2.0, 0.0, 0.0]));
    }

    #[test]
    fn triangle_regions() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]);
        // Face, vertices, edges (including the hypotenuse), all with consistent barycentrics.
        for (p, want) in [
            ([1.0, 1.0, 3.0], [1.0, 1.0, 0.0]),
            ([-1.0, -1.0, 2.0], a),
            ([6.0, -1.0, 0.0], b),
            ([-1.0, 6.0, 0.0], c),
            ([2.0, -3.0, 1.0], [2.0, 0.0, 0.0]),
            ([-2.0, 2.0, 0.0], [0.0, 2.0, 0.0]),
            ([4.0, 4.0, 0.0], [2.0, 2.0, 0.0]),
        ] {
            let (cp, bary) = closest_point_on_triangle(p, a, b, c);
            assert!(close(cp.point, want), "{p:?} -> {:?}", cp.point);
            let rebuilt = [0, 1, 2].map(|k| bary[0] * a[k] + bary[1] * b[k] + bary[2] * c[k]);
            assert!(close(rebuilt, want) && (bary.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        // Degenerate triangle behaves like its edges.
        let (cp, _) = closest_point_on_triangle([1.0, 1.0, 0.0], [0.0; 3], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]);
        assert!(close(cp.point, [1.0, 0.0, 0.0]));
    }

    #[test]
    fn polylines_meshes_and_segment_pairs() {
        let square = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let (c, seg, _) = closest_point_on_polygon_boundary([1.0, 1.9], &square).unwrap();
        assert!(close(c.point, [1.0, 2.0]) && seg == 2);
        assert!(closest_point_on_polyline([0.0, 0.0], &[[1.0, 1.0]]).is_none());
        let verts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]];
        let (c, tri) = closest_point_on_mesh([0.2, 0.2, 4.0], &verts, &[[0, 1, 2], [0, 1, 3]]).unwrap();
        assert_eq!(tri, 1);
        assert!(c.distance < 0.21);
        assert!(closest_point_on_mesh([0.0; 3], &verts, &[[0, 1, 9]]).is_none());
        // Skew segments: x axis vs a segment above the y axis.
        let (c1, c2, d) = closest_points_between_segments([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 2.0], [0.0, 1.0, 2.0]);
        assert!(close(c1, [0.0, 0.0, 0.0]) && close(c2, [0.0, 0.0, 2.0]) && (d - 2.0).abs() < 1e-12);
        // Parallel and end-point-limited cases.
        let (_, _, d) = closest_points_between_segments([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]);
        assert!((d - 1.0).abs() < 1e-12);
        let (_, _, d) = closest_points_between_segments([0.0, 0.0], [1.0, 0.0], [3.0, 4.0], [5.0, 4.0]);
        assert!((d - (2.0f64.powi(2) + 16.0).sqrt()).abs() < 1e-12);
        let (_, _, d) = closest_points_between_segments([1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0]);
        assert!((d - 1.0).abs() < 1e-12);
    }
}
