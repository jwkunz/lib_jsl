//! Intersection tests and constructions between common primitives.
//!
//! Points and vectors are `[f64; 2]` / `[f64; 3]` arrays. Parallel and touching configurations
//! are handled explicitly; tolerances are absolute and stated per function where they matter.

fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn at3(o: [f64; 3], d: [f64; 3], t: f64) -> [f64; 3] {
    [o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]]
}

/// Result of intersecting two segments or lines in the plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Intersection2D {
    /// No common point.
    None,
    /// A single common point.
    Point([f64; 2]),
    /// Collinear segments sharing a sub-segment from the first to the second point.
    Overlap([f64; 2], [f64; 2]),
}

/// Intersection of segments `a`-`b` and `c`-`d`, including touching ends and collinear overlap.
pub fn segment_segment_2d(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> Intersection2D {
    let (r, s) = (sub2(b, a), sub2(d, c));
    let denom = cross2(r, s);
    let ca = sub2(c, a);
    let scale = (dot2(r, r) * dot2(s, s)).sqrt();
    if denom.abs() <= 1e-14 * scale {
        // Parallel: intersect only if collinear.
        if cross2(ca, r).abs() > 1e-12 * dot2(r, r).sqrt().max(dot2(s, s).sqrt()).max(f64::MIN_POSITIVE) * ca[0].abs().max(ca[1].abs()).max(1.0) {
            return Intersection2D::None;
        }
        let rr = dot2(r, r);
        if rr == 0.0 {
            // First segment is a point: on the second?
            let ss = dot2(s, s);
            let on = ss == 0.0 && a == c || ss > 0.0 && cross2(sub2(a, c), s).abs() <= 1e-12 * ss.sqrt() && (0.0..=ss).contains(&dot2(sub2(a, c), s));
            return if on { Intersection2D::Point(a) } else { Intersection2D::None };
        }
        let (t0, t1) = (dot2(ca, r) / rr, dot2(sub2(d, a), r) / rr);
        let (lo, hi) = (t0.min(t1).max(0.0), t0.max(t1).min(1.0));
        return if lo > hi {
            Intersection2D::None
        } else if lo == hi {
            Intersection2D::Point([a[0] + lo * r[0], a[1] + lo * r[1]])
        } else {
            Intersection2D::Overlap([a[0] + lo * r[0], a[1] + lo * r[1]], [a[0] + hi * r[0], a[1] + hi * r[1]])
        };
    }
    let t = cross2(ca, s) / denom;
    let u = cross2(ca, r) / denom;
    let eps = 1e-12;
    if (-eps..=1.0 + eps).contains(&t) && (-eps..=1.0 + eps).contains(&u) {
        Intersection2D::Point([a[0] + t * r[0], a[1] + t * r[1]])
    } else {
        Intersection2D::None
    }
}

/// Intersection point of two infinite lines (point + direction); `None` if parallel.
pub fn line_line_2d(p: [f64; 2], dp: [f64; 2], q: [f64; 2], dq: [f64; 2]) -> Option<[f64; 2]> {
    let denom = cross2(dp, dq);
    if denom.abs() <= 1e-14 * (dot2(dp, dp) * dot2(dq, dq)).sqrt() || denom == 0.0 {
        return None;
    }
    let t = cross2(sub2(q, p), dq) / denom;
    Some([p[0] + t * dp[0], p[1] + t * dp[1]])
}

/// The (up to two) intersection points of two circles, `(centre, radius)`. Coincident circles give
/// `None`; tangent circles give one point (repeated).
pub fn circle_circle(c1: [f64; 2], r1: f64, c2: [f64; 2], r2: f64) -> Option<[[f64; 2]; 2]> {
    let d = sub2(c2, c1);
    let dist = dot2(d, d).sqrt();
    if dist == 0.0 || dist > r1 + r2 || dist < (r1 - r2).abs() {
        return None;
    }
    let a = (r1 * r1 - r2 * r2 + dist * dist) / (2.0 * dist);
    let h = (r1 * r1 - a * a).max(0.0).sqrt();
    let m = [c1[0] + a * d[0] / dist, c1[1] + a * d[1] / dist];
    let off = [-h * d[1] / dist, h * d[0] / dist];
    Some([[m[0] + off[0], m[1] + off[1]], [m[0] - off[0], m[1] - off[1]]])
}

/// Where a ray (or line) meets a plane: the parameter `t` such that `origin + t dir` lies on the
/// plane through `plane_point` with `normal`. `None` if parallel (including lying in the plane).
pub fn line_plane_parameter(origin: [f64; 3], dir: [f64; 3], plane_point: [f64; 3], normal: [f64; 3]) -> Option<f64> {
    let denom = dot3(dir, normal);
    let scale = (dot3(dir, dir) * dot3(normal, normal)).sqrt();
    if denom.abs() <= 1e-14 * scale || denom == 0.0 {
        return None;
    }
    Some(dot3(sub3(plane_point, origin), normal) / denom)
}

/// Ray-plane intersection point (`t >= 0`).
pub fn ray_plane(origin: [f64; 3], dir: [f64; 3], plane_point: [f64; 3], normal: [f64; 3]) -> Option<[f64; 3]> {
    line_plane_parameter(origin, dir, plane_point, normal).filter(|&t| t >= 0.0).map(|t| at3(origin, dir, t))
}

/// Segment-plane intersection point, if the segment `a`-`b` crosses or touches the plane in a
/// single point.
pub fn segment_plane(a: [f64; 3], b: [f64; 3], plane_point: [f64; 3], normal: [f64; 3]) -> Option<[f64; 3]> {
    let d = sub3(b, a);
    line_plane_parameter(a, d, plane_point, normal).filter(|t| (0.0..=1.0).contains(t)).map(|t| at3(a, d, t))
}

/// The line where two planes meet, as `(point, unit direction)`; `None` if parallel.
pub fn plane_plane(p1: [f64; 3], n1: [f64; 3], p2: [f64; 3], n2: [f64; 3]) -> Option<([f64; 3], [f64; 3])> {
    let dir = cross3(n1, n2);
    let d2 = dot3(dir, dir);
    if d2 <= 1e-28 * dot3(n1, n1) * dot3(n2, n2) || d2 == 0.0 {
        return None;
    }
    let (h1, h2) = (dot3(n1, p1), dot3(n2, p2));
    // point = (h1 (n2 × dir) + h2 (dir × n1)) / |dir|²
    let a = cross3(n2, dir);
    let b = cross3(dir, n1);
    let point = [(h1 * a[0] + h2 * b[0]) / d2, (h1 * a[1] + h2 * b[1]) / d2, (h1 * a[2] + h2 * b[2]) / d2];
    let len = d2.sqrt();
    Some((point, [dir[0] / len, dir[1] / len, dir[2] / len]))
}

/// A ray-triangle hit: distance along the ray and barycentric weights of the hit point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleHit {
    pub t: f64,
    /// Weights `[wa, wb, wc]` of the triangle's vertices at the hit point.
    pub barycentric: [f64; 3],
}

/// Ray-triangle intersection by the Möller-Trumbore algorithm. Both faces are hit; `t >= 0`.
pub fn ray_triangle(origin: [f64; 3], dir: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Option<TriangleHit> {
    let (e1, e2) = (sub3(b, a), sub3(c, a));
    let h = cross3(dir, e2);
    let det = dot3(e1, h);
    if det.abs() < 1e-14 * (dot3(e1, e1) * dot3(e2, e2)).sqrt() * dot3(dir, dir).sqrt() || det == 0.0 {
        return None;
    }
    let s = sub3(origin, a);
    let u = dot3(s, h) / det;
    let q = cross3(s, e1);
    let v = dot3(dir, q) / det;
    let t = dot3(e2, q) / det;
    let eps = 1e-12;
    (u >= -eps && v >= -eps && u + v <= 1.0 + eps && t >= 0.0).then_some(TriangleHit { t, barycentric: [1.0 - u - v, u, v] })
}

/// Segment-triangle intersection: the hit if the segment `p`-`q` reaches the triangle.
pub fn segment_triangle(p: [f64; 3], q: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Option<TriangleHit> {
    let d = sub3(q, p);
    ray_triangle(p, d, a, b, c).filter(|h| h.t <= 1.0)
}

/// Ray-sphere intersection: the entry and exit parameters `(t_near, t_far)` with `t_far >= 0`
/// (`t_near` is negative when the origin is inside).
pub fn ray_sphere(origin: [f64; 3], dir: [f64; 3], centre: [f64; 3], radius: f64) -> Option<(f64, f64)> {
    let m = sub3(origin, centre);
    let a = dot3(dir, dir);
    if a == 0.0 {
        return None;
    }
    let b = dot3(m, dir);
    let disc = b * b - a * (dot3(m, m) - radius * radius);
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let (t0, t1) = ((-b - s) / a, (-b + s) / a);
    (t1 >= 0.0).then_some((t0, t1))
}

/// Ray-box intersection by the slab method: `(t_near, t_far)` with `t_far >= max(t_near, 0)`.
pub fn ray_aabb(origin: [f64; 3], dir: [f64; 3], min: [f64; 3], max: [f64; 3]) -> Option<(f64, f64)> {
    let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
    for k in 0..3 {
        if dir[k] == 0.0 {
            if origin[k] < min[k] || origin[k] > max[k] {
                return None;
            }
        } else {
            let (a, b) = ((min[k] - origin[k]) / dir[k], (max[k] - origin[k]) / dir[k]);
            t0 = t0.max(a.min(b));
            t1 = t1.min(a.max(b));
        }
    }
    (t1 >= t0.max(0.0)).then_some((t0, t1))
}

/// Whether two axis-aligned boxes overlap (touching counts).
pub fn aabb_aabb<const D: usize>(min1: [f64; D], max1: [f64; D], min2: [f64; D], max2: [f64; D]) -> bool {
    (0..D).all(|k| min1[k] <= max2[k] && min2[k] <= max1[k])
}

/// Whether two spheres/circles/balls overlap (touching counts).
pub fn sphere_sphere<const D: usize>(c1: [f64; D], r1: f64, c2: [f64; D], r2: f64) -> bool {
    let d2: f64 = (0..D).map(|k| (c1[k] - c2[k]) * (c1[k] - c2[k])).sum();
    d2 <= (r1 + r2) * (r1 + r2)
}

/// The intersection of triangle `abc` with a plane as a segment `(p, q)`, if the plane cuts the
/// triangle properly (crosses it, or contains exactly one of its edges). Triangles lying in the
/// plane, or touching it at a single vertex, give `None`. Distances are signed against the plane
/// through `plane_point` with `normal`; values within `tol` of zero count as on the plane.
pub fn triangle_plane(a: [f64; 3], b: [f64; 3], c: [f64; 3], plane_point: [f64; 3], normal: [f64; 3], tol: f64) -> Option<([f64; 3], [f64; 3])> {
    let v = [a, b, c];
    let nlen = dot3(normal, normal).sqrt();
    if nlen == 0.0 {
        return None;
    }
    let d: Vec<f64> = v.iter().map(|&p| dot3(sub3(p, plane_point), normal) / nlen).map(|x| if x.abs() <= tol { 0.0 } else { x }).collect();
    let (pos, neg, zero) = (d.iter().filter(|&&x| x > 0.0).count(), d.iter().filter(|&&x| x < 0.0).count(), d.iter().filter(|&&x| x == 0.0).count());
    if (pos == 0 && neg == 0) || pos + zero == 3 && zero < 2 || neg + zero == 3 && zero < 2 {
        return None; // coplanar, or touching at one vertex / entirely on one side
    }
    if zero == 2 {
        let idx: Vec<usize> = (0..3).filter(|&i| d[i] == 0.0).collect();
        return Some((v[idx[0]], v[idx[1]]));
    }
    // Collect the points where triangle edges cross the plane (zero vertices count as crossings).
    let mut pts: Vec<[f64; 3]> = Vec::with_capacity(2);
    for i in 0..3 {
        let j = (i + 1) % 3;
        if d[i] == 0.0 {
            pts.push(v[i]);
        } else if d[i] * d[j] < 0.0 {
            let t = d[i] / (d[i] - d[j]);
            pts.push(at3(v[i], sub3(v[j], v[i]), t));
        }
    }
    pts.dedup();
    (pts.len() == 2).then(|| (pts[0], pts[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close<const D: usize>(a: [f64; D], b: [f64; D]) -> bool {
        (0..D).all(|k| (a[k] - b[k]).abs() < 1e-12)
    }

    #[test]
    fn segments_and_lines_2d() {
        assert_eq!(segment_segment_2d([0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]), Intersection2D::Point([1.0, 1.0]));
        assert_eq!(segment_segment_2d([0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, -1.0]), Intersection2D::None);
        // Touching at an endpoint (T junction).
        assert_eq!(segment_segment_2d([0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 3.0]), Intersection2D::Point([1.0, 0.0]));
        // Collinear: overlap, touching, and disjoint.
        assert_eq!(segment_segment_2d([0.0, 0.0], [4.0, 0.0], [2.0, 0.0], [6.0, 0.0]), Intersection2D::Overlap([2.0, 0.0], [4.0, 0.0]));
        assert_eq!(segment_segment_2d([0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]), Intersection2D::Point([1.0, 0.0]));
        assert_eq!(segment_segment_2d([0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]), Intersection2D::None);
        assert_eq!(segment_segment_2d([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]), Intersection2D::None);
        assert!(close(line_line_2d([0.0, 0.0], [1.0, 1.0], [0.0, 2.0], [1.0, -1.0]).unwrap(), [1.0, 1.0]));
        assert!(line_line_2d([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0]).is_none());
        let pts = circle_circle([0.0, 0.0], 5.0, [6.0, 0.0], 5.0).unwrap();
        assert!(close(pts[0], [3.0, 4.0]) && close(pts[1], [3.0, -4.0]));
        assert!(circle_circle([0.0, 0.0], 1.0, [5.0, 0.0], 1.0).is_none() && circle_circle([0.0, 0.0], 3.0, [1.0, 0.0], 1.0).is_none());
        let tangent = circle_circle([0.0, 0.0], 1.0, [2.0, 0.0], 1.0).unwrap();
        assert!(close(tangent[0], [1.0, 0.0]) && close(tangent[1], [1.0, 0.0]));
    }

    #[test]
    fn planes() {
        let (o, d) = ([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(close(ray_plane(o, d, [0.0; 3], [0.0, 0.0, 1.0]).unwrap(), [0.0, 0.0, 0.0]));
        assert!(ray_plane(o, [0.0, 0.0, 1.0], [0.0; 3], [0.0, 0.0, 1.0]).is_none()); // behind
        assert!(ray_plane(o, [1.0, 0.0, 0.0], [0.0; 3], [0.0, 0.0, 1.0]).is_none()); // parallel
        assert!(close(segment_plane([0.0, 0.0, -1.0], [2.0, 0.0, 3.0], [0.0; 3], [0.0, 0.0, 1.0]).unwrap(), [0.5, 0.0, 0.0]));
        assert!(segment_plane([0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0; 3], [0.0, 0.0, 1.0]).is_none());
        let (p, dir) = plane_plane([0.0; 3], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
        assert!(close(p, [0.0; 3]) && close([dir[0].abs(), dir[1].abs(), dir[2].abs()], [0.0, 1.0, 0.0]));
        // Offset planes z = 1 and x = 2 meet along the line (2, y, 1).
        let (p, dir) = plane_plane([0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]).unwrap();
        assert!(close([p[0], p[2]], [2.0, 1.0]) && dir[0].abs() < 1e-12 && dir[2].abs() < 1e-12);
        assert!(plane_plane([0.0; 3], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]).is_none());
    }

    #[test]
    fn rays_against_triangles_spheres_and_boxes() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]);
        let hit = ray_triangle([1.0, 1.0, 5.0], [0.0, 0.0, -1.0], a, b, c).unwrap();
        assert!((hit.t - 5.0).abs() < 1e-12 && close(hit.barycentric, [0.5, 0.25, 0.25]));
        assert!(ray_triangle([3.0, 3.0, 5.0], [0.0, 0.0, -1.0], a, b, c).is_none()); // outside
        assert!(ray_triangle([1.0, 1.0, 5.0], [0.0, 0.0, 1.0], a, b, c).is_none()); // pointing away
        assert!(ray_triangle([1.0, 1.0, 5.0], [1.0, 0.0, 0.0], a, b, c).is_none()); // parallel
        assert!(segment_triangle([1.0, 1.0, 5.0], [1.0, 1.0, 1.0], a, b, c).is_none() && segment_triangle([1.0, 1.0, 5.0], [1.0, 1.0, -1.0], a, b, c).is_some());
        let (n, f) = ray_sphere([0.0, 0.0, -5.0], [0.0, 0.0, 1.0], [0.0; 3], 2.0).unwrap();
        assert!((n - 3.0).abs() < 1e-12 && (f - 7.0).abs() < 1e-12);
        let (n, _) = ray_sphere([0.0; 3], [1.0, 0.0, 0.0], [0.0; 3], 2.0).unwrap();
        assert!(n < 0.0);
        assert!(ray_sphere([0.0, 3.0, -5.0], [0.0, 0.0, 1.0], [0.0; 3], 2.0).is_none() && ray_sphere([0.0, 0.0, 5.0], [0.0, 0.0, 1.0], [0.0; 3], 2.0).is_none());
        let (n, f) = ray_aabb([-5.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0; 3], [1.0; 3]).unwrap();
        assert!((n - 5.0).abs() < 1e-12 && (f - 6.0).abs() < 1e-12);
        assert!(ray_aabb([-5.0, 2.0, 0.5], [1.0, 0.0, 0.0], [0.0; 3], [1.0; 3]).is_none() && ray_aabb([5.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0; 3], [1.0; 3]).is_none());
        assert!(aabb_aabb([0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]) && !aabb_aabb([0.0; 3], [1.0; 3], [1.1; 3], [2.0; 3]));
        assert!(sphere_sphere([0.0; 3], 1.0, [2.0, 0.0, 0.0], 1.0) && !sphere_sphere([0.0, 0.0], 1.0, [3.0, 0.0], 1.0));
    }

    #[test]
    fn triangle_plane_cases() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 4.0]);
        let n = [0.0, 1.0, 0.0];
        // Plane y = 0 contains the whole triangle: no unique segment.
        assert!(triangle_plane(a, b, c, [0.0; 3], n, 1e-12).is_none());
        // Cutting through: vertices at y = -1, -1, +3 -> segment between two edge crossings.
        let (p, q) = triangle_plane([0.0, -1.0, 0.0], [4.0, -1.0, 0.0], [0.0, 3.0, 0.0], [0.0; 3], n, 1e-12).unwrap();
        assert!((p[1]).abs() < 1e-12 && (q[1]).abs() < 1e-12 && (p[0] - q[0]).abs() > 0.5);
        // Plane through one vertex and the opposite edge.
        let (p, q) = triangle_plane([0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, -1.0, 0.0], [0.0; 3], n, 1e-12).unwrap();
        assert!(close(p, [0.0; 3]) && close(q, [2.0, 0.0, 0.0]));
        // Plane containing an edge, triangle on one side.
        let (p, q) = triangle_plane([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0; 3], n, 1e-12).unwrap();
        assert!(close(p, [0.0; 3]) && close(q, [1.0, 0.0, 0.0]));
        // Touching at a single vertex, or entirely on one side.
        assert!(triangle_plane([0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0; 3], n, 1e-12).is_none());
        assert!(triangle_plane([0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 3.0, 1.0], [0.0; 3], n, 1e-12).is_none());
    }
}
