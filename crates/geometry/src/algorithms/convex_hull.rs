//! Convex hulls of point sets in two and three dimensions.

use std::collections::HashMap;

fn cross2(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// The convex hull of 2D points by Andrew's monotone chain, `O(n log n)`.
///
/// Returns indices into `points` of the hull vertices in counter-clockwise order, starting from
/// the lexicographically smallest point. Collinear points on hull edges and duplicate points are
/// excluded. Fewer than three distinct points, or all-collinear input, yields the extreme points
/// only (zero, one or two indices).
pub fn convex_hull_2d(points: &[[f64; 2]]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..points.len()).filter(|&i| points[i].iter().all(|v| v.is_finite())).collect();
    order.sort_by(|&a, &b| points[a][0].total_cmp(&points[b][0]).then(points[a][1].total_cmp(&points[b][1])));
    order.dedup_by(|a, b| points[*a] == points[*b]);
    if order.len() < 3 {
        return order;
    }
    let chain = |iter: &mut dyn Iterator<Item = usize>| {
        let mut half: Vec<usize> = Vec::new();
        for i in iter {
            while half.len() >= 2 && cross2(points[half[half.len() - 2]], points[half[half.len() - 1]], points[i]) <= 0.0 {
                half.pop();
            }
            half.push(i);
        }
        half
    };
    let mut lower = chain(&mut order.iter().copied());
    let mut upper = chain(&mut order.iter().rev().copied());
    // Each chain ends at the other's starting point, so drop the duplicated last vertex of each.
    lower.pop();
    upper.pop();
    let mut hull = lower;
    hull.append(&mut upper);
    hull
}

/// Area enclosed by a polygon given by `indices` into `points` (shoelace formula; positive for
/// counter-clockwise order).
pub fn polygon_area_2d(points: &[[f64; 2]], indices: &[usize]) -> f64 {
    let n = indices.len();
    0.5 * (0..n)
        .map(|k| {
            let (a, b) = (points[indices[k]], points[indices[(k + 1) % n]]);
            a[0] * b[1] - a[1] * b[0]
        })
        .sum::<f64>()
}

/// Whether `p` lies inside or on a convex counter-clockwise polygon.
pub fn convex_polygon_contains_2d(points: &[[f64; 2]], hull: &[usize], p: [f64; 2]) -> bool {
    let n = hull.len();
    n >= 3 && (0..n).all(|k| cross2(points[hull[k]], points[hull[(k + 1) % n]], p) >= 0.0)
}

/// A convex polyhedron: triangular faces indexing into the input points, with counter-clockwise
/// winding when seen from outside (normals point outward).
#[derive(Debug, Clone, PartialEq)]

pub struct Hull3D {
    pub faces: Vec<[usize; 3]>,
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl Hull3D {
    /// Sorted indices of the points that are hull vertices.
    pub fn vertices(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.faces.iter().flatten().copied().collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Enclosed volume.
    pub fn volume(&self, points: &[[f64; 3]]) -> f64 {
        let origin = points[self.faces[0][0]];
        self.faces
            .iter()
            .map(|f| {
                let (a, b, c) = (sub3(points[f[0]], origin), sub3(points[f[1]], origin), sub3(points[f[2]], origin));
                dot3(a, cross3(b, c)) / 6.0
            })
            .sum()
    }

    /// Total surface area.
    pub fn surface_area(&self, points: &[[f64; 3]]) -> f64 {
        self.faces
            .iter()
            .map(|f| {
                let n = cross3(sub3(points[f[1]], points[f[0]]), sub3(points[f[2]], points[f[0]]));
                0.5 * dot3(n, n).sqrt()
            })
            .sum()
    }

    /// Whether `p` is inside or on the hull, within `tol` of every face plane.
    pub fn contains(&self, points: &[[f64; 3]], p: [f64; 3], tol: f64) -> bool {
        self.faces.iter().all(|f| {
            let n = cross3(sub3(points[f[1]], points[f[0]]), sub3(points[f[2]], points[f[0]]));
            let len = dot3(n, n).sqrt();
            dot3(n, sub3(p, points[f[0]])) <= tol * len
        })
    }
}

/// The 3D convex hull by incremental insertion (`O(n·f)`, where `f` is the face count, which is
/// fast for typical inputs).
///
/// Returns `None` if there are fewer than four points or they are all coplanar (within a tolerance
/// relative to the extent of the data) or contain non-finite values. Points inside the hull, on a
/// face, or duplicated are not vertices of the result.
pub fn convex_hull_3d(points: &[[f64; 3]]) -> Option<Hull3D> {
    let n = points.len();
    if n < 4 || points.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let scale = points.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
    let eps = 1e-12 * scale;

    // Initial tetrahedron: extreme points along x, then the farthest from that line, then plane.
    let i0 = (0..n).min_by(|&a, &b| points[a][0].total_cmp(&points[b][0]))?;
    let i1 = (0..n).max_by(|&a, &b| dot3(sub3(points[a], points[i0]), sub3(points[a], points[i0])).total_cmp(&dot3(sub3(points[b], points[i0]), sub3(points[b], points[i0]))))?;
    if dot3(sub3(points[i1], points[i0]), sub3(points[i1], points[i0])).sqrt() <= eps {
        return None;
    }
    let d01 = sub3(points[i1], points[i0]);
    let line_dist = |i: usize| {
        let c = cross3(d01, sub3(points[i], points[i0]));
        dot3(c, c).sqrt() / dot3(d01, d01).sqrt()
    };
    let i2 = (0..n).max_by(|&a, &b| line_dist(a).total_cmp(&line_dist(b)))?;
    if line_dist(i2) <= eps {
        return None;
    }
    let normal = cross3(d01, sub3(points[i2], points[i0]));
    let nlen = dot3(normal, normal).sqrt();
    let plane_dist = |i: usize| dot3(normal, sub3(points[i], points[i0])) / nlen;
    let i3 = (0..n).max_by(|&a, &b| plane_dist(a).abs().total_cmp(&plane_dist(b).abs()))?;
    if plane_dist(i3).abs() <= eps {
        return None;
    }

    // Orient faces outward: the fourth vertex must be on the negative side of every face.
    let mut faces: Vec<[usize; 3]> = if plane_dist(i3) > 0.0 {
        vec![[i0, i2, i1], [i0, i1, i3], [i1, i2, i3], [i2, i0, i3]]
    } else {
        vec![[i0, i1, i2], [i0, i3, i1], [i1, i3, i2], [i2, i3, i0]]
    };
    let signed = |f: &[usize; 3], p: [f64; 3]| {
        let n = cross3(sub3(points[f[1]], points[f[0]]), sub3(points[f[2]], points[f[0]]));
        dot3(n, sub3(p, points[f[0]])) / dot3(n, n).sqrt()
    };

    for (p, &point) in points.iter().enumerate() {
        if [i0, i1, i2, i3].contains(&p) {
            continue;
        }
        let visible: Vec<bool> = faces.iter().map(|f| signed(f, point) > eps).collect();
        if !visible.iter().any(|&v| v) {
            continue;
        }
        // Horizon: directed edges of visible faces whose reverse edge belongs to a hidden face.
        let mut edges: HashMap<(usize, usize), bool> = HashMap::new();
        for (f, &vis) in faces.iter().zip(&visible) {
            for k in 0..3 {
                edges.insert((f[k], f[(k + 1) % 3]), vis);
            }
        }
        let mut next: Vec<[usize; 3]> = Vec::with_capacity(faces.len() + 8);
        for (f, &vis) in faces.iter().zip(&visible) {
            if vis {
                for k in 0..3 {
                    let (a, b) = (f[k], f[(k + 1) % 3]);
                    if edges.get(&(b, a)) == Some(&false) {
                        next.push([a, b, p]);
                    }
                }
            } else {
                next.push(*f);
            }
        }
        faces = next;
    }
    Some(Hull3D { faces })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_with_interior_collinear_and_duplicate_points() {
        let pts = [[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0], [0.0, 0.0], [2.0, 1.0]];
        let hull = convex_hull_2d(&pts);
        assert_eq!(hull.len(), 4);
        let corners: Vec<[f64; 2]> = hull.iter().map(|&i| pts[i]).collect();
        assert_eq!(corners, vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]); // CCW from the smallest point
        assert!((polygon_area_2d(&pts, &hull) - 4.0).abs() < 1e-12);
        assert!(convex_polygon_contains_2d(&pts, &hull, [1.0, 1.9]));
        assert!(convex_polygon_contains_2d(&pts, &hull, [2.0, 2.0]));
        assert!(!convex_polygon_contains_2d(&pts, &hull, [2.1, 1.0]));
    }

    #[test]
    fn degenerate_2d_inputs() {
        assert!(convex_hull_2d(&[]).is_empty());
        assert_eq!(convex_hull_2d(&[[1.0, 1.0], [1.0, 1.0]]).len(), 1);
        assert_eq!(convex_hull_2d(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]).len(), 2);
        assert_eq!(convex_hull_2d(&[[0.0, 0.0], [f64::NAN, 1.0], [1.0, 0.0], [0.0, 1.0]]).len(), 3);
    }

    #[test]
    fn random_2d_hull_contains_everything_and_is_convex() {
        let mut s = 12345u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let pts: Vec<[f64; 2]> = (0..500).map(|_| [rnd() * 10.0 - 5.0, rnd() * 10.0 - 5.0]).collect();
        let hull = convex_hull_2d(&pts);
        assert!(pts.iter().all(|&p| convex_polygon_contains_2d(&pts, &hull, p)));
        let n = hull.len();
        assert!((0..n).all(|k| cross2(pts[hull[k]], pts[hull[(k + 1) % n]], pts[hull[(k + 2) % n]]) > 0.0));
        assert!(polygon_area_2d(&pts, &hull) > 0.0);
    }

    fn cube_corners() -> Vec<[f64; 3]> {
        (0..8).map(|i| [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64]).collect()
    }

    #[test]
    fn cube_hull_has_unit_volume_and_closed_surface() {
        let mut pts = cube_corners();
        pts.extend([[0.5, 0.5, 0.5], [0.2, 0.7, 0.4], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]); // interior, on a face, duplicate
        let hull = convex_hull_3d(&pts).unwrap();
        assert_eq!(hull.vertices(), (0..8).collect::<Vec<_>>());
        assert!((hull.volume(&pts) - 1.0).abs() < 1e-12);
        assert!((hull.surface_area(&pts) - 6.0).abs() < 1e-12);
        // Closed manifold: every directed edge has exactly one opposite partner.
        let mut count: HashMap<(usize, usize), i32> = HashMap::new();
        for f in &hull.faces {
            for k in 0..3 {
                *count.entry((f[k], f[(k + 1) % 3])).or_default() += 1;
            }
        }
        assert!(count.iter().all(|(&(a, b), &c)| c == 1 && count.get(&(b, a)) == Some(&1)));
        assert!(hull.contains(&pts, [0.9, 0.1, 0.5], 1e-12) && !hull.contains(&pts, [1.1, 0.5, 0.5], 1e-12));
    }

    #[test]
    fn sphere_samples_form_a_valid_hull() {
        let mut s = 99u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let pts: Vec<[f64; 3]> = (0..300)
            .map(|_| {
                let (z, t) = (2.0 * rnd() - 1.0, 2.0 * std::f64::consts::PI * rnd());
                let r = (1.0 - z * z).sqrt();
                [r * t.cos(), r * t.sin(), z]
            })
            .collect();
        let hull = convex_hull_3d(&pts).unwrap();
        assert!(pts.iter().all(|&p| hull.contains(&pts, p, 1e-9)));
        let (v, f) = (hull.vertices().len() as i64, hull.faces.len() as i64);
        assert_eq!(v - 3 * f / 2 + f, 2, "Euler characteristic");
        assert!(hull.volume(&pts) > 0.0 && hull.volume(&pts) < 4.18879);
        assert!(hull.volume(&pts) > 3.9, "volume {}", hull.volume(&pts));
    }

    #[test]
    fn degenerate_3d_inputs_return_none() {
        assert!(convex_hull_3d(&[[0.0; 3]; 3]).is_none());
        let planar: Vec<[f64; 3]> = (0..10).map(|i| [i as f64, (i * i) as f64 % 7.0, 0.0]).collect();
        assert!(convex_hull_3d(&planar).is_none());
        let line: Vec<[f64; 3]> = (0..10).map(|i| [i as f64, 2.0 * i as f64, 3.0 * i as f64]).collect();
        assert!(convex_hull_3d(&line).is_none());
        assert!(convex_hull_3d(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [f64::NAN, 0.0, 1.0]]).is_none());
    }
}
