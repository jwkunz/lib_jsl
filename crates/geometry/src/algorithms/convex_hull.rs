//! Convex hulls of point sets in two and three dimensions.
//!
//! See the book entry [Convex hulls and polygon booleans](https://jwkunz.github.io/lib_jsl/book/geometry/hulls-booleans.html).

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
    /// Sorted indices of the points used by the faces: every extreme point, plus any points on a
    /// face or edge that the construction kept (see [`convex_hull_3d`]).
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

/// The 3D convex hull by Quickhull (Barber, Dobkin and Huhdanpaa): starting from a tetrahedron,
/// each face keeps the points outside it, and the point farthest above a face is added by
/// replacing the faces it sees with a cone to their horizon. `O(n log n)` expected for typical
/// inputs, `O(n²)` in the worst case.
///
/// Returns `None` if there are fewer than four points or they are all coplanar (within a tolerance
/// relative to the extent of the data) or contain non-finite values. Points strictly inside the
/// hull and duplicates are never vertices of the result, and nor, usually, are points lying on a
/// face or edge of the hull (a 5×5×5 grid gives its 8 corners). Such a point can still become a
/// vertex of coplanar triangles if it is the farthest point above some intermediate face, so
/// [`Hull3D::vertices`] is guaranteed to contain the extreme points but may, for degenerate
/// inputs, contain a few more. The volume and surface area are unaffected.
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
    let faces: Vec<[usize; 3]> = if plane_dist(i3) > 0.0 {
        vec![[i0, i2, i1], [i0, i1, i3], [i1, i2, i3], [i2, i0, i3]]
    } else {
        vec![[i0, i1, i2], [i0, i3, i1], [i1, i3, i2], [i2, i3, i0]]
    };
    // Each face caches its unit normal and offset, and owns the "outside set" of unprocessed
    // points that see it; a point is in at most one set, and points in none are inside.
    let plane = |f: &[usize; 3]| {
        let n = cross3(sub3(points[f[1]], points[f[0]]), sub3(points[f[2]], points[f[0]]));
        let len = dot3(n, n).sqrt();
        let n = [n[0] / len, n[1] / len, n[2] / len];
        (n, dot3(n, points[f[0]]))
    };
    struct Face {
        vertices: [usize; 3],
        normal: [f64; 3],
        offset: f64,
        outside: Vec<usize>,
        alive: bool,
    }
    let height = |f: &Face, p: usize| dot3(f.normal, points[p]) - f.offset;
    let mut slots: Vec<Face> = faces.iter().map(|f| { let (normal, offset) = plane(f); Face { vertices: *f, normal, offset, outside: Vec::new(), alive: true } }).collect();
    let mut owner: HashMap<(usize, usize), usize> = HashMap::new();
    for (k, f) in slots.iter().enumerate() {
        for j in 0..3 {
            owner.insert((f.vertices[j], f.vertices[(j + 1) % 3]), k);
        }
    }
    for p in (0..n).filter(|p| ![i0, i1, i2, i3].contains(p)) {
        if let Some(k) = (0..4).find(|&k| height(&slots[k], p) > eps) {
            slots[k].outside.push(p);
        }
    }
    let mut pending: Vec<usize> = (0..4).collect();
    while let Some(k) = pending.pop() {
        if !slots[k].alive || slots[k].outside.is_empty() {
            continue;
        }
        // ANCHOR: quickhull
        // Quickhull: add the point farthest above this face, which is certainly a hull vertex.
        let apex = *slots[k].outside.iter().max_by(|&&a, &&b| height(&slots[k], a).total_cmp(&height(&slots[k], b))).unwrap();
        // The faces it sees form a connected cap around face k: find them across shared edges.
        let mut visible = vec![k];
        let mut seen = std::collections::HashSet::from([k]);
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < visible.len() {
            let f = slots[visible[i]].vertices;
            for j in 0..3 {
                let (a, b) = (f[j], f[(j + 1) % 3]);
                let across = owner[&(b, a)];
                if seen.contains(&across) {
                    continue;
                }
                if height(&slots[across], apex) > eps {
                    seen.insert(across);
                    visible.push(across);
                } else {
                    horizon.push((a, b));
                }
            }
            i += 1;
        }
        // Remove the cap, keeping its outside points to hand on to the new faces.
        let mut orphans = Vec::new();
        for &v in &visible {
            let f = &mut slots[v];
            f.alive = false;
            orphans.append(&mut f.outside);
            for j in 0..3 {
                owner.remove(&(f.vertices[j], f.vertices[(j + 1) % 3]));
            }
        }
        // Cone the horizon to the apex, skipping any edge that rounding has left inconsistent
        // (its face across was removed after all).
        let first_new = slots.len();
        for (a, b) in horizon {
            if owner.contains_key(&(a, b)) || !owner.contains_key(&(b, a)) {
                continue;
            }
            let vertices = [a, b, apex];
            let (normal, offset) = plane(&vertices);
            for j in 0..3 {
                owner.insert((vertices[j], vertices[(j + 1) % 3]), slots.len());
            }
            slots.push(Face { vertices, normal, offset, outside: Vec::new(), alive: true });
        }
        for p in orphans.into_iter().filter(|&p| p != apex) {
            if let Some(f) = (first_new..slots.len()).find(|&f| height(&slots[f], p) > eps) {
                slots[f].outside.push(p);
            }
        }
        pending.extend(first_new..slots.len());
        // ANCHOR_END: quickhull
    }
    let faces: Vec<[usize; 3]> = slots.into_iter().filter(|f| f.alive).map(|f| f.vertices).collect();
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
    fn grid_hull_keeps_its_volume_and_all_corners() {
        let grid: Vec<[f64; 3]> = (0..125).map(|i| [(i % 5) as f64, ((i / 5) % 5) as f64, (i / 25) as f64]).collect();
        let hull = convex_hull_3d(&grid).unwrap();
        assert!((hull.volume(&grid) - 64.0).abs() < 1e-12);
        assert!((hull.surface_area(&grid) - 96.0).abs() < 1e-12);
        let vertices = hull.vertices();
        let corners = (0..125).filter(|&i| grid[i].iter().all(|&v| v == 0.0 || v == 4.0));
        assert!(corners.into_iter().all(|c| vertices.contains(&c)));
        // Points interior to the cube are never vertices, and Quickhull keeps only the corners.
        assert!(vertices.iter().all(|&i| grid[i].iter().any(|&v| v == 0.0 || v == 4.0)));
        assert_eq!(vertices.len(), 8);
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
