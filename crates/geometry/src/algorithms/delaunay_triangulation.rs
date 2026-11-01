//! Delaunay triangulation of 2D point sets.

use std::collections::HashSet;

fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Positive if `d` is strictly inside the circumcircle of the counter-clockwise triangle `abc`.
fn in_circle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
    let (ax, ay, bx, by, cx, cy) = (a[0] - d[0], a[1] - d[1], b[0] - d[0], b[1] - d[1], c[0] - d[0], c[1] - d[1]);
    (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay) + (cx * cx + cy * cy) * (ax * by - bx * ay)
}

/// A Delaunay triangulation: triangles (indices into the input points, counter-clockwise) such that
/// no input point lies inside any triangle's circumcircle. Together the triangles tile the convex
/// hull of the points.
#[derive(Debug, Clone, PartialEq)]

pub struct Triangulation {
    pub triangles: Vec<[usize; 3]>,
}

/// Triangulates `points` with the Bowyer-Watson algorithm (`O(n²)` worst case, fast in practice
/// for the sizes this library targets).
///
/// Duplicate points are ignored (they appear in no triangle). Returns `None` for fewer than three
/// distinct finite points or when all points are collinear.
pub fn delaunay_triangulation(points: &[[f64; 2]]) -> Option<Triangulation> {
    let mut seen: HashSet<(u64, u64)> = HashSet::new();
    let ids: Vec<usize> = (0..points.len())
        .filter(|&i| points[i].iter().all(|v| v.is_finite()) && seen.insert((points[i][0].to_bits(), points[i][1].to_bits())))
        .collect();
    if ids.len() < 3 {
        return None;
    }
    // Work in coordinates normalised to a unit box for well-conditioned predicates.
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for &i in &ids {
        for k in 0..2 {
            lo[k] = lo[k].min(points[i][k]);
            hi[k] = hi[k].max(points[i][k]);
        }
    }
    let span = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    if span == 0.0 {
        return None;
    }
    let mut p: Vec<[f64; 2]> = ids.iter().map(|&i| [(points[i][0] - lo[0]) / span, (points[i][1] - lo[1]) / span]).collect();
    let n = p.len();
    // A far-away super-triangle containing every point.
    p.push([-50.0, -50.0]);
    p.push([50.0, -50.0]);
    p.push([0.5, 50.0]);
    let mut tris: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];

    for i in 0..n {
        let bad: Vec<usize> = (0..tris.len()).filter(|&t| in_circle(p[tris[t][0]], p[tris[t][1]], p[tris[t][2]], p[i]) > 0.0).collect();
        let mut directed: HashSet<(usize, usize)> = HashSet::new();
        for &t in &bad {
            for k in 0..3 {
                directed.insert((tris[t][k], tris[t][(k + 1) % 3]));
            }
        }
        let boundary: Vec<(usize, usize)> = directed.iter().copied().filter(|&(a, b)| !directed.contains(&(b, a))).collect();
        let bad_set: HashSet<usize> = bad.into_iter().collect();
        let mut next: Vec<[usize; 3]> = tris.iter().enumerate().filter(|(t, _)| !bad_set.contains(t)).map(|(_, &t)| t).collect();
        for (a, b) in boundary {
            if orient(p[a], p[b], p[i]) > 0.0 {
                next.push([a, b, i]);
            }
        }
        tris = next;
    }

    let triangles: Vec<[usize; 3]> = tris
        .into_iter()
        .filter(|t| t.iter().all(|&v| v < n))
        .map(|t| [ids[t[0]], ids[t[1]], ids[t[2]]])
        .collect();
    (!triangles.is_empty()).then_some(Triangulation { triangles })
}

impl Triangulation {
    /// The unique undirected edges as `(low, high)` index pairs, sorted.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut e: Vec<(usize, usize)> = self
            .triangles
            .iter()
            .flat_map(|t| (0..3).map(move |k| (t[k].min(t[(k + 1) % 3]), t[k].max(t[(k + 1) % 3]))))
            .collect();
        e.sort_unstable();
        e.dedup();
        e
    }

    /// Index of a triangle containing `p` (boundary inclusive), if any.
    pub fn locate(&self, points: &[[f64; 2]], p: [f64; 2]) -> Option<usize> {
        self.triangles.iter().position(|t| (0..3).all(|k| orient(points[t[k]], points[t[(k + 1) % 3]], p) >= 0.0))
    }

    /// Circumcentre of triangle `t`.
    pub fn circumcenter(&self, points: &[[f64; 2]], t: usize) -> [f64; 2] {
        let [a, b, c] = self.triangles[t].map(|i| points[i]);
        let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
        let (a2, b2, c2) = (a[0] * a[0] + a[1] * a[1], b[0] * b[0] + b[1] * b[1], c[0] * c[0] + c[1] * c[1]);
        [(a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d, (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d]
    }

    /// Total area of all triangles (equals the convex hull's area).
    pub fn area(&self, points: &[[f64; 2]]) -> f64 {
        self.triangles.iter().map(|t| 0.5 * orient(points[t[0]], points[t[1]], points[t[2]])).sum()
    }

    /// Checks the empty-circumcircle property against every input point (`O(n·t)`, for testing),
    /// allowing `tol` relative slack for points that are cocircular to rounding.
    pub fn is_delaunay(&self, points: &[[f64; 2]], tol: f64) -> bool {
        self.triangles.iter().all(|t| {
            let [a, b, c] = t.map(|i| points[i]);
            let scale = orient(a, b, c).abs().max(f64::MIN_POSITIVE);
            points.iter().enumerate().filter(|(i, _)| !t.contains(i)).all(|(_, &q)| in_circle(a, b, c, q) <= tol * scale)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::convex_hull::{convex_hull_2d, polygon_area_2d};

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn square_with_centre_gives_four_triangles() {
        let pts = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]];
        let t = delaunay_triangulation(&pts).unwrap();
        assert_eq!(t.triangles.len(), 4);
        assert!((t.area(&pts) - 4.0).abs() < 1e-12);
        assert_eq!(t.edges().len(), 8);
        assert!(t.is_delaunay(&pts, 1e-9));
        let c = t.circumcenter(&pts, 0);
        assert!(c.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn random_points_tile_the_hull_and_satisfy_the_delaunay_property() {
        let mut rnd = lcg(7);
        let pts: Vec<[f64; 2]> = (0..200).map(|_| [rnd() * 100.0, rnd() * 50.0 - 25.0]).collect();
        let t = delaunay_triangulation(&pts).unwrap();
        assert!(t.is_delaunay(&pts, 1e-9));
        let hull = convex_hull_2d(&pts);
        let hull_area = polygon_area_2d(&pts, &hull);
        assert!((t.area(&pts) - hull_area).abs() < 1e-6 * hull_area, "{} vs {}", t.area(&pts), hull_area);
        assert!(t.triangles.iter().all(|tr| orient(pts[tr[0]], pts[tr[1]], pts[tr[2]]) > 0.0));
        // Euler: for n points with h on the hull, triangles = 2n - h - 2.
        assert_eq!(t.triangles.len(), 2 * pts.len() - hull.len() - 2);
        assert!(t.locate(&pts, [50.0, 0.0]).is_some());
        assert!(t.locate(&pts, [500.0, 0.0]).is_none());
    }

    #[test]
    fn grid_with_cocircular_points_and_duplicates() {
        let mut pts: Vec<[f64; 2]> = (0..6).flat_map(|i| (0..6).map(move |j| [i as f64, j as f64])).collect();
        pts.push([2.0, 2.0]); // duplicate
        let t = delaunay_triangulation(&pts).unwrap();
        assert_eq!(t.triangles.len(), 2 * 25);
        assert!((t.area(&pts) - 25.0).abs() < 1e-9);
        assert!(t.triangles.iter().flatten().all(|&i| i != 36));
        assert!(t.is_delaunay(&pts, 1e-9));
    }

    #[test]
    fn degenerate_inputs() {
        assert!(delaunay_triangulation(&[]).is_none());
        assert!(delaunay_triangulation(&[[0.0, 0.0], [1.0, 1.0]]).is_none());
        assert!(delaunay_triangulation(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]).is_none());
        assert!(delaunay_triangulation(&[[1.0, 1.0]; 5]).is_none());
        assert_eq!(delaunay_triangulation(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]).unwrap().triangles.len(), 1);
    }
}
