//! Delaunay triangulation of 2D point sets.
//!
//! See the book entry [Delaunay triangulations and Voronoi diagrams](https://jwkunz.github.io/lib_jsl/book/geometry/delaunay-voronoi.html#delaunay-triangulation).

use std::collections::{HashMap, HashSet};

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

/// Orientation with a rounding-error bound: the determinant and the sum of its terms' magnitudes.
fn orient_bound(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> (f64, f64) {
    let (l, r) = ((b[0] - a[0]) * (c[1] - a[1]), (b[1] - a[1]) * (c[0] - a[0]));
    (l - r, l.abs() + r.abs())
}

/// The in-circle determinant and the sum of its terms' magnitudes (its rounding-error scale).
fn in_circle_bound(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> (f64, f64) {
    let (ax, ay, bx, by, cx, cy) = (a[0] - d[0], a[1] - d[1], b[0] - d[0], b[1] - d[1], c[0] - d[0], c[1] - d[1]);
    let (a2, b2, c2) = (ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy);
    let det = in_circle(a, b, c, d);
    let scale = a2 * ((bx * cy).abs() + (cx * by).abs()) + b2 * ((ax * cy).abs() + (cx * ay).abs()) + c2 * ((ax * by).abs() + (bx * ay).abs());
    (det, scale)
}

/// Relative slack for the predicates: determinants smaller than this times their term scale are
/// treated as zero (collinear or cocircular), which keeps rounding from creating slivers or
/// flipping an edge back and forth.
const EPS: f64 = 1e-12;

/// Triangulates `points` by a sweep with Lawson flips (`O(n log n)` typical): points are inserted
/// in lexicographic order, so each lies outside the triangulation so far and is joined to the hull
/// edges it can see; every new edge is then checked against the empty-circle test and flipped
/// until the triangulation is Delaunay again.
///
/// Duplicate points are ignored (they appear in no triangle). Returns `None` for fewer than three
/// distinct finite points or when all points are collinear. Points cocircular to rounding are
/// triangulated in either valid way.
pub fn delaunay_triangulation(points: &[[f64; 2]]) -> Option<Triangulation> {
    let mut seen: HashSet<(u64, u64)> = HashSet::new();
    let mut ids: Vec<usize> = (0..points.len())
        .filter(|&i| points[i].iter().all(|v| v.is_finite()) && seen.insert(((points[i][0] + 0.0).to_bits(), (points[i][1] + 0.0).to_bits())))
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
    ids.sort_by(|&i, &j| points[i][0].total_cmp(&points[j][0]).then(points[i][1].total_cmp(&points[j][1])));
    let p: Vec<[f64; 2]> = ids.iter().map(|&i| [(points[i][0] - lo[0]) / span, (points[i][1] - lo[1]) / span]).collect();
    let n = p.len();
    let left = |a: usize, b: usize, c: usize| {
        let (d, s) = orient_bound(p[a], p[b], p[c]);
        if d > EPS * s { 1 } else if d < -EPS * s { -1 } else { 0 }
    };

    // The first points may be collinear; the first point off their line closes a fan over them.
    let mut k = 2;
    while k < n && left(0, 1, k) == 0 {
        k += 1;
    }
    if k == n {
        return None;
    }
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(2 * n);
    let mut edge: HashMap<(usize, usize), usize> = HashMap::with_capacity(6 * n);
    let add = |tris: &mut Vec<[usize; 3]>, edge: &mut HashMap<(usize, usize), usize>, t: [usize; 3]| {
        for j in 0..3 {
            edge.insert((t[j], t[(j + 1) % 3]), tris.len());
        }
        tris.push(t);
    };
    // The hull is a counter-clockwise cycle of vertices.
    let (mut next, mut prev) = (vec![usize::MAX; n], vec![usize::MAX; n]);
    let side = left(0, 1, k);
    let mut cycle: Vec<usize> = Vec::with_capacity(k + 1);
    for j in 0..k - 1 {
        add(&mut tris, &mut edge, if side > 0 { [j, j + 1, k] } else { [j + 1, j, k] });
    }
    if side > 0 {
        cycle.extend(0..=k);
    } else {
        cycle.push(0);
        cycle.push(k);
        cycle.extend((1..k).rev());
    }
    for j in 0..cycle.len() {
        next[cycle[j]] = cycle[(j + 1) % cycle.len()];
        prev[cycle[(j + 1) % cycle.len()]] = cycle[j];
    }

    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut last = k;
    for i in k + 1..n {
        // Find a hull edge a -> next[a] that sees p_i (p_i to its right), near the last insertion.
        let sees = |a: usize, next: &[usize]| left(a, next[a], i) < 0;
        let start = if sees(last, &next) {
            Some(last)
        } else if sees(prev[last], &next) {
            Some(prev[last])
        } else {
            let mut a = next[last];
            loop {
                if sees(a, &next) {
                    break Some(a);
                }
                if a == last {
                    break None;
                }
                a = next[a];
            }
        };
        let Some(start) = start else { continue };
        // Extend to the whole visible chain first..=end (edges first -> ... -> end -> next[end]).
        let (mut first, mut end) = (start, start);
        while prev[first] != end && sees(prev[first], &next) {
            first = prev[first];
        }
        while next[end] != first && sees(next[end], &next) {
            end = next[end];
        }
        let mut a = first;
        loop {
            let b = next[a];
            add(&mut tris, &mut edge, [b, a, i]);
            stack.push((b, a));
            if a == end {
                break;
            }
            a = b;
        }
        let after = next[end];
        next[first] = i;
        prev[i] = first;
        next[i] = after;
        prev[after] = i;
        last = i;

        // ANCHOR: lawson
        // Lawson flips: edge (a, b) of triangle (a, b, c) is illegal if the triangle (b, a, d)
        // across it has d inside the circle through a, b, c.
        while let Some((a, b)) = stack.pop() {
            let (Some(&t), Some(&u)) = (edge.get(&(a, b)), edge.get(&(b, a))) else { continue };
            let c = tris[t].into_iter().find(|&v| v != a && v != b).unwrap();
            let d = tris[u].into_iter().find(|&v| v != a && v != b).unwrap();
            let (det, scale) = in_circle_bound(p[a], p[b], p[c], p[d]);
            if det <= EPS * scale || left(c, a, d) <= 0 || left(c, d, b) <= 0 {
                continue;
            }
            edge.remove(&(a, b));
            edge.remove(&(b, a));
            tris[t] = [c, a, d];
            tris[u] = [c, d, b];
            for (x, y, tri) in [(c, a, t), (a, d, t), (d, c, t), (c, d, u), (d, b, u), (b, c, u)] {
                edge.insert((x, y), tri);
            }
            stack.push((a, d));
            stack.push((d, b));
        }
        // ANCHOR_END: lawson
    }
    let triangles = tris.into_iter().map(|t| [ids[t[0]], ids[t[1]], ids[t[2]]]).collect();
    Some(Triangulation { triangles })
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

    fn assert_tiles_hull(pts: &[[f64; 2]]) {
        let t = delaunay_triangulation(pts).unwrap();
        let hull = convex_hull_2d(pts);
        let hull_area = polygon_area_2d(pts, &hull);
        assert!((t.area(pts) - hull_area).abs() < 1e-9 * hull_area, "{} vs {}", t.area(pts), hull_area);
        assert!(t.triangles.iter().all(|tr| orient(pts[tr[0]], pts[tr[1]], pts[tr[2]]) > 0.0));
        assert!(t.is_delaunay(pts, 1e-9));
    }

    #[test]
    fn nearly_cocircular_points_do_not_overlap() {
        // Forty points on a shallow arc and one near its centre: almost every quadruple is
        // cocircular, which made the earlier Bowyer-Watson version produce overlapping triangles.
        let mut pts: Vec<[f64; 2]> = (0..40)
            .map(|i| {
                let t = -0.3 + 0.6 * i as f64 / 39.0;
                [10.0 * f64::sin(t), 10.0 * f64::cos(t) - 9.0]
            })
            .collect();
        pts.push([0.0, 0.0]);
        assert_tiles_hull(&pts);
        let circle: Vec<[f64; 2]> = (0..100).map(|i| [f64::cos(i as f64 * 0.0628), f64::sin(i as f64 * 0.0628)]).collect();
        assert_tiles_hull(&circle);
    }

    #[test]
    fn thin_point_sets_keep_their_hull_triangles() {
        // Points close to a line: a finite super-triangle used to cut off the hull's sliver triangles.
        let mut rnd = lcg(3);
        let pts: Vec<[f64; 2]> = (0..100)
            .map(|i| {
                let x = i as f64 / 99.0;
                [x, 1e-3 * x * (1.0 - x) + 1e-4 * rnd()]
            })
            .collect();
        assert_tiles_hull(&pts);
        let mut fan: Vec<[f64; 2]> = (0..10).map(|i| [0.0, i as f64]).collect();
        fan.push([-1.0, 3.0]);
        assert_tiles_hull(&fan);
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
