//! Quad trees: point region trees in the plane.
//!
//! See the book entry [Spatial trees](https://jwkunz.github.io/lib_jsl/book/geometry/spatial-trees.html#quadtrees-and-octrees).

use super::region_tree::RegionTree;

/// A point quad tree over a rectangle. Each leaf splits into four quadrants when it overflows. See
/// [`RegionTree`] for the full interface (insert, remove, box/radius/nearest queries).
///
/// ```
/// use lib_jsl_geometry::algorithms::QuadTree;
/// let mut tree = QuadTree::new([0.0, 0.0], [10.0, 10.0]);
/// tree.insert(0, [1.0, 1.0]);
/// tree.insert(1, [9.0, 9.0]);
/// assert_eq!(tree.nearest([8.0, 8.0]).unwrap().0, 1);
/// ```
pub type QuadTree = RegionTree<2>;

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn queries_match_brute_force() {
        let mut rnd = lcg(5);
        let pts: Vec<[f64; 2]> = (0..600).map(|_| [rnd() * 20.0 - 10.0, rnd() * 8.0]).collect();
        let tree = QuadTree::from_points(&pts);
        assert_eq!(tree.len(), 600);
        assert!(tree.depth() > 1);
        let mut got = tree.query_box([-3.0, 2.0], [4.0, 6.0]);
        got.sort_unstable();
        let want: Vec<usize> = (0..pts.len()).filter(|&i| pts[i][0] >= -3.0 && pts[i][0] <= 4.0 && pts[i][1] >= 2.0 && pts[i][1] <= 6.0).collect();
        assert_eq!(got, want);
        let mut got = tree.query_radius([1.0, 4.0], 3.0);
        got.sort_unstable();
        let want: Vec<usize> = (0..pts.len()).filter(|&i| (pts[i][0] - 1.0).hypot(pts[i][1] - 4.0) <= 3.0).collect();
        assert_eq!(got, want);
        for _ in 0..20 {
            let q = [rnd() * 24.0 - 12.0, rnd() * 10.0 - 1.0];
            let mut brute: Vec<(usize, f64)> = pts.iter().enumerate().map(|(i, p)| (i, (p[0] - q[0]).hypot(p[1] - q[1]))).collect();
            brute.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            let got = tree.k_nearest(q, 5);
            assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), brute[..5].iter().map(|b| b.0).collect::<Vec<_>>());
            assert!(got.iter().zip(&brute).all(|(g, b)| (g.1 - b.1).abs() < 1e-12));
        }
    }

    #[test]
    fn insert_remove_bounds_and_coincident_points() {
        let mut t = QuadTree::new([0.0, 0.0], [1.0, 1.0]);
        assert!(!t.insert(0, [2.0, 0.5]) && !t.insert(1, [f64::NAN, 0.5]));
        for id in 0..50 {
            assert!(t.insert(id, [0.25, 0.25])); // coincident points must not recurse forever
        }
        assert_eq!(t.len(), 50);
        assert!(t.depth() <= 20);
        assert!(t.remove(7, [0.25, 0.25]) && !t.remove(7, [0.25, 0.25]));
        assert_eq!(t.query_radius([0.25, 0.25], 0.0).len(), 49);
        assert!(QuadTree::new([0.0; 2], [1.0; 2]).nearest([0.5, 0.5]).is_none());
    }
}
