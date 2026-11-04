//! Oct trees: point region trees in 3D.
//!
//! See the book entry [Spatial trees](https://jwkunz.github.io/lib_jsl/book/geometry/spatial-trees.html#quadtrees-and-octrees).

use super::region_tree::RegionTree;

/// A point oct tree over a box. Each leaf splits into eight octants when it overflows. See
/// [`RegionTree`] for the full interface (insert, remove, box/radius/nearest queries).
pub type OctTree = RegionTree<3>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_brute_force_in_3d() {
        let mut s = 17u64;
        let mut rnd = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let pts: Vec<[f64; 3]> = (0..500).map(|_| [rnd() * 5.0, rnd() * 5.0, rnd() * 5.0]).collect();
        let tree = OctTree::from_points(&pts);
        let d = |p: &[f64; 3], q: &[f64; 3]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
        let q = [2.5, 1.0, 3.5];
        let mut got = tree.query_radius(q, 1.5);
        got.sort_unstable();
        assert_eq!(got, (0..pts.len()).filter(|&i| d(&pts[i], &q) <= 1.5).collect::<Vec<_>>());
        let mut got = tree.query_box([1.0, 1.0, 1.0], [2.0, 4.0, 3.0]);
        got.sort_unstable();
        let want: Vec<usize> = (0..pts.len()).filter(|&i| (0..3).all(|k| pts[i][k] >= [1.0, 1.0, 1.0][k] && pts[i][k] <= [2.0, 4.0, 3.0][k])).collect();
        assert_eq!(got, want);
        let (id, dist) = tree.nearest(q).unwrap();
        let best = (0..pts.len()).min_by(|&a, &b| d(&pts[a], &q).total_cmp(&d(&pts[b], &q))).unwrap();
        assert_eq!(id, best);
        assert!((dist - d(&pts[best], &q)).abs() < 1e-12);
    }
}
