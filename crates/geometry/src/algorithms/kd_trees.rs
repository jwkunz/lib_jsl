//! k-d trees for nearest-neighbour and range queries in `D` dimensions.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A static, balanced k-d tree over points in `D` dimensions.
///
/// Built once in `O(n log n)` by recursive median splits (cycling through the axes), then queried
/// in roughly `O(log n)` for nearest neighbours. Queries return indices into the point slice the
/// tree was built from. Points with non-finite coordinates are not indexed.
#[derive(Debug, Clone)]
pub struct KdTree<const D: usize> {
    points: Vec<[f64; D]>,
    /// Point indices arranged so that for each subslice the median is the node and the halves are
    /// its subtrees.
    order: Vec<usize>,
}

#[derive(PartialEq)]
struct Candidate {
    dist2: f64,
    index: usize,
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist2.total_cmp(&other.dist2).then(self.index.cmp(&other.index))
    }
}

fn dist2<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    (0..D).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum()
}

impl<const D: usize> KdTree<D> {
    pub fn new(points: &[[f64; D]]) -> Self {
        let mut order: Vec<usize> = (0..points.len()).filter(|&i| points[i].iter().all(|v| v.is_finite())).collect();
        Self::build(points, &mut order, 0);
        KdTree { points: points.to_vec(), order }
    }

    fn build(points: &[[f64; D]], slice: &mut [usize], depth: usize) {
        if slice.len() <= 1 {
            return;
        }
        let axis = depth % D;
        let mid = slice.len() / 2;
        slice.select_nth_unstable_by(mid, |&a, &b| points[a][axis].total_cmp(&points[b][axis]));
        let (left, rest) = slice.split_at_mut(mid);
        Self::build(points, left, depth + 1);
        Self::build(points, &mut rest[1..], depth + 1);
    }

    /// Number of indexed points.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The closest point to `query` as `(index, distance)`; ties go to the lower index.
    pub fn nearest(&self, query: [f64; D]) -> Option<(usize, f64)> {
        self.k_nearest(query, 1).into_iter().next()
    }

    /// The `k` closest points, nearest first, as `(index, distance)`.
    pub fn k_nearest(&self, query: [f64; D], k: usize) -> Vec<(usize, f64)> {
        if k == 0 || self.is_empty() || query.iter().any(|v| !v.is_finite()) {
            return Vec::new();
        }
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
        self.search_k(&self.order, 0, &query, k, &mut heap);
        let mut out: Vec<Candidate> = heap.into_vec();
        out.sort();
        out.into_iter().map(|c| (c.index, c.dist2.sqrt())).collect()
    }

    fn search_k(&self, slice: &[usize], depth: usize, q: &[f64; D], k: usize, heap: &mut BinaryHeap<Candidate>) {
        if slice.is_empty() {
            return;
        }
        let axis = depth % D;
        let mid = slice.len() / 2;
        let node = slice[mid];
        let cand = Candidate { dist2: dist2(&self.points[node], q), index: node };
        if heap.len() < k {
            heap.push(cand);
        } else if heap.peek().is_some_and(|worst| cand < *worst) {
            heap.pop();
            heap.push(cand);
        }
        let diff = q[axis] - self.points[node][axis];
        let (near, far) = if diff < 0.0 { (&slice[..mid], &slice[mid + 1..]) } else { (&slice[mid + 1..], &slice[..mid]) };
        self.search_k(near, depth + 1, q, k, heap);
        // The far side can only help if the splitting plane is closer than the current k-th best.
        if heap.len() < k || heap.peek().is_some_and(|worst| diff * diff <= worst.dist2) {
            self.search_k(far, depth + 1, q, k, heap);
        }
    }

    /// Indices of all points within `radius` of `query` (inclusive), in unspecified order.
    pub fn within_radius(&self, query: [f64; D], radius: f64) -> Vec<usize> {
        let mut out = Vec::new();
        if radius >= 0.0 {
            self.search_radius(&self.order, 0, &query, radius * radius, &mut out);
        }
        out
    }

    fn search_radius(&self, slice: &[usize], depth: usize, q: &[f64; D], r2: f64, out: &mut Vec<usize>) {
        if slice.is_empty() {
            return;
        }
        let axis = depth % D;
        let mid = slice.len() / 2;
        let node = slice[mid];
        if dist2(&self.points[node], q) <= r2 {
            out.push(node);
        }
        let diff = q[axis] - self.points[node][axis];
        if diff <= 0.0 || diff * diff <= r2 {
            self.search_radius(&slice[..mid], depth + 1, q, r2, out);
        }
        if diff >= 0.0 || diff * diff <= r2 {
            self.search_radius(&slice[mid + 1..], depth + 1, q, r2, out);
        }
    }

    /// Indices of all points inside the axis-aligned box `[min, max]` (inclusive).
    pub fn range_query(&self, min: [f64; D], max: [f64; D]) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_box(&self.order, 0, &min, &max, &mut out);
        out
    }

    fn search_box(&self, slice: &[usize], depth: usize, min: &[f64; D], max: &[f64; D], out: &mut Vec<usize>) {
        if slice.is_empty() {
            return;
        }
        let axis = depth % D;
        let mid = slice.len() / 2;
        let node = slice[mid];
        let p = &self.points[node];
        if (0..D).all(|k| p[k] >= min[k] && p[k] <= max[k]) {
            out.push(node);
        }
        if min[axis] <= p[axis] {
            self.search_box(&slice[..mid], depth + 1, min, max, out);
        }
        if max[axis] >= p[axis] {
            self.search_box(&slice[mid + 1..], depth + 1, min, max, out);
        }
    }
}

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

    fn brute_k<const D: usize>(pts: &[[f64; D]], q: [f64; D], k: usize) -> Vec<(usize, f64)> {
        let mut all: Vec<(usize, f64)> = pts.iter().enumerate().map(|(i, p)| (i, dist2(p, &q).sqrt())).collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        all.truncate(k);
        all
    }

    #[test]
    fn matches_brute_force_in_2d_and_3d_and_5d() {
        let mut rnd = lcg(3);
        let p2: Vec<[f64; 2]> = (0..400).map(|_| [rnd(), rnd()]).collect();
        let p3: Vec<[f64; 3]> = (0..400).map(|_| [rnd(), rnd(), rnd()]).collect();
        let p5: Vec<[f64; 5]> = (0..300).map(|_| [rnd(), rnd(), rnd(), rnd(), rnd()]).collect();
        let (t2, t3, t5) = (KdTree::new(&p2), KdTree::new(&p3), KdTree::new(&p5));
        for _ in 0..40 {
            let q2 = [rnd() * 1.2 - 0.1, rnd()];
            assert_eq!(t2.k_nearest(q2, 7), brute_k(&p2, q2, 7));
            let q3 = [rnd(), rnd(), rnd()];
            assert_eq!(t3.nearest(q3).unwrap(), brute_k(&p3, q3, 1)[0]);
            let q5 = [rnd(), rnd(), rnd(), rnd(), rnd()];
            assert_eq!(t5.k_nearest(q5, 3), brute_k(&p5, q5, 3));
        }
    }

    #[test]
    fn radius_and_box_queries() {
        let mut rnd = lcg(11);
        let pts: Vec<[f64; 3]> = (0..500).map(|_| [rnd() * 10.0, rnd() * 10.0, rnd() * 10.0]).collect();
        let tree = KdTree::new(&pts);
        let q = [5.0, 4.0, 6.0];
        let mut got = tree.within_radius(q, 2.5);
        got.sort_unstable();
        let want: Vec<usize> = (0..pts.len()).filter(|&i| dist2(&pts[i], &q).sqrt() <= 2.5).collect();
        assert_eq!(got, want);
        let (lo, hi) = ([2.0, 3.0, 1.0], [6.0, 7.0, 4.0]);
        let mut got = tree.range_query(lo, hi);
        got.sort_unstable();
        let want: Vec<usize> = (0..pts.len()).filter(|&i| (0..3).all(|k| pts[i][k] >= lo[k] && pts[i][k] <= hi[k])).collect();
        assert_eq!(got, want);
        assert!(tree.within_radius(q, -1.0).is_empty());
    }

    #[test]
    fn edge_cases() {
        let empty: KdTree<2> = KdTree::new(&[]);
        assert!(empty.is_empty() && empty.nearest([0.0, 0.0]).is_none());
        let pts = [[1.0, 1.0], [f64::NAN, 0.0], [1.0, 1.0], [3.0, 3.0]];
        let t = KdTree::new(&pts);
        assert_eq!(t.len(), 3); // the NaN point is not indexed
        assert_eq!(t.nearest([1.1, 1.0]).unwrap().0, 0); // duplicate: lower index wins
        assert_eq!(t.k_nearest([0.0, 0.0], 10).len(), 3);
        assert!(t.k_nearest([0.0, 0.0], 0).is_empty() && t.k_nearest([f64::NAN, 0.0], 1).is_empty());
    }
}
