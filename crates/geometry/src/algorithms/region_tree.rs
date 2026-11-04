//! A point region tree: the shared engine behind quad trees (`D = 2`) and oct trees (`D = 3`).
//!
//! See the book entry [Spatial trees](https://jwkunz.github.io/lib_jsl/book/geometry/spatial-trees.html#quadtrees-and-octrees).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]

struct Node<const D: usize> {
    min: [f64; D],
    max: [f64; D],
    entries: Vec<(usize, [f64; D])>,
    children: Vec<Node<D>>,
}

#[derive(PartialEq)]

struct Candidate {
    dist2: f64,
    id: usize,
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist2.total_cmp(&other.dist2).then(self.id.cmp(&other.id))
    }
}

fn point_dist2<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    (0..D).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum()
}

impl<const D: usize> Node<D> {
    fn new(min: [f64; D], max: [f64; D]) -> Self {
        Node { min, max, entries: Vec::new(), children: Vec::new() }
    }

    fn centre(&self) -> [f64; D] {
        std::array::from_fn(|k| 0.5 * (self.min[k] + self.max[k]))
    }

    fn child_index(&self, p: &[f64; D]) -> usize {
        let c = self.centre();
        (0..D).fold(0, |acc, k| acc | (usize::from(p[k] >= c[k]) << k))
    }

    /// Squared distance from `q` to the closest point of this node's box.
    fn box_dist2(&self, q: &[f64; D]) -> f64 {
        (0..D)
            .map(|k| {
                let d = (self.min[k] - q[k]).max(q[k] - self.max[k]).max(0.0);
                d * d
            })
            .sum()
    }

    fn split(&mut self) {
        let c = self.centre();
        self.children = (0..1usize << D)
            .map(|i| {
                let min = std::array::from_fn(|k| if i >> k & 1 == 1 { c[k] } else { self.min[k] });
                let max = std::array::from_fn(|k| if i >> k & 1 == 1 { self.max[k] } else { c[k] });
                Node::new(min, max)
            })
            .collect();
        for (id, p) in std::mem::take(&mut self.entries) {
            let ci = self.child_index(&p);
            self.children[ci].entries.push((id, p));
        }
    }

    fn insert(&mut self, id: usize, p: [f64; D], capacity: usize, depth: usize, max_depth: usize) {
        if self.children.is_empty() {
            self.entries.push((id, p));
            if self.entries.len() > capacity && depth < max_depth {
                self.split();
                // Redistribution may overfill a child (clustered points); split those further.
                for child in &mut self.children {
                    child.resplit(capacity, depth + 1, max_depth);
                }
            }
        } else {
            let ci = self.child_index(&p);
            self.children[ci].insert(id, p, capacity, depth + 1, max_depth);
        }
    }

    fn resplit(&mut self, capacity: usize, depth: usize, max_depth: usize) {
        if self.entries.len() > capacity && depth < max_depth {
            self.split();
            for child in &mut self.children {
                child.resplit(capacity, depth + 1, max_depth);
            }
        }
    }

    fn remove(&mut self, id: usize, p: &[f64; D]) -> bool {
        if self.children.is_empty() {
            if let Some(pos) = self.entries.iter().position(|(i, q)| *i == id && q == p) {
                self.entries.swap_remove(pos);
                return true;
            }
            return false;
        }
        let ci = self.child_index(p);
        let removed = self.children[ci].remove(id, p);
        // Collapse if the whole subtree has become small enough to be one leaf again.
        if removed && self.children.iter().all(|c| c.children.is_empty()) && self.children.iter().map(|c| c.entries.len()).sum::<usize>() == 0 {
            self.children.clear();
        }
        removed
    }

    fn query_box(&self, min: &[f64; D], max: &[f64; D], out: &mut Vec<usize>) {
        if (0..D).any(|k| self.max[k] < min[k] || self.min[k] > max[k]) {
            return;
        }
        out.extend(self.entries.iter().filter(|(_, p)| (0..D).all(|k| p[k] >= min[k] && p[k] <= max[k])).map(|(id, _)| *id));
        self.children.iter().for_each(|c| c.query_box(min, max, out));
    }

    fn query_radius(&self, q: &[f64; D], r2: f64, out: &mut Vec<usize>) {
        if self.box_dist2(q) > r2 {
            return;
        }
        out.extend(self.entries.iter().filter(|(_, p)| point_dist2(p, q) <= r2).map(|(id, _)| *id));
        self.children.iter().for_each(|c| c.query_radius(q, r2, out));
    }

    fn k_nearest(&self, q: &[f64; D], k: usize, heap: &mut BinaryHeap<Candidate>) {
        if heap.len() >= k && heap.peek().is_some_and(|w| self.box_dist2(q) > w.dist2) {
            return;
        }
        for (id, p) in &self.entries {
            let cand = Candidate { dist2: point_dist2(p, q), id: *id };
            if heap.len() < k {
                heap.push(cand);
            } else if heap.peek().is_some_and(|w| cand < *w) {
                heap.pop();
                heap.push(cand);
            }
        }
        // Visit children nearest-first so pruning kicks in early.
        let mut order: Vec<&Node<D>> = self.children.iter().collect();
        order.sort_by(|a, b| a.box_dist2(q).total_cmp(&b.box_dist2(q)));
        for child in order {
            child.k_nearest(q, k, heap);
        }
    }
}

/// A point tree over an axis-aligned region of `D`-space. Each leaf holds up to `capacity` points
/// before splitting into `2^D` equal sub-boxes (down to `max_depth` levels, so coincident points
/// cannot cause unbounded splitting). Entries are `(id, point)` pairs; ids are chosen by the caller.
///
/// See [`QuadTree`](super::quad_trees::QuadTree) (`D = 2`) and [`OctTree`](super::oct_trees::OctTree)
/// (`D = 3`).
#[derive(Debug, Clone)]

pub struct RegionTree<const D: usize> {
    root: Node<D>,
    capacity: usize,
    max_depth: usize,
    count: usize,
}

impl<const D: usize> RegionTree<D> {
    /// An empty tree covering `[min, max]` with leaf capacity 8 and depth limit 20.
    pub fn new(min: [f64; D], max: [f64; D]) -> Self {
        Self::with_limits(min, max, 8, 20)
    }

    pub fn with_limits(min: [f64; D], max: [f64; D], capacity: usize, max_depth: usize) -> Self {
        RegionTree { root: Node::new(min, max), capacity: capacity.max(1), max_depth, count: 0 }
    }

    /// Builds a tree sized to the bounding box of `points`, using each point's index as its id.
    pub fn from_points(points: &[[f64; D]]) -> Self {
        let (mut lo, mut hi) = ([f64::INFINITY; D], [f64::NEG_INFINITY; D]);
        for p in points.iter().filter(|p| p.iter().all(|v| v.is_finite())) {
            for k in 0..D {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        if lo[0] > hi[0] {
            (lo, hi) = ([0.0; D], [1.0; D]);
        }
        let mut tree = Self::new(lo, hi);
        for (i, &p) in points.iter().enumerate() {
            tree.insert(i, p);
        }
        tree
    }

    /// Adds a point; returns `false` (and stores nothing) if it is non-finite or outside the region.
    pub fn insert(&mut self, id: usize, point: [f64; D]) -> bool {
        let r = &self.root;
        if !(0..D).all(|k| point[k].is_finite() && point[k] >= r.min[k] && point[k] <= r.max[k]) {
            return false;
        }
        self.root.insert(id, point, self.capacity, 0, self.max_depth);
        self.count += 1;
        true
    }

    /// Removes the entry with this `id` and `point`; returns whether it was present.
    pub fn remove(&mut self, id: usize, point: [f64; D]) -> bool {
        let removed = self.root.remove(id, &point);
        if removed {
            self.count -= 1;
        }
        removed
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Ids of all points in the box `[min, max]` (inclusive).
    pub fn query_box(&self, min: [f64; D], max: [f64; D]) -> Vec<usize> {
        let mut out = Vec::new();
        self.root.query_box(&min, &max, &mut out);
        out
    }

    /// Ids of all points within `radius` of `centre` (inclusive).
    pub fn query_radius(&self, centre: [f64; D], radius: f64) -> Vec<usize> {
        let mut out = Vec::new();
        if radius >= 0.0 {
            self.root.query_radius(&centre, radius * radius, &mut out);
        }
        out
    }

    /// The `k` nearest points as `(id, distance)`, nearest first (ties by id).
    pub fn k_nearest(&self, query: [f64; D], k: usize) -> Vec<(usize, f64)> {
        if k == 0 || self.is_empty() || query.iter().any(|v| !v.is_finite()) {
            return Vec::new();
        }
        let mut heap = BinaryHeap::with_capacity(k + 1);
        self.root.k_nearest(&query, k, &mut heap);
        let mut v = heap.into_vec();
        v.sort();
        v.into_iter().map(|c| (c.id, c.dist2.sqrt())).collect()
    }

    pub fn nearest(&self, query: [f64; D]) -> Option<(usize, f64)> {
        self.k_nearest(query, 1).into_iter().next()
    }

    /// Deepest level reached by any leaf (0 for a single leaf), for diagnostics.
    pub fn depth(&self) -> usize {
        fn go<const D: usize>(n: &Node<D>) -> usize {
            n.children.iter().map(|c| 1 + go(c)).max().unwrap_or(0)
        }
        go(&self.root)
    }
}
