//! Convex hulls, Delaunay triangulations and Voronoi diagrams in any dimension, half-space
//! intersections and Voronoi diagrams on the sphere, following `scipy.spatial` (which wraps
//! Qhull): [`ConvexHullNd`], [`DelaunayNd`], [`VoronoiNd`], [`HalfspaceIntersection`] and
//! [`SphericalVoronoi`].
//!
//! Everything rests on one n-dimensional Quickhull (Barber, Dobkin and Huhdanpaa, 1996), the
//! algorithm inside Qhull: start from a simplex, and repeatedly take the point farthest outside
//! some facet, delete the facets it can see, and join it to the horizon. Delaunay triangulations
//! are lower hulls of the points lifted to the paraboloid `x_{d+1} = ‖x‖²`, Voronoi diagrams
//! their duals, and half-space intersections hulls of the dual points.
//!
//! See the book entry [Geometry in any dimension](https://jwkunz.github.io/lib_jsl/book/geometry/hulls-nd.html).

#![allow(clippy::needless_range_loop)]

use lib_jsl_core::ErrorsJSL;
use std::collections::HashMap;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// The determinant of a square matrix by Gaussian elimination with partial pivoting.
fn det(mut m: Vec<Vec<f64>>) -> f64 {
    let n = m.len();
    let mut d = 1.0;
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| m[i][k].abs().total_cmp(&m[j][k].abs())).unwrap();
        if m[p][k] == 0.0 {
            return 0.0;
        }
        if p != k {
            m.swap(p, k);
            d = -d;
        }
        d *= m[k][k];
        for i in k + 1..n {
            let f = m[i][k] / m[k][k];
            for j in k..n {
                m[i][j] -= f * m[k][j];
            }
        }
    }
    d
}

/// A unit vector orthogonal to the `d − 1` rows of `rows` (each of length `d`): the last column
/// of the Householder QR factor of `rowsᵀ`.
fn normal_of(rows: &[Vec<f64>], d: usize) -> Vec<f64> {
    let k = rows.len();
    // Columns of A = rowsᵀ (d × k); accumulate Householder reflectors H_j = I − 2 v vᵀ.
    let mut a: Vec<Vec<f64>> = rows.to_vec(); // a[j] is column j
    let mut vs: Vec<Vec<f64>> = Vec::new();
    for j in 0..k {
        let mut x = a[j].clone();
        for v in &vs {
            let c = 2.0 * dot(v, &x);
            x.iter_mut().zip(v).for_each(|(xi, vi)| *xi -= c * vi);
        }
        let tail_norm = x[j..].iter().map(|t| t * t).sum::<f64>().sqrt();
        let mut v = vec![0.0; d];
        if tail_norm > 0.0 {
            let alpha = if x[j] > 0.0 { -tail_norm } else { tail_norm };
            v[j..].copy_from_slice(&x[j..]);
            v[j] -= alpha;
            let vn = v.iter().map(|t| t * t).sum::<f64>().sqrt();
            if vn > 0.0 {
                v.iter_mut().for_each(|t| *t /= vn);
            }
        }
        vs.push(v);
        a[j] = x;
    }
    // Q e_d = H_0 H_1 … H_{k−1} e_d.
    let mut e = vec![0.0; d];
    e[d - 1] = 1.0;
    for v in vs.iter().rev() {
        let c = 2.0 * dot(v, &e);
        e.iter_mut().zip(v).for_each(|(ei, vi)| *ei -= c * vi);
    }
    e
}

/// One hull facet: `d` vertices, the outward unit normal `n` and offset `b` with
/// `n·x + b = 0` on the facet and `< 0` inside; `neighbors[i]` is the facet across the ridge
/// opposite `vertices[i]`.
#[derive(Clone, Debug)]
struct Facet {
    vertices: Vec<usize>,
    normal: Vec<f64>,
    offset: f64,
    neighbors: Vec<usize>,
    outside: Vec<usize>,
    alive: bool,
}

/// The raw Quickhull result: facets as vertex lists with their hyperplanes and neighbours.
struct Hull {
    facets: Vec<Facet>,
}

fn make_facet(points: &[Vec<f64>], vertices: Vec<usize>, interior: &[f64]) -> Facet {
    let d = interior.len();
    let p0 = &points[vertices[0]];
    let rows: Vec<Vec<f64>> = vertices[1..].iter().map(|&v| sub(&points[v], p0)).collect();
    let mut normal = normal_of(&rows, d);
    let mut offset = -dot(&normal, p0);
    if dot(&normal, interior) + offset > 0.0 {
        normal.iter_mut().for_each(|t| *t = -*t);
        offset = -offset;
    }
    Facet { vertices, normal, offset, neighbors: Vec::new(), outside: Vec::new(), alive: true }
}

// ANCHOR: quickhull
/// Quickhull in `d ≥ 2` dimensions. `eps` is the distance below which a point counts as on a
/// facet; such points are not added (Qhull's default "coplanar points are inside").
fn quickhull(points: &[Vec<f64>], eps: f64) -> Result<Hull, ErrorsJSL> {
    let n = points.len();
    let d = points.first().map_or(0, |p| p.len());
    if d < 2 || n < d + 1 {
        return Err(ErrorsJSL::InvalidInputRange("Quickhull needs d ≥ 2 and at least d + 1 points"));
    }
    // Initial simplex: the extreme pair along the first coordinate, then repeatedly the point
    // farthest from the affine hull of those chosen (Gram-Schmidt on the differences).
    let lo = (0..n).min_by(|&a, &b| points[a][0].total_cmp(&points[b][0])).unwrap();
    let hi = (0..n).max_by(|&a, &b| points[a][0].total_cmp(&points[b][0])).unwrap();
    let mut simplex = vec![lo];
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut candidate = Some(hi);
    while simplex.len() < d + 1 {
        let next = match candidate.take() {
            Some(c) if c != simplex[0] => c,
            _ => (0..n)
                .max_by(|&a, &b| {
                    let r = |i: usize| {
                        let mut v = sub(&points[i], &points[simplex[0]]);
                        for q in &basis {
                            let c = dot(q, &v);
                            v.iter_mut().zip(q).for_each(|(x, y)| *x -= c * y);
                        }
                        dot(&v, &v)
                    };
                    r(a).total_cmp(&r(b))
                })
                .unwrap(),
        };
        let mut v = sub(&points[next], &points[simplex[0]]);
        for q in &basis {
            let c = dot(q, &v);
            v.iter_mut().zip(q).for_each(|(x, y)| *x -= c * y);
        }
        let vn = dot(&v, &v).sqrt();
        if vn <= eps.max(f64::MIN_POSITIVE) {
            return Err(ErrorsJSL::InvalidInputRange("the points lie in a lower-dimensional subspace (the initial simplex is flat)"));
        }
        basis.push(v.iter().map(|t| t / vn).collect());
        simplex.push(next);
    }
    let interior: Vec<f64> = (0..d).map(|k| simplex.iter().map(|&i| points[i][k]).sum::<f64>() / (d + 1) as f64).collect();
    let mut facets: Vec<Facet> = (0..=d)
        .map(|skip| make_facet(points, simplex.iter().enumerate().filter(|(i, _)| *i != skip).map(|(_, &v)| v).collect(), &interior))
        .collect();
    // Facet i omits simplex vertex i; the facet across the ridge opposite vertex v of facet i is
    // the facet that omits v.
    for i in 0..=d {
        let verts = facets[i].vertices.clone();
        facets[i].neighbors = verts.iter().map(|v| simplex.iter().position(|s| s == v).unwrap()).collect();
    }
    let in_simplex: Vec<bool> = (0..n).map(|i| simplex.contains(&i)).collect();
    for p in 0..n {
        if in_simplex[p] {
            continue;
        }
        if let Some((f, _)) = facets.iter().enumerate().map(|(f, fa)| (f, dot(&fa.normal, &points[p]) + fa.offset)).filter(|(_, dist)| *dist > eps).max_by(|a, b| a.1.total_cmp(&b.1)) {
            facets[f].outside.push(p);
        }
    }
    while let Some(start) = facets.iter().position(|f| f.alive && !f.outside.is_empty()) {
        let fs = &facets[start];
        let apex = *fs.outside.iter().max_by(|&&a, &&b| (dot(&fs.normal, &points[a]) + fs.offset).total_cmp(&(dot(&fs.normal, &points[b]) + fs.offset))).unwrap();
        // Visible facets by a search over neighbours.
        let mut visible = vec![start];
        let mut seen: HashMap<usize, bool> = HashMap::from([(start, true)]);
        let mut k = 0;
        while k < visible.len() {
            let f = visible[k];
            k += 1;
            for &g in &facets[f].neighbors.clone() {
                if seen.contains_key(&g) {
                    continue;
                }
                let vis = dot(&facets[g].normal, &points[apex]) + facets[g].offset > eps;
                seen.insert(g, vis);
                if vis {
                    visible.push(g);
                }
            }
        }
        // Horizon ridges: (visible facet, index of the opposite vertex) with an invisible neighbour.
        let mut new_ids = Vec::new();
        let mut ridge_map: HashMap<Vec<usize>, (usize, usize)> = HashMap::new();
        for &f in &visible {
            for i in 0..d {
                let g = facets[f].neighbors[i];
                if seen.get(&g).copied().unwrap_or(false) {
                    continue;
                }
                let mut verts: Vec<usize> = facets[f].vertices.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, &v)| v).collect();
                verts.push(apex);
                let mut nf = make_facet(points, verts, &interior);
                let id = facets.len();
                // Neighbour opposite the apex (the last vertex) is g; repoint g's link from f.
                nf.neighbors = vec![usize::MAX; d];
                nf.neighbors[d - 1] = g;
                if let Some(slot) = facets[g].neighbors.iter().position(|&h| h == f) {
                    facets[g].neighbors[slot] = id;
                }
                // Ridges through the apex, keyed by the sorted vertex set without one vertex.
                for j in 0..d - 1 {
                    let mut key: Vec<usize> = nf.vertices.iter().enumerate().filter(|(m, _)| *m != j).map(|(_, &v)| v).collect();
                    key.sort_unstable();
                    if let Some(&(other, oj)) = ridge_map.get(&key) {
                        nf.neighbors[j] = other;
                        facets[other].neighbors[oj] = id;
                    } else {
                        ridge_map.insert(key, (id, j));
                    }
                }
                facets.push(nf);
                new_ids.push(id);
            }
        }
        // Reassign the outside points of the deleted facets.
        let mut orphans = Vec::new();
        for &f in &visible {
            facets[f].alive = false;
            orphans.append(&mut facets[f].outside);
        }
        for p in orphans {
            if p == apex {
                continue;
            }
            if let Some((f, _)) = new_ids.iter().map(|&f| (f, dot(&facets[f].normal, &points[p]) + facets[f].offset)).filter(|(_, dist)| *dist > eps).max_by(|a, b| a.1.total_cmp(&b.1)) {
                facets[f].outside.push(p);
            }
        }
    }
    Ok(Hull { facets })
}
// ANCHOR_END: quickhull

/// A tolerance for "on the facet" from the size of the coordinates (Qhull's is similar).
fn default_eps(points: &[Vec<f64>]) -> f64 {
    let d = points.first().map_or(1, |p| p.len());
    let m = points.iter().flat_map(|p| p.iter()).fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    let spread = (0..d).map(|k| points.iter().map(|p| p[k]).fold(f64::NEG_INFINITY, f64::max) - points.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min)).fold(0.0f64, f64::max);
    1e2 * (d as f64) * f64::EPSILON * m.max(spread)
}

fn factorial(k: usize) -> f64 {
    (1..=k).map(|i| i as f64).product()
}

/// The `(d − 1)`-volume of a simplex with `d` vertices in `ℝ^d`, from the Gram determinant.
fn facet_area(points: &[Vec<f64>], verts: &[usize]) -> f64 {
    let p0 = &points[verts[0]];
    let e: Vec<Vec<f64>> = verts[1..].iter().map(|&v| sub(&points[v], p0)).collect();
    let g: Vec<Vec<f64>> = e.iter().map(|a| e.iter().map(|b| dot(a, b)).collect()).collect();
    det(g).max(0.0).sqrt() / factorial(e.len())
}

// ANCHOR: convex_hull_nd
/// The convex hull of points in any dimension `d ≥ 2` (SciPy's `ConvexHull`), with simplicial
/// facets. Points within a small tolerance of a facet are treated as inside.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ConvexHullNd {
    pub points: Vec<Vec<f64>>,
    /// Each facet as `d` point indices (SciPy's `simplices`).
    pub simplices: Vec<Vec<usize>>,
    /// Each facet's hyperplane `[n, b]` with outward unit normal `n` and `n·x + b ≤ 0` inside
    /// (SciPy's `equations`).
    pub equations: Vec<Vec<f64>>,
    /// For each facet, the facets across the ridges opposite each of its vertices.
    pub neighbors: Vec<Vec<usize>>,
    /// The hull's vertices, sorted (SciPy's `vertices` is in counter-clockwise order in 2-D).
    pub vertices: Vec<usize>,
    /// The `d`-volume enclosed and the `(d − 1)`-volume of the boundary.
    pub volume: f64,
    pub area: f64,
}

impl ConvexHullNd {
    pub fn new(points: &[Vec<f64>]) -> Result<ConvexHullNd, ErrorsJSL> {
        let d = points.first().map_or(0, |p| p.len());
        if points.iter().any(|p| p.len() != d || p.iter().any(|v| !v.is_finite())) {
            return Err(ErrorsJSL::InvalidInputRange("points must be finite and of one dimension"));
        }
        let hull = quickhull(points, default_eps(points))?;
        let alive: Vec<usize> = (0..hull.facets.len()).filter(|&i| hull.facets[i].alive).collect();
        let index: HashMap<usize, usize> = alive.iter().enumerate().map(|(k, &i)| (i, k)).collect();
        let simplices: Vec<Vec<usize>> = alive.iter().map(|&i| hull.facets[i].vertices.clone()).collect();
        let equations = alive.iter().map(|&i| {
            let mut e = hull.facets[i].normal.clone();
            e.push(hull.facets[i].offset);
            e
        }).collect();
        let neighbors = alive.iter().map(|&i| hull.facets[i].neighbors.iter().map(|g| index[g]).collect()).collect();
        let mut vertices: Vec<usize> = simplices.iter().flatten().copied().collect();
        vertices.sort_unstable();
        vertices.dedup();
        // Volume from a cone over each facet with apex at the vertex centroid.
        let c: Vec<f64> = (0..d).map(|k| vertices.iter().map(|&v| points[v][k]).sum::<f64>() / vertices.len() as f64).collect();
        let mut volume = 0.0;
        let mut area = 0.0;
        for (s, &i) in simplices.iter().zip(&alive) {
            let a = facet_area(points, s);
            area += a;
            let f = &hull.facets[i];
            volume += a * (-(dot(&f.normal, &c) + f.offset)) / d as f64;
        }
        Ok(ConvexHullNd { points: points.to_vec(), simplices, equations, neighbors, vertices, volume, area })
    }
}
// ANCHOR_END: convex_hull_nd

// ANCHOR: delaunay_nd
/// The Delaunay triangulation of points in any dimension `d ≥ 2` (SciPy's `Delaunay`): the
/// lower facets of the hull of the points lifted to `(x, ‖x‖²)`. Degenerate (cospherical) sets,
/// such as grids, are triangulated by a tiny deterministic perturbation of the lifted
/// coordinate, which picks one of their valid triangulations (Qhull's "joggle").
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DelaunayNd {
    pub points: Vec<Vec<f64>>,
    /// Each simplex as `d + 1` point indices.
    pub simplices: Vec<Vec<usize>>,
    /// `neighbors[s][i]`: the simplex opposite vertex `i` of simplex `s`, or `None` on the hull.
    pub neighbors: Vec<Vec<Option<usize>>>,
    /// For each simplex, the barycentric transform: rows `0..d` hold `T⁻¹` and row `d` the
    /// vertex `r = x_{v_d}`, so that `c = T⁻¹ (x − r)` gives the first `d` barycentric
    /// coordinates (SciPy's `transform`).
    pub transform: Vec<Vec<Vec<f64>>>,
}

fn invert(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut a: Vec<Vec<f64>> = m.iter().enumerate().map(|(i, r)| {
        let mut r = r.clone();
        r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
        r
    }).collect();
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if a[p][k].abs() < 1e-300 {
            return None;
        }
        a.swap(k, p);
        let piv = a[k][k];
        a[k].iter_mut().for_each(|v| *v /= piv);
        for i in 0..n {
            if i != k {
                let f = a[i][k];
                if f != 0.0 {
                    let rk = a[k].clone();
                    a[i].iter_mut().zip(&rk).for_each(|(x, y)| *x -= f * y);
                }
            }
        }
    }
    Some(a.into_iter().map(|r| r[n..].to_vec()).collect())
}

impl DelaunayNd {
    pub fn new(points: &[Vec<f64>]) -> Result<DelaunayNd, ErrorsJSL> {
        let d = points.first().map_or(0, |p| p.len());
        if d < 1 || points.iter().any(|p| p.len() != d || p.iter().any(|v| !v.is_finite())) {
            return Err(ErrorsJSL::InvalidInputRange("points must be finite and of one dimension"));
        }
        // Scale to a unit box (Qhull's Qbb) and lift, with a tiny deterministic joggle.
        let lo: Vec<f64> = (0..d).map(|k| points.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min)).collect();
        let hi: Vec<f64> = (0..d).map(|k| points.iter().map(|p| p[k]).fold(f64::NEG_INFINITY, f64::max)).collect();
        let scale = (0..d).map(|k| hi[k] - lo[k]).fold(0.0f64, f64::max).max(f64::MIN_POSITIVE);
        let mut state = 0x9E3779B97F4A7C15u64;
        let lifted: Vec<Vec<f64>> = points
            .iter()
            .map(|p| {
                let mut q: Vec<f64> = p.iter().zip(&lo).map(|(x, l)| (x - l) / scale).collect();
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let jog = ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 1e-11;
                let r2: f64 = q.iter().map(|v| v * v).sum();
                q.push(r2 + jog);
                q
            })
            .collect();
        // A point far above the paraboloid closes the hull (Qhull's Qz) and keeps upper facets
        // separate from lower ones.
        let mut with_top = lifted.clone();
        let mut top: Vec<f64> = vec![0.5; d];
        top.push(1e3 * (d as f64 + 1.0));
        with_top.push(top);
        let hull = quickhull(&with_top, 1e2 * (d as f64 + 1.0) * f64::EPSILON * 4.0)?;
        let alive: Vec<usize> = (0..hull.facets.len()).filter(|&i| hull.facets[i].alive && hull.facets[i].normal[d] < -1e-6 && !hull.facets[i].vertices.contains(&points.len())).collect();
        let index: HashMap<usize, usize> = alive.iter().enumerate().map(|(k, &i)| (i, k)).collect();
        let simplices: Vec<Vec<usize>> = alive.iter().map(|&i| hull.facets[i].vertices.clone()).collect();
        let neighbors = alive.iter().map(|&i| hull.facets[i].neighbors.iter().map(|g| index.get(g).copied()).collect()).collect();
        let transform = simplices
            .iter()
            .map(|s| {
                let r = &points[s[d]];
                let t: Vec<Vec<f64>> = (0..d).map(|row| (0..d).map(|col| points[s[col]][row] - r[row]).collect()).collect();
                let mut out = invert(&t).unwrap_or_else(|| vec![vec![f64::NAN; d]; d]);
                out.push(r.clone());
                out
            })
            .collect();
        Ok(DelaunayNd { points: points.to_vec(), simplices, neighbors, transform })
    }

    /// The barycentric coordinates of `x` in simplex `s` (`d + 1` values summing to 1).
    pub fn barycentric(&self, s: usize, x: &[f64]) -> Vec<f64> {
        let t = &self.transform[s];
        let d = x.len();
        let dx = sub(x, &t[d]);
        let mut c: Vec<f64> = (0..d).map(|i| dot(&t[i], &dx)).collect();
        let last = 1.0 - c.iter().sum::<f64>();
        c.push(last);
        c
    }

    /// The simplex containing `x`, or `None` outside the triangulation (SciPy's `find_simplex`),
    /// by a walk from simplex 0 towards `x`, falling back to a full search.
    pub fn find_simplex(&self, x: &[f64]) -> Option<usize> {
        let tol = 1e2 * f64::EPSILON;
        if self.simplices.is_empty() {
            return None;
        }
        let mut s = 0;
        for _ in 0..4 * self.simplices.len() + 10 {
            let c = self.barycentric(s, x);
            let (i, m) = c.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).map(|(i, m)| (i, *m)).unwrap();
            if m >= -tol {
                return Some(s);
            }
            match self.neighbors[s][i] {
                Some(t) => s = t,
                None => break,
            }
        }
        (0..self.simplices.len()).find(|&s| self.barycentric(s, x).iter().all(|&c| c >= -tol))
    }
}
// ANCHOR_END: delaunay_nd

/// The circumcentre of a simplex with `d + 1` vertices in `ℝ^d`.
fn circumcenter(pts: &[&Vec<f64>]) -> Option<Vec<f64>> {
    let d = pts[0].len();
    let p0 = pts[0];
    let a: Vec<Vec<f64>> = pts[1..].iter().map(|p| sub(p, p0).iter().map(|v| 2.0 * v).collect()).collect();
    let b: Vec<f64> = pts[1..].iter().map(|p| dot(p, p) - dot(p0, p0)).collect();
    let inv = invert(&a)?;
    Some((0..d).map(|i| dot(&inv[i], &b)).collect())
}

// ANCHOR: voronoi_nd
/// The Voronoi diagram of points in any dimension (SciPy's `Voronoi`), as the dual of the
/// Delaunay triangulation: one Voronoi vertex per Delaunay simplex (its circumcentre), one
/// region per input point, and one ridge per Delaunay edge (pair of neighbouring points).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct VoronoiNd {
    pub points: Vec<Vec<f64>>,
    pub vertices: Vec<Vec<f64>>,
    /// The pairs of input points that each ridge separates.
    pub ridge_points: Vec<[usize; 2]>,
    /// The Voronoi vertices of each ridge, with `None` for a vertex at infinity (SciPy's −1).
    pub ridge_vertices: Vec<Vec<Option<usize>>>,
    /// For each input point, the vertices of its region (unordered), with `None` if unbounded.
    pub regions: Vec<Vec<Option<usize>>>,
}

impl VoronoiNd {
    pub fn new(points: &[Vec<f64>]) -> Result<VoronoiNd, ErrorsJSL> {
        let tri = DelaunayNd::new(points)?;
        let vertices: Vec<Vec<f64>> = tri.simplices.iter().map(|s| circumcenter(&s.iter().map(|&i| &points[i]).collect::<Vec<_>>()).unwrap_or_else(|| vec![f64::NAN; points[0].len()])).collect();
        // Ridges: point pairs sharing a Delaunay edge, with the simplices around that edge.
        let mut ridges: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
        let mut on_hull: HashMap<[usize; 2], bool> = HashMap::new();
        for (si, s) in tri.simplices.iter().enumerate() {
            for a in 0..s.len() {
                for b in a + 1..s.len() {
                    let key = [s[a].min(s[b]), s[a].max(s[b])];
                    ridges.entry(key).or_default().push(si);
                    // The edge is on the hull if some facet opposite another vertex is.
                    let hull_edge = (0..s.len()).any(|k| k != a && k != b && tri.neighbors[si][k].is_none());
                    *on_hull.entry(key).or_insert(false) |= hull_edge;
                }
            }
        }
        let mut keys: Vec<[usize; 2]> = ridges.keys().copied().collect();
        keys.sort_unstable();
        let mut ridge_points = Vec::new();
        let mut ridge_vertices = Vec::new();
        let mut regions: Vec<Vec<Option<usize>>> = vec![Vec::new(); points.len()];
        for key in keys {
            // In 3-D and above an edge in a single simplex gives a degenerate ridge with one
            // finite vertex, which Qhull (and so SciPy) leaves out.
            if points[0].len() >= 3 && ridges[&key].len() < 2 {
                continue;
            }
            let mut vs: Vec<Option<usize>> = ridges[&key].iter().map(|&s| Some(s)).collect();
            if on_hull[&key] {
                vs.insert(0, None);
            }
            for &p in &key {
                for v in &vs {
                    if !regions[p].contains(v) {
                        regions[p].push(*v);
                    }
                }
            }
            ridge_points.push(key);
            ridge_vertices.push(vs);
        }
        Ok(VoronoiNd { points: points.to_vec(), vertices, ridge_points, ridge_vertices, regions })
    }
}
// ANCHOR_END: voronoi_nd

// ANCHOR: halfspace
/// The intersection of half-spaces `A x + b ≤ 0` (SciPy's `HalfspaceIntersection`), given a
/// point strictly inside all of them: each half-space maps to the dual point
/// `aᵢ / −(aᵢ·c + bᵢ)` (relative to the interior point `c`), each facet of the dual points'
/// convex hull to a vertex of the intersection.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HalfspaceIntersection {
    /// The vertices of the intersection, one per dual facet (SciPy's `intersections`).
    pub intersections: Vec<Vec<f64>>,
    /// For each vertex, the half-spaces whose boundaries meet there (SciPy's `dual_facets`).
    pub dual_facets: Vec<Vec<usize>>,
    /// The dual points' hull vertices: the half-spaces that are not redundant.
    pub dual_vertices: Vec<usize>,
}

impl HalfspaceIntersection {
    /// `halfspaces` rows are `[a, b]` (length `d + 1`).
    pub fn new(halfspaces: &[Vec<f64>], interior_point: &[f64]) -> Result<HalfspaceIntersection, ErrorsJSL> {
        let d = interior_point.len();
        if halfspaces.iter().any(|h| h.len() != d + 1) {
            return Err(ErrorsJSL::InvalidInputRange("each half-space is [a, b] with a of the point's dimension"));
        }
        let duals: Vec<Vec<f64>> = halfspaces
            .iter()
            .map(|h| {
                let s = -(dot(&h[..d], interior_point) + h[d]);
                h[..d].iter().map(|a| a / s).collect()
            })
            .collect();
        if halfspaces.iter().any(|h| dot(&h[..d], interior_point) + h[d] >= 0.0) {
            return Err(ErrorsJSL::InvalidInputRange("the interior point must lie strictly inside every half-space"));
        }
        let hull = ConvexHullNd::new(&duals)?;
        let intersections = hull.equations.iter().map(|e| (0..d).map(|k| -e[k] / e[d] + interior_point[k]).collect()).collect();
        Ok(HalfspaceIntersection { intersections, dual_facets: hull.simplices.clone(), dual_vertices: hull.vertices })
    }
}
// ANCHOR_END: halfspace

// ANCHOR: spherical_voronoi
/// The Voronoi diagram of points on a sphere in `ℝ³` (SciPy's `SphericalVoronoi`): the convex
/// hull of the points is their Delaunay triangulation on the sphere, and each triangle's
/// circumcentre projected onto the sphere (its outward normal times the radius) is a Voronoi
/// vertex. Regions are sorted counter-clockwise seen from outside.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SphericalVoronoi {
    pub vertices: Vec<[f64; 3]>,
    pub regions: Vec<Vec<usize>>,
    pub radius: f64,
    pub center: [f64; 3],
}

impl SphericalVoronoi {
    pub fn new(points: &[[f64; 3]], radius: f64, center: [f64; 3]) -> Result<SphericalVoronoi, ErrorsJSL> {
        if points.len() < 4 || radius.is_nan() || radius <= 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("need at least 4 points and a positive radius"));
        }
        let rel: Vec<Vec<f64>> = points.iter().map(|p| (0..3).map(|k| p[k] - center[k]).collect()).collect();
        if rel.iter().any(|p| ((dot(p, p)).sqrt() - radius).abs() > 1e-8 * radius.max(1.0)) {
            return Err(ErrorsJSL::InvalidInputRange("the points must lie on the sphere"));
        }
        let hull = ConvexHullNd::new(&rel)?;
        if hull.vertices.len() != points.len() {
            return Err(ErrorsJSL::InvalidInputRange("duplicate points"));
        }
        let vertices: Vec<[f64; 3]> = hull.equations.iter().map(|e| [e[0] * radius + center[0], e[1] * radius + center[1], e[2] * radius + center[2]]).collect();
        let mut regions: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
        for (f, s) in hull.simplices.iter().enumerate() {
            for &p in s {
                regions[p].push(f);
            }
        }
        // Sort each region's vertices by angle around the generator.
        for (p, reg) in regions.iter_mut().enumerate() {
            let n: Vec<f64> = rel[p].iter().map(|v| v / radius).collect();
            let helper = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            let u = {
                let c = dot(&helper, &n);
                let u: Vec<f64> = (0..3).map(|k| helper[k] - c * n[k]).collect();
                let un = dot(&u, &u).sqrt();
                u.iter().map(|x| x / un).collect::<Vec<_>>()
            };
            let w = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
            reg.sort_by(|&a, &b| {
                let ang = |v: usize| {
                    let q: Vec<f64> = (0..3).map(|k| vertices[v][k] - center[k]).collect();
                    dot(&q, &w).atan2(dot(&q, &u))
                };
                ang(a).total_cmp(&ang(b))
            });
        }
        Ok(SphericalVoronoi { vertices, regions, radius, center })
    }

    /// The area of each region on the sphere (SciPy's `calculate_areas`), from the spherical
    /// excess of the triangles fanned from the generator.
    pub fn calculate_areas(&self, points: &[[f64; 3]]) -> Vec<f64> {
        let r = self.radius;
        self.regions
            .iter()
            .zip(points)
            .map(|(reg, p)| {
                let unit = |q: &[f64; 3]| {
                    let v: Vec<f64> = (0..3).map(|k| q[k] - self.center[k]).collect();
                    let n = dot(&v, &v).sqrt();
                    v.iter().map(|x| x / n).collect::<Vec<f64>>()
                };
                let g = unit(p);
                let k = reg.len();
                (0..k)
                    .map(|i| {
                        let a = unit(&self.vertices[reg[i]]);
                        let b = unit(&self.vertices[reg[(i + 1) % k]]);
                        // Van Oosterom and Strackee: tan(E/2) = |g·(a×b)| / (1 + g·a + a·b + b·g).
                        let cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
                        let num = dot(&g, &cross).abs();
                        let den = 1.0 + dot(&g, &a) + dot(&a, &b) + dot(&b, &g);
                        2.0 * num.atan2(den)
                    })
                    .sum::<f64>()
                    * r
                    * r
            })
            .collect()
    }
}
// ANCHOR_END: spherical_voronoi

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                (0..d)
                    .map(|_| {
                        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (s >> 11) as f64 / (1u64 << 53) as f64
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn hull_of_a_cube_and_random_points() {
        // The unit cube with interior points: 8 vertices, 12 triangles, volume 1, area 6.
        let mut pts: Vec<Vec<f64>> = (0..8).map(|i| vec![(i & 1) as f64, (i >> 1 & 1) as f64, (i >> 2 & 1) as f64]).collect();
        pts.extend(lcg(50, 3, 1).into_iter().map(|p| p.iter().map(|v| 0.1 + 0.8 * v).collect()));
        let h = ConvexHullNd::new(&pts).unwrap();
        assert_eq!(h.vertices, (0..8).collect::<Vec<_>>());
        assert!((h.volume - 1.0).abs() < 1e-12 && (h.area - 6.0).abs() < 1e-12, "{} {}", h.volume, h.area);
        // 4-D: every random point is inside or on the hull, and Euler's relation for the
        // simplicial boundary (each ridge in two facets).
        let pts = lcg(60, 4, 2);
        let h = ConvexHullNd::new(&pts).unwrap();
        assert!(pts.iter().all(|p| h.equations.iter().all(|e| dot(&e[..4], p) + e[4] <= 1e-10)));
        assert!(h.neighbors.iter().enumerate().all(|(f, ns)| ns.iter().all(|&g| h.neighbors[g].contains(&f))));
    }

    #[test]
    fn delaunay_is_empty_sphere_and_covers_the_hull() {
        for d in [2usize, 3] {
            let pts = lcg(40, d, 3 + d as u64);
            let t = DelaunayNd::new(&pts).unwrap();
            // Empty circumspheres.
            for s in &t.simplices {
                let c = circumcenter(&s.iter().map(|&i| &pts[i]).collect::<Vec<_>>()).unwrap();
                let r2 = dot(&sub(&pts[s[0]], &c), &sub(&pts[s[0]], &c));
                assert!(pts.iter().all(|p| dot(&sub(p, &c), &sub(p, &c)) >= r2 * (1.0 - 1e-9)));
            }
            // Volumes add up to the hull's.
            let vol: f64 = t.simplices.iter().map(|s| det((1..=d).map(|k| sub(&pts[s[k]], &pts[s[0]])).collect()).abs() / factorial(d)).sum();
            assert!((vol - ConvexHullNd::new(&pts).unwrap().volume).abs() < 1e-12, "{d}");
            let x: Vec<f64> = vec![0.5; d];
            let s = t.find_simplex(&x).unwrap();
            assert!(t.barycentric(s, &x).iter().all(|c| *c >= -1e-12));
        }
        // A grid (cospherical quadruples) still triangulates.
        let grid: Vec<Vec<f64>> = (0..16).map(|i| vec![(i % 4) as f64, (i / 4) as f64]).collect();
        let t = DelaunayNd::new(&grid).unwrap();
        assert_eq!(t.simplices.len(), 18);
    }

    #[test]
    fn halfspaces_of_a_square_and_a_spherical_voronoi() {
        // |x| ≤ 1, |y| ≤ 1 plus a redundant x + y ≤ 5.
        let hs = vec![vec![1.0, 0.0, -1.0], vec![-1.0, 0.0, -1.0], vec![0.0, 1.0, -1.0], vec![0.0, -1.0, -1.0], vec![1.0, 1.0, -5.0]];
        let h = HalfspaceIntersection::new(&hs, &[0.1, -0.2]).unwrap();
        let mut v: Vec<(i64, i64)> = h.intersections.iter().map(|p| ((p[0] * 1e9).round() as i64, (p[1] * 1e9).round() as i64)).collect();
        v.sort_unstable();
        assert_eq!(v, vec![(-1_000_000_000, -1_000_000_000), (-1_000_000_000, 1_000_000_000), (1_000_000_000, -1_000_000_000), (1_000_000_000, 1_000_000_000)]);
        assert_eq!(h.dual_vertices, vec![0, 1, 2, 3]);
        // The octahedron's vertices on the unit sphere: the Voronoi regions are the 8 octants'
        // cube corners, each region of area 4π/6.
        let pts = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
        let sv = SphericalVoronoi::new(&pts, 1.0, [0.0; 3]).unwrap();
        assert_eq!(sv.vertices.len(), 8);
        let s3 = 1.0 / 3f64.sqrt();
        assert!(sv.vertices.iter().all(|v| v.iter().all(|c| (c.abs() - s3).abs() < 1e-12)));
        assert!(sv.calculate_areas(&pts).iter().all(|a| (a - 4.0 * std::f64::consts::PI / 6.0).abs() < 1e-12));
    }
}
