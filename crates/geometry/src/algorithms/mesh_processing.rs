//! Triangle mesh smoothing and decimation, on indexed meshes (`vertices`, `triangles`).

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap};

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The neighbouring vertices of every vertex.
pub fn vertex_neighbours(vertex_count: usize, triangles: &[[usize; 3]]) -> Vec<BTreeSet<usize>> {
    let mut n = vec![BTreeSet::new(); vertex_count];
    for t in triangles {
        for k in 0..3 {
            n[t[k]].insert(t[(k + 1) % 3]);
            n[t[(k + 1) % 3]].insert(t[k]);
        }
    }
    n
}

/// Whether each vertex lies on the boundary (on an edge used by only one triangle).
pub fn boundary_vertices(vertex_count: usize, triangles: &[[usize; 3]]) -> Vec<bool> {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for t in triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let mut boundary = vec![false; vertex_count];
    for (&(a, b), &c) in &count {
        if c == 1 {
            boundary[a] = true;
            boundary[b] = true;
        }
    }
    boundary
}

fn umbrella_step(vertices: &mut [V3], neighbours: &[BTreeSet<usize>], fixed: &[bool], factor: f64) {
    let old = vertices.to_vec();
    for (i, v) in vertices.iter_mut().enumerate() {
        if fixed[i] || neighbours[i].is_empty() {
            continue;
        }
        let k = neighbours[i].len() as f64;
        for d in 0..3 {
            let mean = neighbours[i].iter().map(|&j| old[j][d]).sum::<f64>() / k;
            v[d] += factor * (mean - old[i][d]);
        }
    }
}

/// Laplacian (umbrella) smoothing: each vertex moves a fraction `lambda` towards the mean of its
/// neighbours, `iterations` times. Boundary vertices stay put with `keep_boundary`. Simple but it
/// shrinks closed surfaces; see [`taubin_smoothing`].
pub fn laplacian_smoothing(vertices: &[V3], triangles: &[[usize; 3]], iterations: usize, lambda: f64, keep_boundary: bool) -> Vec<V3> {
    let neighbours = vertex_neighbours(vertices.len(), triangles);
    let fixed = if keep_boundary { boundary_vertices(vertices.len(), triangles) } else { vec![false; vertices.len()] };
    let mut v = vertices.to_vec();
    for _ in 0..iterations {
        umbrella_step(&mut v, &neighbours, &fixed, lambda);
    }
    v
}

/// Taubin's λ|μ smoothing: alternating shrinking (`λ > 0`) and inflating (`μ < −λ`) Laplacian
/// steps form a low-pass filter that removes noise without the shrinkage of plain Laplacian
/// smoothing (typical values `λ = 0.5`, `μ = −0.53`).
pub fn taubin_smoothing(vertices: &[V3], triangles: &[[usize; 3]], iterations: usize, lambda: f64, mu: f64, keep_boundary: bool) -> Vec<V3> {
    let neighbours = vertex_neighbours(vertices.len(), triangles);
    let fixed = if keep_boundary { boundary_vertices(vertices.len(), triangles) } else { vec![false; vertices.len()] };
    let mut v = vertices.to_vec();
    for _ in 0..iterations {
        umbrella_step(&mut v, &neighbours, &fixed, lambda);
        umbrella_step(&mut v, &neighbours, &fixed, mu);
    }
    v
}

/// A symmetric 4×4 quadric stored as its 10 upper-triangle entries.
#[derive(Clone, Copy, Default)]

struct Quadric([f64; 10]);

impl Quadric {
    fn plane(n: V3, d: f64, weight: f64) -> Self {
        let p = [n[0], n[1], n[2], d];
        let mut q = [0.0; 10];
        let mut k = 0;
        for i in 0..4 {
            for j in i..4 {
                q[k] = weight * p[i] * p[j];
                k += 1;
            }
        }
        Quadric(q)
    }

    fn add(self, o: Quadric) -> Quadric {
        let mut q = self.0;
        q.iter_mut().zip(o.0).for_each(|(a, b)| *a += b);
        Quadric(q)
    }

    /// `vᵀ Q v` for the homogeneous point `(x, y, z, 1)`.
    fn error(&self, v: V3) -> f64 {
        let q = &self.0;
        let (x, y, z) = (v[0], v[1], v[2]);
        q[0] * x * x + 2.0 * q[1] * x * y + 2.0 * q[2] * x * z + 2.0 * q[3] * x + q[4] * y * y + 2.0 * q[5] * y * z + 2.0 * q[6] * y + q[7] * z * z + 2.0 * q[8] * z + q[9]
    }

    /// The minimiser of the quadric, if its 3×3 block is well conditioned.
    fn optimum(&self) -> Option<V3> {
        let q = &self.0;
        let a = [[q[0], q[1], q[2]], [q[1], q[4], q[5]], [q[2], q[5], q[7]]];
        let b = [-q[3], -q[6], -q[8]];
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        let scale = (q[0] + q[4] + q[7]).powi(3).max(1e-300);
        if det.abs() < 1e-10 * scale {
            return None;
        }
        let col = |c: usize| {
            let mut m = a;
            for r in 0..3 {
                m[r][c] = b[r];
            }
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };
        Some([col(0) / det, col(1) / det, col(2) / det])
    }
}

#[derive(PartialEq)]

struct Candidate {
    cost: f64,
    a: usize,
    b: usize,
    stamp: (u32, u32),
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.total_cmp(&self.cost)
    }
}

/// Simplifies a triangle mesh to about `target_triangles` triangles by quadric error metric edge
/// collapse (Garland and Heckbert): each vertex accumulates the area-weighted squared distances
/// to the planes of its original triangles, and the edge whose collapse to the quadric-optimal
/// point adds the least error is collapsed first. Collapses that would flip a triangle, break the
/// manifold link condition are skipped; boundary vertices never move (interior vertices may
/// collapse onto them, but boundary edges are never collapsed), so open boundaries are
/// preserved exactly. Returns the compacted vertices and triangles.
pub fn decimate(vertices: &[V3], triangles: &[[usize; 3]], target_triangles: usize) -> (Vec<V3>, Vec<[usize; 3]>) {
    let n = vertices.len();
    let mut pos = vertices.to_vec();
    let mut tris: Vec<Option<[usize; 3]>> = triangles.iter().map(|&t| Some(t)).collect();
    let mut quadric = vec![Quadric::default(); n];
    let mut incident: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for (ti, t) in triangles.iter().enumerate() {
        let normal = cross(sub(pos[t[1]], pos[t[0]]), sub(pos[t[2]], pos[t[0]]));
        let length = dot(normal, normal).sqrt();
        if length > 0.0 {
            let unit = [normal[0] / length, normal[1] / length, normal[2] / length];
            let q = Quadric::plane(unit, -dot(unit, pos[t[0]]), length / 2.0);
            for &v in t {
                quadric[v] = quadric[v].add(q);
            }
        }
        for &v in t {
            incident[v].insert(ti);
        }
    }
    let boundary = boundary_vertices(n, triangles);
    let mut version = vec![0u32; n];
    let mut alive_count = triangles.len();
    // Where an edge collapses to: onto its boundary endpoint (boundaries never move), otherwise
    // to the quadric optimum (or the midpoint when the quadric is singular).
    let placement = |a: usize, b: usize, pos: &[V3], q: &Quadric| -> V3 {
        if boundary[a] {
            return pos[a];
        }
        if boundary[b] {
            return pos[b];
        }
        let midpoint = [(pos[a][0] + pos[b][0]) / 2.0, (pos[a][1] + pos[b][1]) / 2.0, (pos[a][2] + pos[b][2]) / 2.0];
        q.optimum().unwrap_or(midpoint)
    };
    let candidate = |a: usize, b: usize, pos: &[V3], quadric: &[Quadric], version: &[u32]| -> Candidate {
        let q = quadric[a].add(quadric[b]);
        let target = placement(a, b, pos, &q);
        // The survivor is a; keep a boundary endpoint as the survivor.
        let (a, b) = if boundary[b] { (b, a) } else { (a, b) };
        Candidate { cost: q.error(target).max(0.0), a, b, stamp: (version[a], version[b]) }
    };
    let mut heap = BinaryHeap::new();
    for (a, b) in edges(&tris) {
        if !(boundary[a] && boundary[b]) {
            heap.push(candidate(a, b, &pos, &quadric, &version));
        }
    }
    while alive_count > target_triangles {
        let Some(c) = heap.pop() else { break };
        let (a, b) = (c.a, c.b);
        if c.stamp != (version[a], version[b]) {
            continue;
        }
        // The edge must still exist, and the link condition must hold: a and b share exactly
        // the two neighbours of the triangles on the edge.
        let shared: Vec<usize> = incident[a].intersection(&incident[b]).copied().collect();
        if shared.len() != 2 {
            continue;
        }
        let neighbours = |v: usize, inc: &BTreeSet<usize>, tris: &[Option<[usize; 3]>]| -> BTreeSet<usize> { inc.iter().flat_map(|&t| tris[t].expect("alive")).filter(|&w| w != v).collect() };
        let common: Vec<usize> = neighbours(a, &incident[a], &tris).intersection(&neighbours(b, &incident[b], &tris)).copied().collect();
        if common.len() != 2 {
            continue;
        }
        let q = quadric[a].add(quadric[b]);
        let target = placement(a, b, &pos, &q);
        // Reject collapses that flip any surviving triangle around a or b.
        let flips = incident[a].union(&incident[b]).filter(|t| !shared.contains(t)).any(|&t| {
            let tri = tris[t].expect("alive");
            let before = cross(sub(pos[tri[1]], pos[tri[0]]), sub(pos[tri[2]], pos[tri[0]]));
            let moved: Vec<V3> = tri.iter().map(|&v| if v == a || v == b { target } else { pos[v] }).collect();
            let after = cross(sub(moved[1], moved[0]), sub(moved[2], moved[0]));
            dot(before, after) <= 1e-12 * dot(before, before)
        });
        if flips {
            continue;
        }
        // Collapse b into a.
        for &t in &shared {
            for &v in &tris[t].expect("alive") {
                incident[v].remove(&t);
            }
            tris[t] = None;
            alive_count -= 1;
        }
        let moved: Vec<usize> = incident[b].iter().copied().collect();
        for t in moved {
            let tri = tris[t].as_mut().expect("alive");
            tri.iter_mut().for_each(|v| if *v == b { *v = a });
            incident[a].insert(t);
        }
        incident[b].clear();
        pos[a] = target;
        quadric[a] = q;
        version[a] += 1;
        version[b] += 1;
        let around: BTreeSet<usize> = incident[a].iter().flat_map(|&t| tris[t].expect("alive")).filter(|&w| w != a).collect();
        for w in around {
            if !(boundary[a] && boundary[w]) {
                heap.push(candidate(a, w, &pos, &quadric, &version));
            }
        }
    }
    // Compact the surviving vertices.
    let mut remap = vec![usize::MAX; n];
    let mut out_vertices = Vec::new();
    let mut out_triangles = Vec::new();
    for t in tris.into_iter().flatten() {
        out_triangles.push(t.map(|v| {
            if remap[v] == usize::MAX {
                remap[v] = out_vertices.len();
                out_vertices.push(pos[v]);
            }
            remap[v]
        }));
    }
    (out_vertices, out_triangles)
}

fn edges(tris: &[Option<[usize; 3]>]) -> BTreeSet<(usize, usize)> {
    tris.iter().flatten().flat_map(|t| (0..3).map(move |k| (t[k].min(t[(k + 1) % 3]), t[k].max(t[(k + 1) % 3])))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A UV sphere of radius one with `rings × segments` quads split into triangles.
    fn sphere(rings: usize, segments: usize) -> (Vec<V3>, Vec<[usize; 3]>) {
        use std::f64::consts::PI;
        let mut v = vec![[0.0, 0.0, 1.0]];
        for r in 1..rings {
            let theta = PI * r as f64 / rings as f64;
            for s in 0..segments {
                let phi = 2.0 * PI * s as f64 / segments as f64;
                v.push([theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()]);
            }
        }
        v.push([0.0, 0.0, -1.0]);
        let south = v.len() - 1;
        let id = |r: usize, s: usize| 1 + (r - 1) * segments + s % segments;
        let mut t = Vec::new();
        for s in 0..segments {
            t.push([0, id(1, s), id(1, s + 1)]);
            t.push([south, id(rings - 1, s + 1), id(rings - 1, s)]);
            for r in 1..rings - 1 {
                t.push([id(r, s), id(r + 1, s), id(r + 1, s + 1)]);
                t.push([id(r, s), id(r + 1, s + 1), id(r, s + 1)]);
            }
        }
        (v, t)
    }

    fn euler_characteristic(v: usize, t: &[[usize; 3]]) -> i64 {
        v as i64 - edges(&t.iter().map(|&x| Some(x)).collect::<Vec<_>>()).len() as i64 + t.len() as i64
    }

    #[test]
    fn decimation_keeps_a_closed_sphere_round_and_watertight() {
        let (v, t) = sphere(24, 48);
        let (dv, dt) = decimate(&v, &t, t.len() / 5);
        assert!(dt.len() <= t.len() / 5 + 2 && dt.len() > t.len() / 6, "{} of {}", dt.len(), t.len());
        assert_eq!(euler_characteristic(dv.len(), &dt), 2);
        assert!(boundary_vertices(dv.len(), &dt).iter().all(|&b| !b));
        let worst = dv.iter().map(|p| (dot(*p, *p).sqrt() - 1.0).abs()).fold(0.0, f64::max);
        assert!(worst < 0.02, "{worst}");
    }

    #[test]
    fn flat_grids_collapse_without_error_and_keep_their_boundary() {
        let n = 10;
        let v: Vec<V3> = (0..=n).flat_map(|j| (0..=n).map(move |i| [i as f64, j as f64, 0.0])).collect();
        let id = |i: usize, j: usize| j * (n + 1) + i;
        let t: Vec<[usize; 3]> = (0..n).flat_map(|j| (0..n).flat_map(move |i| [[id(i, j), id(i + 1, j), id(i + 1, j + 1)], [id(i, j), id(i + 1, j + 1), id(i, j + 1)]])).collect();
        let (dv, dt) = decimate(&v, &t, 0);
        // Only the boundary vertices remain, plus at most one interior vertex whose collapse onto
        // the boundary would create zero-area triangles; the area (100) is preserved.
        assert!(dv.len() <= 4 * n + 1 && dv.iter().all(|p| p[2] == 0.0), "{} vertices, {} triangles", dv.len(), dt.len());
        let area: f64 = dt.iter().map(|tri| 0.5 * cross(sub(dv[tri[1]], dv[tri[0]]), sub(dv[tri[2]], dv[tri[0]]))[2]).sum();
        assert!((area - 100.0).abs() < 1e-9, "{area}");
    }

    #[test]
    fn taubin_smoothing_denoises_without_shrinking() {
        let (v, t) = sphere(16, 32);
        let mut s: u64 = 9;
        let noisy: Vec<V3> = v.iter().map(|p| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let e = 1.0 + 0.05 * ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
            [p[0] * e, p[1] * e, p[2] * e]
        }).collect();
        let radius_spread = |m: &[V3]| {
            let r: Vec<f64> = m.iter().map(|p| dot(*p, *p).sqrt()).collect();
            let mean = r.iter().sum::<f64>() / r.len() as f64;
            (mean, (r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / r.len() as f64).sqrt())
        };
        let (_, noise) = radius_spread(&noisy);
        let (lap_mean, _) = radius_spread(&laplacian_smoothing(&noisy, &t, 20, 0.5, false));
        let (taubin_mean, taubin_spread) = radius_spread(&taubin_smoothing(&noisy, &t, 20, 0.5, -0.53, false));
        assert!(taubin_spread < noise / 2.0, "{taubin_spread} vs {noise}");
        assert!((1.0 - taubin_mean).abs() < 0.03 && 1.0 - lap_mean > 2.0 * (1.0 - taubin_mean).abs(), "{lap_mean} {taubin_mean}");
        // Fixed boundaries stay put.
        let flat = laplacian_smoothing(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], &[[0, 1, 2]], 5, 0.5, true);
        assert_eq!(flat[1], [1.0, 0.0, 0.0]);
    }
}
