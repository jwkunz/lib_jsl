//! Interpolation of scattered 2-D data on a Delaunay triangulation, following
//! `scipy.interpolate`: piecewise linear (`LinearNDInterpolator`), the C¹ Clough-Tocher scheme
//! with globally estimated gradients (`CloughTocher2DInterpolator`), nearest neighbour
//! (`NearestNDInterpolator`, any dimension) and the `griddata` front end.
//!
//! See the book entry [Piecewise polynomials, Hermite data and scattered interpolation](https://jwkunz.github.io/lib_jsl/book/interpolation/piecewise-scattered.html#scattered-data-on-a-triangulation).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_geometry::algorithms::delaunay_triangulation;
use std::collections::HashMap;

/// A Delaunay triangulation with neighbour links and a bucket grid for point location.
#[derive(Clone, Debug)]
pub struct TriangulationIndex {
    pub points: Vec<[f64; 2]>,
    /// Triangles, counter-clockwise.
    pub triangles: Vec<[usize; 3]>,
    /// `neighbors[t][k]`: the triangle across the edge opposite vertex `k` (Qhull's convention).
    pub neighbors: Vec<[Option<usize>; 3]>,
    origin: [f64; 2],
    cell: [f64; 2],
    dims: [usize; 2],
    buckets: Vec<Vec<usize>>,
}

impl TriangulationIndex {
    /// Triangulates the points (at least three, not all collinear).
    pub fn new(points: &[[f64; 2]]) -> Result<TriangulationIndex, ErrorsJSL> {
        let tri = delaunay_triangulation(points).ok_or(ErrorsJSL::InvalidInputRange("need three or more points, not all collinear"))?;
        let triangles = tri.triangles;
        let mut edge_owner: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
        for (t, tr) in triangles.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (tr[(k + 1) % 3], tr[(k + 2) % 3]);
                edge_owner.entry((a.min(b), a.max(b))).or_default().push((t, k));
            }
        }
        let mut neighbors = vec![[None; 3]; triangles.len()];
        for owners in edge_owner.values() {
            if let [(t1, k1), (t2, k2)] = owners[..] {
                neighbors[t1][k1] = Some(t2);
                neighbors[t2][k2] = Some(t1);
            }
        }
        // Bucket grid of about one triangle per cell.
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for p in points {
            for d in 0..2 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        let side = (triangles.len() as f64).sqrt().ceil().max(1.0) as usize;
        let dims = [side, side];
        let cell = [((hi[0] - lo[0]) / side as f64).max(f64::MIN_POSITIVE), ((hi[1] - lo[1]) / side as f64).max(f64::MIN_POSITIVE)];
        let mut buckets = vec![Vec::new(); side * side];
        let to_cell = |v: f64, d: usize| (((v - lo[d]) / cell[d]).floor().max(0.0) as usize).min(dims[d] - 1);
        for (t, tr) in triangles.iter().enumerate() {
            let xs = tr.map(|i| points[i][0]);
            let ys = tr.map(|i| points[i][1]);
            let (cx0, cx1) = (to_cell(xs.iter().copied().fold(f64::INFINITY, f64::min), 0), to_cell(xs.iter().copied().fold(f64::NEG_INFINITY, f64::max), 0));
            let (cy0, cy1) = (to_cell(ys.iter().copied().fold(f64::INFINITY, f64::min), 1), to_cell(ys.iter().copied().fold(f64::NEG_INFINITY, f64::max), 1));
            for cx in cx0..=cx1 {
                for cy in cy0..=cy1 {
                    buckets[cx * side + cy].push(t);
                }
            }
        }
        Ok(TriangulationIndex { points: points.to_vec(), triangles, neighbors, origin: lo, cell, dims, buckets })
    }

    /// The barycentric coordinates of `p` in triangle `t`.
    pub fn barycentric(&self, t: usize, p: [f64; 2]) -> [f64; 3] {
        let [a, b, c] = self.triangles[t].map(|i| self.points[i]);
        let det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
        let l1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det;
        let l2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det;
        [l1, l2, 1.0 - l1 - l2]
    }

    /// A triangle containing `p` and its barycentric coordinates, allowing a relative slack of
    /// `100 ε` on the coordinates (as Qhull's `find_simplex`); `None` outside the convex hull.
    pub fn find(&self, p: [f64; 2]) -> Option<(usize, [f64; 3])> {
        if !p.iter().all(|v| v.is_finite()) {
            return None;
        }
        let cx = (p[0] - self.origin[0]) / self.cell[0];
        let cy = (p[1] - self.origin[1]) / self.cell[1];
        let eps = 100.0 * f64::EPSILON;
        if cx < -eps || cy < -eps || cx > self.dims[0] as f64 + eps || cy > self.dims[1] as f64 + eps {
            return None;
        }
        let (ix, iy) = ((cx.floor().max(0.0) as usize).min(self.dims[0] - 1), (cy.floor().max(0.0) as usize).min(self.dims[1] - 1));
        self.buckets[ix * self.dims[1] + iy].iter().find_map(|&t| {
            let b = self.barycentric(t, p);
            b.iter().all(|&v| v >= -eps).then_some((t, b))
        })
    }

    /// The neighbours of each vertex (sorted, without duplicates).
    pub fn vertex_neighbors(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.points.len()];
        for tr in &self.triangles {
            for k in 0..3 {
                out[tr[k]].push(tr[(k + 1) % 3]);
                out[tr[k]].push(tr[(k + 2) % 3]);
            }
        }
        for v in out.iter_mut() {
            v.sort_unstable();
            v.dedup();
        }
        out
    }
}

/// Piecewise-linear interpolation on the Delaunay triangulation (SciPy's
/// `LinearNDInterpolator` in 2-D): barycentric weights of the containing triangle.
#[derive(Clone, Debug)]
pub struct LinearNDInterpolator {
    tri: TriangulationIndex,
    values: Vec<f64>,
    pub fill_value: f64,
}

impl LinearNDInterpolator {
    pub fn new(points: &[[f64; 2]], values: &[f64]) -> Result<LinearNDInterpolator, ErrorsJSL> {
        if points.len() != values.len() {
            return Err(ErrorsJSL::InvalidInputRange("one value per point"));
        }
        Ok(LinearNDInterpolator { tri: TriangulationIndex::new(points)?, values: values.to_vec(), fill_value: f64::NAN })
    }

    /// The interpolated value, or `fill_value` (`NaN` by default) outside the convex hull.
    pub fn evaluate(&self, p: [f64; 2]) -> f64 {
        match self.tri.find(p) {
            Some((t, b)) => (0..3).map(|k| b[k] * self.values[self.tri.triangles[t][k]]).sum(),
            None => self.fill_value,
        }
    }
}

// ANCHOR: gradients
/// Gradient estimates at the vertices that minimise the integrated squared second derivative of
/// the Clough-Tocher interpolant along every edge (Nielson's global method, SciPy's
/// `estimate_gradients_2d_global`). Each vertex's 2 × 2 local problem is solved in turn with the
/// current estimates at its neighbours (Gauss-Seidel), until the largest change is below `tol`
/// (relative to the gradient, or absolute when it is below 1) or after `max_iterations` sweeps.
pub fn estimate_gradients_2d_global(tri: &TriangulationIndex, values: &[f64], tol: f64, max_iterations: usize) -> Vec<[f64; 2]> {
    let nbrs = tri.vertex_neighbors();
    let mut y = vec![[0.0; 2]; tri.points.len()];
    for _ in 0..max_iterations {
        let mut err = 0.0f64;
        for ip in 0..tri.points.len() {
            let (mut q0, mut q1, mut q3, mut s0, mut s1) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for &jp in &nbrs[ip] {
                let ex = tri.points[jp][0] - tri.points[ip][0];
                let ey = tri.points[jp][1] - tri.points[ip][1];
                let l = ex.hypot(ey);
                let l3 = l * l * l;
                let (f1, f2) = (values[ip], values[jp]);
                let df2 = -ex * y[jp][0] - ey * y[jp][1];
                q0 += 4.0 * ex * ex / l3;
                q1 += 4.0 * ex * ey / l3;
                q3 += 4.0 * ey * ey / l3;
                s0 += (6.0 * (f1 - f2) - 2.0 * df2) * ex / l3;
                s1 += (6.0 * (f1 - f2) - 2.0 * df2) * ey / l3;
            }
            let det = q0 * q3 - q1 * q1;
            let r0 = (q3 * s0 - q1 * s1) / det;
            let r1 = (-q1 * s0 + q0 * s1) / det;
            let change = (y[ip][0] + r0).abs().max((y[ip][1] + r1).abs()) / r0.abs().max(r1.abs()).max(1.0);
            y[ip] = [-r0, -r1];
            err = err.max(change);
        }
        if err < tol {
            break;
        }
    }
    y
}
// ANCHOR_END: gradients

/// The C¹ piecewise-cubic Clough-Tocher interpolant (SciPy's `CloughTocher2DInterpolator`): each
/// triangle is split at its centroid into three cubic Bézier patches, fixed by the values and
/// gradients at the vertices and by requiring the cross-boundary derivative to be linear along
/// each edge.
#[derive(Clone, Debug)]
pub struct CloughTocher2DInterpolator {
    tri: TriangulationIndex,
    values: Vec<f64>,
    gradients: Vec<[f64; 2]>,
    pub fill_value: f64,
}

impl CloughTocher2DInterpolator {
    /// Triangulates, estimates the gradients (tolerance `1e-6`, at most 400 sweeps, as SciPy).
    pub fn new(points: &[[f64; 2]], values: &[f64]) -> Result<CloughTocher2DInterpolator, ErrorsJSL> {
        if points.len() != values.len() {
            return Err(ErrorsJSL::InvalidInputRange("one value per point"));
        }
        let tri = TriangulationIndex::new(points)?;
        let gradients = estimate_gradients_2d_global(&tri, values, 1e-6, 400);
        Ok(CloughTocher2DInterpolator { tri, values: values.to_vec(), gradients, fill_value: f64::NAN })
    }

    /// The estimated gradients at the data points.
    pub fn gradients(&self) -> &[[f64; 2]] {
        &self.gradients
    }

    /// The interpolated value, or `fill_value` outside the convex hull.
    pub fn evaluate(&self, p: [f64; 2]) -> f64 {
        match self.tri.find(p) {
            Some((t, b)) => self.single(t, b),
            None => self.fill_value,
        }
    }

    // ANCHOR: clough_tocher
    /// SciPy's `_clough_tocher_2d_single`: the Bézier ordinates of the three sub-triangles from
    /// the vertex data, the cross-derivative direction taken towards the neighbouring triangle's
    /// centroid (which makes the scheme affine invariant), and evaluation in extended barycentric
    /// coordinates.
    fn single(&self, t: usize, b: [f64; 3]) -> f64 {
        let tr = self.tri.triangles[t];
        let p = |k: usize| self.tri.points[tr[k]];
        let (e12x, e12y) = (p(1)[0] - p(0)[0], p(1)[1] - p(0)[1]);
        let (e23x, e23y) = (p(2)[0] - p(1)[0], p(2)[1] - p(1)[1]);
        let (e31x, e31y) = (p(0)[0] - p(2)[0], p(0)[1] - p(2)[1]);
        let f = |k: usize| self.values[tr[k]];
        let g = |k: usize| self.gradients[tr[k]];
        let (f1, f2, f3) = (f(0), f(1), f(2));
        let df12 = g(0)[0] * e12x + g(0)[1] * e12y;
        let df21 = -(g(1)[0] * e12x + g(1)[1] * e12y);
        let df23 = g(1)[0] * e23x + g(1)[1] * e23y;
        let df32 = -(g(2)[0] * e23x + g(2)[1] * e23y);
        let df31 = g(2)[0] * e31x + g(2)[1] * e31y;
        let df13 = -(g(0)[0] * e31x + g(0)[1] * e31y);
        let c3000 = f1;
        let c2100 = (df12 + 3.0 * c3000) / 3.0;
        let c2010 = (df13 + 3.0 * c3000) / 3.0;
        let c0300 = f2;
        let c1200 = (df21 + 3.0 * c0300) / 3.0;
        let c0210 = (df23 + 3.0 * c0300) / 3.0;
        let c0030 = f3;
        let c1020 = (df31 + 3.0 * c0030) / 3.0;
        let c0120 = (df32 + 3.0 * c0030) / 3.0;
        let c2001 = (c2100 + c2010 + c3000) / 3.0;
        let c0201 = (c1200 + c0300 + c0210) / 3.0;
        let c0021 = (c1020 + c0120 + c0030) / 3.0;
        let mut gk = [0.0; 3];
        for (k, gv) in gk.iter_mut().enumerate() {
            let Some(nt) = self.tri.neighbors[t][k] else {
                *gv = -0.5;
                continue;
            };
            let ntr = self.tri.triangles[nt];
            let cen = [(0..3).map(|i| self.tri.points[ntr[i]][0]).sum::<f64>() / 3.0, (0..3).map(|i| self.tri.points[ntr[i]][1]).sum::<f64>() / 3.0];
            let c = self.tri.barycentric(t, cen);
            *gv = match k {
                0 => (2.0 * c[2] + c[1] - 1.0) / (2.0 - 3.0 * c[2] - 3.0 * c[1]),
                1 => (2.0 * c[0] + c[2] - 1.0) / (2.0 - 3.0 * c[0] - 3.0 * c[2]),
                _ => (2.0 * c[1] + c[0] - 1.0) / (2.0 - 3.0 * c[1] - 3.0 * c[0]),
            };
        }
        let c0111 = (gk[0] * (-c0300 + 3.0 * c0210 - 3.0 * c0120 + c0030) + (-c0300 + 2.0 * c0210 - c0120 + c0021 + c0201)) / 2.0;
        let c1011 = (gk[1] * (-c0030 + 3.0 * c1020 - 3.0 * c2010 + c3000) + (-c0030 + 2.0 * c1020 - c2010 + c2001 + c0021)) / 2.0;
        let c1101 = (gk[2] * (-c3000 + 3.0 * c2100 - 3.0 * c1200 + c0300) + (-c3000 + 2.0 * c2100 - c1200 + c2001 + c0201)) / 2.0;
        let c1002 = (c1101 + c1011 + c2001) / 3.0;
        let c0102 = (c1101 + c0111 + c0201) / 3.0;
        let c0012 = (c1011 + c0111 + c0021) / 3.0;
        let c0003 = (c1002 + c0102 + c0012) / 3.0;
        let minval = b[0].min(b[1]).min(b[2]);
        let (b1, b2, b3, b4) = (b[0] - minval, b[1] - minval, b[2] - minval, 3.0 * minval);
        b1.powi(3) * c3000 + 3.0 * b1 * b1 * b2 * c2100 + 3.0 * b1 * b1 * b3 * c2010 + 3.0 * b1 * b1 * b4 * c2001 + 3.0 * b1 * b2 * b2 * c1200
            + 6.0 * b1 * b2 * b4 * c1101 + 3.0 * b1 * b3 * b3 * c1020 + 6.0 * b1 * b3 * b4 * c1011 + 3.0 * b1 * b4 * b4 * c1002
            + b2.powi(3) * c0300 + 3.0 * b2 * b2 * b3 * c0210 + 3.0 * b2 * b2 * b4 * c0201 + 3.0 * b2 * b3 * b3 * c0120
            + 6.0 * b2 * b3 * b4 * c0111 + 3.0 * b2 * b4 * b4 * c0102 + b3.powi(3) * c0030 + 3.0 * b3 * b3 * b4 * c0021
            + 3.0 * b3 * b4 * b4 * c0012 + b4.powi(3) * c0003
    }
    // ANCHOR_END: clough_tocher
}

/// Nearest-neighbour interpolation in any dimension (SciPy's `NearestNDInterpolator`); ties go
/// to the first point.
#[derive(Clone, Debug)]
pub struct NearestNDInterpolator {
    points: Vec<Vec<f64>>,
    values: Vec<f64>,
}

impl NearestNDInterpolator {
    pub fn new(points: &[Vec<f64>], values: &[f64]) -> Result<NearestNDInterpolator, ErrorsJSL> {
        if points.is_empty() || points.len() != values.len() {
            return Err(ErrorsJSL::InvalidInputRange("need one value per point"));
        }
        let d = points[0].len();
        if points.iter().any(|p| p.len() != d) {
            return Err(ErrorsJSL::InvalidInputRange("all points must have the same dimension"));
        }
        Ok(NearestNDInterpolator { points: points.to_vec(), values: values.to_vec() })
    }

    pub fn evaluate(&self, p: &[f64]) -> f64 {
        let dist = |q: &Vec<f64>| q.iter().zip(p).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
        let mut best = (0, f64::INFINITY);
        for (i, q) in self.points.iter().enumerate() {
            let d = dist(q);
            if d < best.1 {
                best = (i, d);
            }
        }
        self.values[best.0]
    }
}

/// The method of [`griddata`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GriddataMethod {
    Nearest,
    Linear,
    Cubic,
}

/// Interpolates scattered 2-D data at the query points (SciPy's `griddata`): nearest neighbour,
/// piecewise linear or Clough-Tocher; outside the convex hull the linear and cubic methods return
/// `NaN`.
pub fn griddata(points: &[[f64; 2]], values: &[f64], queries: &[[f64; 2]], method: GriddataMethod) -> Result<Vec<f64>, ErrorsJSL> {
    Ok(match method {
        GriddataMethod::Nearest => {
            let n = NearestNDInterpolator::new(&points.iter().map(|p| p.to_vec()).collect::<Vec<_>>(), values)?;
            queries.iter().map(|q| n.evaluate(q)).collect()
        }
        GriddataMethod::Linear => {
            let l = LinearNDInterpolator::new(points, values)?;
            queries.iter().map(|&q| l.evaluate(q)).collect()
        }
        GriddataMethod::Cubic => {
            let c = CloughTocher2DInterpolator::new(points, values)?;
            queries.iter().map(|&q| c.evaluate(q)).collect()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points() -> Vec<[f64; 2]> {
        let mut s: u64 = 12345;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut p: Vec<[f64; 2]> = (0..80).map(|_| [next(), next()]).collect();
        p.extend([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        p
    }

    #[test]
    fn linear_reproduces_planes_and_cubic_reproduces_quadratics_approximately() {
        let p = points();
        let plane: Vec<f64> = p.iter().map(|q| 2.0 * q[0] - q[1] + 0.5).collect();
        let l = LinearNDInterpolator::new(&p, &plane).unwrap();
        for q in [[0.3, 0.7], [0.91, 0.05], [0.5, 0.5]] {
            assert!((l.evaluate(q) - (2.0 * q[0] - q[1] + 0.5)).abs() < 1e-12);
        }
        assert!(l.evaluate([1.5, 0.5]).is_nan());
        let ct = CloughTocher2DInterpolator::new(&p, &plane).unwrap();
        for q in [[0.3, 0.7], [0.91, 0.05], [0.5, 0.5]] {
            assert!((ct.evaluate(q) - (2.0 * q[0] - q[1] + 0.5)).abs() < 1e-6);
        }
        let quad: Vec<f64> = p.iter().map(|q| q[0] * q[0] + q[0] * q[1]).collect();
        let ct = CloughTocher2DInterpolator::new(&p, &quad).unwrap();
        let ln = LinearNDInterpolator::new(&p, &quad).unwrap();
        let (mut e_ct, mut e_l) = (0.0f64, 0.0f64);
        for i in 1..20 {
            for j in 1..20 {
                let q = [i as f64 / 20.0, j as f64 / 20.0];
                let exact = q[0] * q[0] + q[0] * q[1];
                e_ct = e_ct.max((ct.evaluate(q) - exact).abs());
                e_l = e_l.max((ln.evaluate(q) - exact).abs());
            }
        }
        assert!(e_ct < e_l / 3.0, "Clough-Tocher {e_ct} vs linear {e_l}");
        let nn = griddata(&p, &quad, &[p[7]], GriddataMethod::Nearest).unwrap();
        assert_eq!(nn[0], quad[7]);
    }
}
