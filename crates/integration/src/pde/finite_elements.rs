//! Finite element methods with linear (P1) elements.
//!
//! * [`fem_1d`]: the two-point problem `−(p u')' + q u = f` on a mesh of `[a, b]` with Dirichlet
//!   or natural (flux) end conditions, by Galerkin assembly with two-point Gauss quadrature.
//! * [`fem_poisson_2d`]: `−∇·(k ∇u) = f` on a triangle mesh with Dirichlet data on chosen nodes,
//!   assembled into a sparse matrix and solved by Jacobi-preconditioned conjugate gradients.
//!   [`TriangleMesh::rectangle`] builds a structured mesh of a rectangle.
//!
//! Linear elements converge at second order in the L² and nodal errors and first order in the
//! energy norm.
//!
//! See the book entry [Finite element methods](https://jwkunz.github.io/lib_jsl/book/integration/finite-elements.html).

use lib_jsl_core::ErrorsJSL;
use std::collections::BTreeMap;

/// An end condition for [`fem_1d`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EndCondition {
    /// `u = value`.
    Dirichlet(f64),
    /// The outward flux `p u' · n = value` (natural condition; zero for insulated ends).
    Flux(f64),
}

/// Solves `−(p u')' + q u = f` on the increasing `nodes`, returning `u` at the nodes.
pub fn fem_1d<P, Q, F>(p: P, q: Q, f: F, nodes: &[f64], left: EndCondition, right: EndCondition) -> Result<Vec<f64>, ErrorsJSL>
where
    P: Fn(f64) -> f64,
    Q: Fn(f64) -> f64,
    F: Fn(f64) -> f64,
{
    let n = nodes.len();
    if n < 2 || nodes.windows(2).any(|w| w[1] <= w[0]) {
        return Err(ErrorsJSL::InvalidInputRange("Need at least two strictly increasing nodes."));
    }
    // Tridiagonal stiffness + mass matrix and load vector.
    let (mut lower, mut diagonal, mut upper, mut load) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let gauss = [(0.5 - 0.5 / 3f64.sqrt(), 0.5), (0.5 + 0.5 / 3f64.sqrt(), 0.5)];
    for e in 0..n - 1 {
        let (x0, x1) = (nodes[e], nodes[e + 1]);
        let h = x1 - x0;
        let mut k = [[0.0; 2]; 2];
        let mut b = [0.0; 2];
        for &(s, w) in &gauss {
            let x = x0 + s * h;
            let phi = [1.0 - s, s];
            let dphi = [-1.0 / h, 1.0 / h];
            let (pv, qv, fv) = (p(x), q(x), f(x));
            for i in 0..2 {
                b[i] += w * h * fv * phi[i];
                for j in 0..2 {
                    k[i][j] += w * h * (pv * dphi[i] * dphi[j] + qv * phi[i] * phi[j]);
                }
            }
        }
        diagonal[e] += k[0][0];
        diagonal[e + 1] += k[1][1];
        upper[e] += k[0][1];
        lower[e + 1] += k[1][0];
        load[e] += b[0];
        load[e + 1] += b[1];
    }
    for (end, condition) in [(0, left), (n - 1, right)] {
        match condition {
            EndCondition::Flux(g) => load[end] += g,
            EndCondition::Dirichlet(value) => {
                // Replace the row by u = value and move the column to the right-hand side.
                if end == 0 {
                    load[1] -= lower[1] * value;
                    lower[1] = 0.0;
                    upper[0] = 0.0;
                } else {
                    load[n - 2] -= upper[n - 2] * value;
                    upper[n - 2] = 0.0;
                    lower[n - 1] = 0.0;
                }
                diagonal[end] = 1.0;
                load[end] = value;
            }
        }
    }
    super::finite_differences::solve_tridiagonal(&lower, &diagonal, &upper, &load)
}

/// A triangle mesh: node coordinates and counter-clockwise (or any consistent) node triples.
#[derive(Debug, Clone)]
pub struct TriangleMesh {
    /// The node coordinates.
    pub nodes: Vec<[f64; 2]>,
    /// Each triangle's three node indices.
    pub triangles: Vec<[usize; 3]>,
}

impl TriangleMesh {
    /// The rectangle `[0, width] × [0, height]` divided into `nx × ny` cells, each split into two
    /// triangles. Node `(i, j)` has index `j (nx + 1) + i`.
    pub fn rectangle(width: f64, height: f64, nx: usize, ny: usize) -> Self {
        let nodes = (0..=ny).flat_map(|j| (0..=nx).map(move |i| [width * i as f64 / nx as f64, height * j as f64 / ny as f64])).collect();
        let id = |i: usize, j: usize| j * (nx + 1) + i;
        let triangles = (0..ny).flat_map(|j| (0..nx).flat_map(move |i| [[id(i, j), id(i + 1, j), id(i + 1, j + 1)], [id(i, j), id(i + 1, j + 1), id(i, j + 1)]])).collect();
        TriangleMesh { nodes, triangles }
    }

    /// Indices of nodes on the boundary of the bounding box (for rectangle meshes, the boundary).
    pub fn box_boundary_nodes(&self) -> Vec<usize> {
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for p in &self.nodes {
            for d in 0..2 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        let tol = 1e-12 * (hi[0] - lo[0]).max(hi[1] - lo[1]);
        (0..self.nodes.len()).filter(|&i| (0..2).any(|d| (self.nodes[i][d] - lo[d]).abs() < tol || (self.nodes[i][d] - hi[d]).abs() < tol)).collect()
    }
}

/// Solves `−∇·(k ∇u) = f` on the mesh with `u = g` at `dirichlet_nodes` (natural zero-flux
/// conditions elsewhere), returning `u` at every node.
pub fn fem_poisson_2d<K, F, G>(mesh: &TriangleMesh, k: K, f: F, g: G, dirichlet_nodes: &[usize]) -> Result<Vec<f64>, ErrorsJSL>
where
    K: Fn(f64, f64) -> f64,
    F: Fn(f64, f64) -> f64,
    G: Fn(f64, f64) -> f64,
{
    let n = mesh.nodes.len();
    if mesh.triangles.iter().flatten().any(|&v| v >= n) || dirichlet_nodes.iter().any(|&v| v >= n) {
        return Err(ErrorsJSL::InvalidInputRange("Mesh or boundary index out of range."));
    }
    if dirichlet_nodes.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("Need at least one Dirichlet node for a unique solution."));
    }
    let mut rows: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n];
    let mut load = vec![0.0; n];
    for t in &mesh.triangles {
        let [p0, p1, p2] = [mesh.nodes[t[0]], mesh.nodes[t[1]], mesh.nodes[t[2]]];
        let area2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);
        if area2.abs() < 1e-300 {
            return Err(ErrorsJSL::InvalidInputRange("Degenerate triangle in the mesh."));
        }
        let area = 0.5 * area2.abs();
        // Gradients of the barycentric basis functions.
        let grad = [
            [(p1[1] - p2[1]) / area2, (p2[0] - p1[0]) / area2],
            [(p2[1] - p0[1]) / area2, (p0[0] - p2[0]) / area2],
            [(p0[1] - p1[1]) / area2, (p1[0] - p0[0]) / area2],
        ];
        let centroid = [(p0[0] + p1[0] + p2[0]) / 3.0, (p0[1] + p1[1] + p2[1]) / 3.0];
        let kc = k(centroid[0], centroid[1]);
        // Load by the three-edge-midpoint rule (exact for quadratics).
        let mids = [[(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0], [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0], [(p2[0] + p0[0]) / 2.0, (p2[1] + p0[1]) / 2.0]];
        let fm: Vec<f64> = mids.iter().map(|m| f(m[0], m[1])).collect();
        // φ_i at the midpoints: ½ on the two edges touching node i, 0 on the opposite edge.
        let phi_at_mid = [[0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]];
        for a in 0..3 {
            load[t[a]] += area / 3.0 * (0..3).map(|m| fm[m] * phi_at_mid[a][m]).sum::<f64>();
            for b in 0..3 {
                *rows[t[a]].entry(t[b]).or_insert(0.0) += kc * area * (grad[a][0] * grad[b][0] + grad[a][1] * grad[b][1]);
            }
        }
    }
    // Eliminate the Dirichlet nodes symmetrically.
    let mut fixed = vec![None; n];
    for &d in dirichlet_nodes {
        let p = mesh.nodes[d];
        fixed[d] = Some(g(p[0], p[1]));
    }
    for i in 0..n {
        if fixed[i].is_some() {
            continue;
        }
        let shift: f64 = rows[i].iter().filter_map(|(&j, &v)| fixed[j].map(|gj| v * gj)).sum();
        load[i] -= shift;
    }
    let free: Vec<usize> = (0..n).filter(|&i| fixed[i].is_none()).collect();
    let mut position = vec![usize::MAX; n];
    free.iter().enumerate().for_each(|(k, &i)| position[i] = k);
    let csr: Vec<Vec<(usize, f64)>> = free.iter().map(|&i| rows[i].iter().filter(|(j, _)| position[**j] != usize::MAX).map(|(&j, &v)| (position[j], v)).collect()).collect();
    let diagonal: Vec<f64> = free.iter().map(|&i| rows[i][&i]).collect();
    let b: Vec<f64> = free.iter().map(|&i| load[i]).collect();
    // Jacobi-preconditioned CG on the symmetric positive-definite reduced system.
    let solution = preconditioned_cg(&csr, &diagonal, &b, 1e-12, 20 * free.len().max(50))?;
    let mut u: Vec<f64> = fixed.iter().map(|v| v.unwrap_or(0.0)).collect();
    free.iter().zip(solution).for_each(|(&i, v)| u[i] = v);
    Ok(u)
}

fn preconditioned_cg(a: &[Vec<(usize, f64)>], diagonal: &[f64], b: &[f64], tolerance: f64, max_iterations: usize) -> Result<Vec<f64>, ErrorsJSL> {
    let apply = |v: &[f64]| -> Vec<f64> { a.iter().map(|row| row.iter().map(|&(j, w)| w * v[j]).sum()).collect() };
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();
    let mut x = vec![0.0; b.len()];
    let mut r = b.to_vec();
    let mut z: Vec<f64> = r.iter().zip(diagonal).map(|(ri, d)| ri / d).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let target = tolerance * dot(b, b).sqrt().max(1e-300);
    for _ in 0..max_iterations {
        if dot(&r, &r).sqrt() <= target {
            return Ok(x);
        }
        let ap = apply(&p);
        let alpha = rz / dot(&p, &ap);
        x.iter_mut().zip(&p).for_each(|(xi, pi)| *xi += alpha * pi);
        r.iter_mut().zip(&ap).for_each(|(ri, api)| *ri -= alpha * api);
        z = r.iter().zip(diagonal).map(|(ri, d)| ri / d).collect();
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        p.iter_mut().zip(&z).for_each(|(pi, zi)| *pi = zi + beta * *pi);
        rz = rz_new;
    }
    if dot(&r, &r).sqrt() <= 1e3 * target { Ok(x) } else { Err(ErrorsJSL::RuntimeError("Conjugate gradients did not converge.")) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn one_dimensional_problems_converge_and_handle_both_end_conditions() {
        // −u'' = π² sin(πx), u(0) = u(1) = 0: linear elements are nodally exact up to quadrature.
        let error = |n: usize| {
            let nodes: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
            let u = fem_1d(|_| 1.0, |_| 0.0, |x| PI * PI * (PI * x).sin(), &nodes, EndCondition::Dirichlet(0.0), EndCondition::Dirichlet(0.0)).unwrap();
            nodes.iter().zip(&u).map(|(x, v)| (v - (PI * x).sin()).abs()).fold(0.0, f64::max)
        };
        let (e1, e2) = (error(10), error(20));
        assert!(e2 < 1e-3 && e1 / e2 > 3.5, "{e1} {e2}");
        // Variable coefficients with a flux end: −((1 + x) u')' + u = f for u = x², with
        // u(0) = 0 and the outward flux (1 + x) u' = 4 at x = 1.
        let f = |x: f64| -(2.0 + 4.0 * x) + x * x;
        let nodes: Vec<f64> = (0..=40).map(|i| i as f64 / 40.0).collect();
        let u = fem_1d(|x| 1.0 + x, |_| 1.0, f, &nodes, EndCondition::Dirichlet(0.0), EndCondition::Flux(4.0)).unwrap();
        assert!(nodes.iter().zip(&u).all(|(x, v)| (v - x * x).abs() < 1e-3), "{:?}", &u[38..]);
        assert!(fem_1d(|_| 1.0, |_| 0.0, |_| 0.0, &[0.0, 0.0], EndCondition::Dirichlet(0.0), EndCondition::Dirichlet(0.0)).is_err());
    }

    #[test]
    fn two_dimensional_poisson_converges_at_second_order() {
        let exact = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin() + x - 2.0 * y;
        let f = |x: f64, y: f64| 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
        let error = |n: usize| {
            let mesh = TriangleMesh::rectangle(1.0, 1.0, n, n);
            let u = fem_poisson_2d(&mesh, |_, _| 1.0, f, exact, &mesh.box_boundary_nodes()).unwrap();
            mesh.nodes.iter().zip(&u).map(|(p, v)| (v - exact(p[0], p[1])).abs()).fold(0.0, f64::max)
        };
        let (coarse, fine) = (error(8), error(16));
        assert!(fine < 0.01 && coarse / fine > 3.3, "{coarse} {fine}");
        // A linear solution is reproduced exactly with any conductivity constant.
        let mesh = TriangleMesh::rectangle(2.0, 1.0, 6, 3);
        let u = fem_poisson_2d(&mesh, |_, _| 3.0, |_, _| 0.0, |x, y| 1.0 + 2.0 * x - y, &mesh.box_boundary_nodes()).unwrap();
        assert!(mesh.nodes.iter().zip(&u).all(|(p, v)| (v - (1.0 + 2.0 * p[0] - p[1])).abs() < 1e-10));
        assert!(fem_poisson_2d(&mesh, |_, _| 1.0, |_, _| 1.0, |_, _| 0.0, &[]).is_err());
    }
}
