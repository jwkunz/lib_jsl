//! LGMRES: restarted GMRES augmented with approximations of the error from previous cycles
//! (Baker, Jessup and Manteuffel, 2005; SciPy's `lgmres`).

use super::iterative::{IterativeOptions, IterativeResult, LinearOperator};
use crate::error::{LinalgError, Result};

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// A stateful LGMRES solver: the augmentation vectors it keeps carry over from one call of
/// [`LgmresSolver::solve`] to the next, which is how SciPy's `newton_krylov` reuses them across
/// Newton iterations.
#[derive(Clone, Debug)]
pub struct LgmresSolver {
    /// Arnoldi steps per cycle, besides the augmentation vectors (SciPy's `inner_m`, default 30).
    pub inner_m: usize,
    /// The most augmentation vectors kept (SciPy's `outer_k`, default 3).
    pub outer_k: usize,
    /// Keep `A z` with each augmentation vector `z`, saving one product each when it is reused
    /// (SciPy's `store_outer_Av`, default true).
    pub store_outer_av: bool,
    /// Put the augmentation vectors before the Krylov vectors in each cycle rather than after
    /// (SciPy's `prepend_outer_v`, default false).
    pub prepend_outer_v: bool,
    outer: Vec<(Vec<f64>, Option<Vec<f64>>)>,
}

impl Default for LgmresSolver {
    fn default() -> Self {
        LgmresSolver { inner_m: 30, outer_k: 3, store_outer_av: true, prepend_outer_v: false, outer: Vec::new() }
    }
}

impl LgmresSolver {
    /// A solver with `inner_m` Krylov steps per cycle and at most `outer_k` augmentation
    /// vectors, storing `A z` and appending them (SciPy's other defaults).
    pub fn new(inner_m: usize, outer_k: usize) -> Self {
        LgmresSolver { inner_m, outer_k, ..Default::default() }
    }

    /// The augmentation vectors kept from previous cycles, newest last.
    pub fn augmentation_len(&self) -> usize {
        self.outer.len()
    }

    // ANCHOR: lgmres
    /// Solves `A x = b`, with an optional left preconditioner `M ≈ A⁻¹`. Each cycle runs flexible
    /// GMRES on `M A` from the preconditioned residual over `inner_m` Krylov vectors plus the
    /// kept error approximations `z` (with `M A z`), and the correction it finds becomes a new
    /// augmentation vector. A cycle stops early at a relative residual `ptol`, which adapts: it
    /// tightens when a cycle reaches it, and relaxes when one does not.
    /// `options.max_iterations` is the number of cycles (SciPy's `maxiter`, default 1000).
    pub fn solve(&mut self, a: &dyn LinearOperator, b: &[f64], x0: Option<&[f64]>, m: Option<&dyn LinearOperator>, options: IterativeOptions) -> Result<IterativeResult> {
        let (rows, cols) = a.shape();
        if rows != cols || b.len() != rows {
            return Err(LinalgError::Shape(format!("operator {rows} × {cols} with right-hand side of length {}", b.len())));
        }
        let n = b.len();
        let psolve = |v: &[f64]| m.map_or_else(|| v.to_vec(), |m| m.matvec(v));
        let mut x = x0.map_or(vec![0.0; n], |v| v.to_vec());
        let b_norm = norm(b);
        let tol = options.atol.max(options.rtol * b_norm);
        let mut matvecs = 0;
        let mut history = Vec::new();
        if b_norm == 0.0 {
            return Ok(IterativeResult { x: vec![0.0; n], iterations: 0, converged: true, residual_history: vec![0.0] });
        }
        let mut ptol_max_factor: f64 = 1.0;
        let max_cycles = options.max_iterations.unwrap_or(1000);
        let mut converged = false;
        for _ in 0..max_cycles {
            let ax = a.matvec(&x);
            matvecs += 1;
            let r: Vec<f64> = ax.iter().zip(b).map(|(p, q)| p - q).collect();
            let r_norm = norm(&r);
            history.push(r_norm);
            if r_norm <= tol {
                converged = true;
                break;
            }
            let mut v0: Vec<f64> = psolve(&r).iter().map(|v| -v).collect();
            let inner_res_0 = norm(&v0);
            if inner_res_0 == 0.0 {
                return Err(LinalgError::Shape("lgmres: the preconditioner returned a zero vector".into()));
            }
            v0.iter_mut().for_each(|v| *v /= inner_res_0);
            let ptol = ptol_max_factor.min(tol / r_norm);
            // Flexible GMRES over [Krylov vectors, augmentation vectors].
            let total = self.inner_m + self.outer.len();
            let mut vs: Vec<Vec<f64>> = vec![v0.clone()];
            let mut zs: Vec<Vec<f64>> = Vec::new();
            let mut h: Vec<Vec<f64>> = Vec::new(); // columns of the Hessenberg matrix
            let (mut cs, mut sn): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
            let mut g = vec![1.0];
            let mut pres = f64::NAN;
            let k_outer = self.outer.len();
            for j in 0..total {
                let (z, stored) = if self.prepend_outer_v && j < k_outer {
                    (self.outer[j].0.clone(), self.outer[j].1.clone())
                } else if self.prepend_outer_v && j == k_outer {
                    (v0.clone(), None)
                } else if !self.prepend_outer_v && j >= total - k_outer {
                    let o = &self.outer[j - (total - k_outer)];
                    (o.0.clone(), o.1.clone())
                } else {
                    (vs[vs.len() - 1].clone(), None)
                };
                let mut w = match stored {
                    Some(w) => w,
                    None => {
                        matvecs += 1;
                        psolve(&a.matvec(&z))
                    }
                };
                let w_norm = norm(&w);
                // Modified Gram-Schmidt against the basis.
                let mut col = vec![0.0; j + 2];
                for (i, v) in vs.iter().enumerate() {
                    col[i] = dot(v, &w);
                    w.iter_mut().zip(v).for_each(|(wi, vi)| *wi -= col[i] * vi);
                }
                col[j + 1] = norm(&w);
                // Written to treat a NaN norm as a breakdown, as SciPy does.
                let breakdown = col[j + 1].partial_cmp(&(f64::EPSILON * w_norm)) != Some(std::cmp::Ordering::Greater);
                if col[j + 1].is_finite() && col[j + 1] != 0.0 {
                    let s = col[j + 1];
                    w.iter_mut().for_each(|wi| *wi /= s);
                }
                vs.push(w);
                zs.push(z);
                // Givens rotations keep the least-squares problem triangular.
                for i in 0..j {
                    let t = cs[i] * col[i] + sn[i] * col[i + 1];
                    col[i + 1] = -sn[i] * col[i] + cs[i] * col[i + 1];
                    col[i] = t;
                }
                let denom = col[j].hypot(col[j + 1]);
                let (c, s) = if denom == 0.0 { (1.0, 0.0) } else { (col[j] / denom, col[j + 1] / denom) };
                cs.push(c);
                sn.push(s);
                col[j] = denom;
                col[j + 1] = 0.0;
                g.push(-s * g[j]);
                g[j] *= c;
                h.push(col);
                pres = g[j + 1].abs();
                if pres < ptol || breakdown {
                    break;
                }
            }
            let k = h.len();
            let mut y = vec![0.0; k];
            for i in (0..k).rev() {
                let s: f64 = (i + 1..k).map(|c| h[c][i] * y[c]).sum();
                y[i] = (g[i] - s) / h[i][i];
            }
            if y.iter().any(|v| !v.is_finite()) {
                return Ok(IterativeResult { x, iterations: matvecs, converged: false, residual_history: history });
            }
            y.iter_mut().for_each(|v| *v *= inner_res_0);
            ptol_max_factor = if pres > ptol { (1.5 * ptol_max_factor).min(1.0) } else { (0.25 * ptol_max_factor).max(1e-16) };
            let mut dx = vec![0.0; n];
            for (z, yc) in zs.iter().zip(&y) {
                dx.iter_mut().zip(z).for_each(|(d, zi)| *d += yc * zi);
            }
            let nx = norm(&dx);
            if nx > 0.0 {
                let w = self.store_outer_av.then(|| {
                    // M A dx = V_{k+1} H y, without a product. The columns hold R = G H after
                    // the rotations G, so H y = Gᵀ (R y), undoing the rotations in reverse.
                    let mut q = vec![0.0; k + 1];
                    for (i, qi) in q.iter_mut().enumerate().take(k) {
                        *qi = (i..k).map(|c| h[c][i] * y[c]).sum();
                    }
                    for i in (0..k).rev() {
                        let (a0, a1) = (q[i], q[i + 1]);
                        q[i] = cs[i] * a0 - sn[i] * a1;
                        q[i + 1] = sn[i] * a0 + cs[i] * a1;
                    }
                    let mut ax = vec![0.0; n];
                    for (v, qc) in vs.iter().zip(&q) {
                        ax.iter_mut().zip(v).for_each(|(s, vi)| *s += qc * vi);
                    }
                    ax.iter().map(|v| v / nx).collect::<Vec<f64>>()
                });
                self.outer.push((dx.iter().map(|v| v / nx).collect(), w));
            }
            while self.outer.len() > self.outer_k {
                self.outer.remove(0);
            }
            x.iter_mut().zip(&dx).for_each(|(xi, d)| *xi += d);
        }
        Ok(IterativeResult { x, iterations: matvecs, converged, residual_history: history })
    }
    // ANCHOR_END: lgmres
}

/// LGMRES with SciPy's defaults (`inner_m = 30`, `outer_k = 3`), from a fresh state.
/// `options.max_iterations` counts cycles (default 1000).
pub fn lgmres(a: &dyn LinearOperator, b: &[f64], x0: Option<&[f64]>, m: Option<&dyn LinearOperator>, options: IterativeOptions) -> Result<IterativeResult> {
    LgmresSolver::default().solve(a, b, x0, m, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::{CooMatrix, CsrMatrix, spsolve};

    fn convection(m: usize, gamma: f64) -> CsrMatrix {
        let mut coo = CooMatrix::new(m * m, m * m);
        for i in 0..m {
            for j in 0..m {
                let r = i * m + j;
                coo.push(r, r, 4.0);
                for (ok, c, v) in [(i > 0, r.wrapping_sub(m), -1.0 - gamma), (i + 1 < m, r + m, -1.0 + gamma), (j > 0, r.wrapping_sub(1), -1.0 - gamma), (j + 1 < m, r + 1, -1.0 + gamma)] {
                    if ok {
                        coo.push(r, c, v);
                    }
                }
            }
        }
        coo.to_csr().unwrap()
    }

    #[test]
    fn agrees_with_the_direct_solver_and_reuses_its_vectors() {
        let a = convection(20, 0.7);
        let b: Vec<f64> = (0..400).map(|k| 1.0 + 0.5 * (0.37 * k as f64).sin()).collect();
        let exact = spsolve(&a, &b).unwrap();
        let o = IterativeOptions { rtol: 1e-10, atol: 0.0, max_iterations: None };
        let r = lgmres(&a, &b, None, None, o).unwrap();
        assert!(r.converged);
        assert!(r.x.iter().zip(&exact).all(|(p, q)| (p - q).abs() < 1e-8));
        // A second solve with the augmentation vectors kept from the first still converges to the
        // right answer (whether they save work depends on the problem).
        let b2: Vec<f64> = b.iter().enumerate().map(|(k, v)| v + 1e-3 * (k as f64).cos()).collect();
        let mut solver = LgmresSolver::default();
        solver.solve(&a, &b, None, None, o).unwrap();
        assert_eq!(solver.augmentation_len(), 3);
        let warm = solver.solve(&a, &b2, None, None, o).unwrap();
        let exact2 = spsolve(&a, &b2).unwrap();
        assert!(warm.converged && warm.x.iter().zip(&exact2).all(|(p, q)| (p - q).abs() < 1e-8));
        assert!(lgmres(&a, &b[..10], None, None, o).is_err());
    }
}
