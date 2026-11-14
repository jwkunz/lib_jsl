//! Krylov-subspace solvers for large sparse systems, following `scipy.sparse.linalg`: conjugate
//! gradients (`cg`) for symmetric positive definite matrices, MINRES for symmetric indefinite
//! ones, restarted GMRES and BiCGSTAB for general matrices, and LSQR for least squares. Each
//! works with any [`LinearOperator`], and the square solvers accept a preconditioner `M ≈ A⁻¹`.

use super::formats::{CscMatrix, CsrMatrix};
use crate::error::{LinalgError, Result};
use ndarray::Array2;

/// A matrix seen only through its products with vectors (SciPy's `LinearOperator`).
pub trait LinearOperator {
    /// `(rows, columns)`.
    fn shape(&self) -> (usize, usize);
    /// `A x`.
    fn matvec(&self, x: &[f64]) -> Vec<f64>;
    /// `Aᵀ x` (needed by LSQR).
    fn rmatvec(&self, x: &[f64]) -> Vec<f64>;
}

impl LinearOperator for CsrMatrix {
    fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
    fn matvec(&self, x: &[f64]) -> Vec<f64> {
        CsrMatrix::matvec(self, x)
    }
    fn rmatvec(&self, x: &[f64]) -> Vec<f64> {
        CsrMatrix::rmatvec(self, x)
    }
}

impl LinearOperator for CscMatrix {
    fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }
    fn matvec(&self, x: &[f64]) -> Vec<f64> {
        CscMatrix::matvec(self, x)
    }
    fn rmatvec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.ncols];
        for (j, yj) in y.iter_mut().enumerate() {
            *yj = (self.indptr[j]..self.indptr[j + 1]).map(|p| self.data[p] * x[self.indices[p]]).sum();
        }
        y
    }
}

impl LinearOperator for Array2<f64> {
    fn shape(&self) -> (usize, usize) {
        self.dim()
    }
    fn matvec(&self, x: &[f64]) -> Vec<f64> {
        self.dot(&ndarray::ArrayView1::from(x)).to_vec()
    }
    fn rmatvec(&self, x: &[f64]) -> Vec<f64> {
        self.t().dot(&ndarray::ArrayView1::from(x)).to_vec()
    }
}

/// A linear operator defined by closures.
pub struct FnOperator<F: Fn(&[f64]) -> Vec<f64>, G: Fn(&[f64]) -> Vec<f64>> {
    pub shape: (usize, usize),
    pub matvec: F,
    pub rmatvec: G,
}

impl<F: Fn(&[f64]) -> Vec<f64>, G: Fn(&[f64]) -> Vec<f64>> LinearOperator for FnOperator<F, G> {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
    fn matvec(&self, x: &[f64]) -> Vec<f64> {
        (self.matvec)(x)
    }
    fn rmatvec(&self, x: &[f64]) -> Vec<f64> {
        (self.rmatvec)(x)
    }
}

/// The Jacobi (diagonal) preconditioner `M = diag(A)⁻¹`.
pub struct Jacobi {
    inverse_diagonal: Vec<f64>,
}

impl Jacobi {
    pub fn new(a: &CsrMatrix) -> Result<Self> {
        let d = a.diagonal();
        if d.contains(&0.0) {
            return Err(LinalgError::Singular(0));
        }
        Ok(Jacobi { inverse_diagonal: d.iter().map(|v| 1.0 / v).collect() })
    }
}

impl LinearOperator for Jacobi {
    fn shape(&self) -> (usize, usize) {
        (self.inverse_diagonal.len(), self.inverse_diagonal.len())
    }
    fn matvec(&self, x: &[f64]) -> Vec<f64> {
        x.iter().zip(&self.inverse_diagonal).map(|(a, b)| a * b).collect()
    }
    fn rmatvec(&self, x: &[f64]) -> Vec<f64> {
        self.matvec(x)
    }
}

/// Options for the iterative solvers. Convergence when `‖b − A x‖ ≤ max(rtol ‖b‖, atol)`
/// (SciPy's defaults: `rtol = 10⁻⁵`, `atol = 0`).
#[derive(Clone, Copy, Debug)]
pub struct IterativeOptions {
    pub rtol: f64,
    pub atol: f64,
    /// Default: 10 n (GMRES: 10 n inner iterations).
    pub max_iterations: Option<usize>,
}

impl Default for IterativeOptions {
    fn default() -> Self {
        IterativeOptions { rtol: 1e-5, atol: 0.0, max_iterations: None }
    }
}

/// The result of an iterative solve.
#[derive(Clone, Debug)]
pub struct IterativeResult {
    pub x: Vec<f64>,
    /// Matrix-vector products with `A`.
    pub iterations: usize,
    pub converged: bool,
    /// The residual norm after each iteration, starting with the initial one.
    pub residual_history: Vec<f64>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    y.iter_mut().zip(x).for_each(|(yi, xi)| *yi += a * xi);
}

fn residual(a: &dyn LinearOperator, b: &[f64], x: &[f64]) -> Vec<f64> {
    let ax = a.matvec(x);
    b.iter().zip(&ax).map(|(p, q)| p - q).collect()
}

fn setup(a: &dyn LinearOperator, b: &[f64], x0: Option<&[f64]>) -> Result<Vec<f64>> {
    let (m, n) = a.shape();
    if m != n || b.len() != n {
        return Err(LinalgError::Shape(format!("operator {m} × {n} with right-hand side of length {}", b.len())));
    }
    Ok(x0.map_or(vec![0.0; n], |v| v.to_vec()))
}

fn precondition(m: Option<&dyn LinearOperator>, r: &[f64]) -> Vec<f64> {
    m.map_or_else(|| r.to_vec(), |m| m.matvec(r))
}

// ANCHOR: cg
/// Conjugate gradients for symmetric positive definite `A` (SciPy's `cg`): each step minimises
/// the `A`-norm of the error over a growing Krylov subspace, with search directions that are
/// `A`-conjugate. With `M`, the preconditioned residual `z = M r` replaces `r` in the recurrences.
pub fn cg(a: &dyn LinearOperator, b: &[f64], x0: Option<&[f64]>, m: Option<&dyn LinearOperator>, options: IterativeOptions) -> Result<IterativeResult> {
    let mut x = setup(a, b, x0)?;
    let tol = (options.rtol * norm(b)).max(options.atol);
    let max_it = options.max_iterations.unwrap_or(10 * b.len());
    let mut r = residual(a, b, &x);
    let mut history = vec![norm(&r)];
    let mut z = precondition(m, &r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut it = 0;
    while history[history.len() - 1] > tol && it < max_it {
        let ap = a.matvec(&p);
        let alpha = rz / dot(&p, &ap);
        axpy(&mut x, alpha, &p);
        axpy(&mut r, -alpha, &ap);
        it += 1;
        history.push(norm(&r));
        z = precondition(m, &r);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        p = z.iter().zip(&p).map(|(zi, pi)| zi + beta * pi).collect();
    }
    let converged = history[history.len() - 1] <= tol;
    Ok(IterativeResult { x, iterations: it, converged, residual_history: history })
}
// ANCHOR_END: cg

/// BiCGSTAB for general square `A` (SciPy's `bicgstab`): biconjugate gradients stabilised by a
/// one-step minimal-residual smoothing, two products with `A` per iteration.
pub fn bicgstab(a: &dyn LinearOperator, b: &[f64], x0: Option<&[f64]>, m: Option<&dyn LinearOperator>, options: IterativeOptions) -> Result<IterativeResult> {
    let mut x = setup(a, b, x0)?;
    let tol = (options.rtol * norm(b)).max(options.atol);
    let max_it = options.max_iterations.unwrap_or(10 * b.len());
    let mut r = residual(a, b, &x);
    let rhat = r.clone();
    let mut history = vec![norm(&r)];
    let (mut rho, mut alpha, mut omega) = (1.0, 1.0, 1.0);
    let n = b.len();
    let (mut v, mut p) = (vec![0.0; n], vec![0.0; n]);
    let mut it = 0;
    while history[history.len() - 1] > tol && it < max_it {
        let rho_new = dot(&rhat, &r);
        if rho_new == 0.0 {
            break;
        }
        let beta = rho_new / rho * alpha / omega;
        rho = rho_new;
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
        let phat = precondition(m, &p);
        v = a.matvec(&phat);
        alpha = rho / dot(&rhat, &v);
        let s: Vec<f64> = r.iter().zip(&v).map(|(ri, vi)| ri - alpha * vi).collect();
        it += 1;
        if norm(&s) <= tol {
            axpy(&mut x, alpha, &phat);
            history.push(norm(&s));
            r = s;
            break;
        }
        let shat = precondition(m, &s);
        let t = a.matvec(&shat);
        omega = dot(&t, &s) / dot(&t, &t);
        axpy(&mut x, alpha, &phat);
        axpy(&mut x, omega, &shat);
        r = s.iter().zip(&t).map(|(si, ti)| si - omega * ti).collect();
        history.push(norm(&r));
        if omega == 0.0 {
            break;
        }
    }
    let converged = norm(&r) <= tol;
    Ok(IterativeResult { x, iterations: it, converged, residual_history: history })
}

// ANCHOR: gmres
/// Restarted GMRES(`restart`) for general square `A` (SciPy's `gmres`, default restart 20): the
/// Arnoldi process builds an orthonormal Krylov basis, and Givens rotations keep the small least-
/// squares problem for the residual-minimising combination triangular, which gives the residual
/// norm at every step without forming `x`. Right preconditioning, `A M y = b`, `x = M y`.
pub fn gmres(a: &dyn LinearOperator, b: &[f64], x0: Option<&[f64]>, m: Option<&dyn LinearOperator>, restart: usize, options: IterativeOptions) -> Result<IterativeResult> {
    let mut x = setup(a, b, x0)?;
    let n = b.len();
    let restart = restart.clamp(1, n.max(1));
    let tol = (options.rtol * norm(b)).max(options.atol);
    let max_it = options.max_iterations.unwrap_or(10 * n);
    let mut r = residual(a, b, &x);
    let mut history = vec![norm(&r)];
    let mut it = 0;
    while history[history.len() - 1] > tol && it < max_it {
        let beta = norm(&r);
        let mut basis: Vec<Vec<f64>> = vec![r.iter().map(|v| v / beta).collect()];
        let mut h = vec![vec![0.0; restart]; restart + 1];
        let (mut cs, mut sn) = (vec![0.0; restart], vec![0.0; restart]);
        let mut g = vec![0.0; restart + 1];
        g[0] = beta;
        let mut k_done = 0;
        for k in 0..restart {
            let mut w = a.matvec(&precondition(m, &basis[k]));
            // Modified Gram-Schmidt.
            for (j, vj) in basis.iter().enumerate() {
                h[j][k] = dot(&w, vj);
                axpy(&mut w, -h[j][k], vj);
            }
            h[k + 1][k] = norm(&w);
            // Apply the previous rotations to the new column, then a new rotation.
            for j in 0..k {
                let t = cs[j] * h[j][k] + sn[j] * h[j + 1][k];
                h[j + 1][k] = -sn[j] * h[j][k] + cs[j] * h[j + 1][k];
                h[j][k] = t;
            }
            let denom = h[k][k].hypot(h[k + 1][k]);
            cs[k] = h[k][k] / denom;
            sn[k] = h[k + 1][k] / denom;
            h[k][k] = denom;
            h[k + 1][k] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k] *= cs[k];
            it += 1;
            k_done = k + 1;
            history.push(g[k + 1].abs());
            if g[k + 1].abs() <= tol || it >= max_it {
                break;
            }
            let hk1 = h_norm(&w);
            if hk1 == 0.0 {
                break;
            }
            basis.push(w.iter().map(|v| v / hk1).collect());
        }
        // Back-substitution for y, then x += M V y.
        let mut y = vec![0.0; k_done];
        for i in (0..k_done).rev() {
            let s: f64 = (i + 1..k_done).map(|j| h[i][j] * y[j]).sum();
            y[i] = (g[i] - s) / h[i][i];
        }
        let mut update = vec![0.0; n];
        for (j, yj) in y.iter().enumerate() {
            axpy(&mut update, *yj, &basis[j]);
        }
        let dx = precondition(m, &update);
        axpy(&mut x, 1.0, &dx);
        r = residual(a, b, &x);
        let true_norm = norm(&r);
        let last = history.len() - 1;
        history[last] = true_norm;
    }
    let converged = history[history.len() - 1] <= tol;
    Ok(IterativeResult { x, iterations: it, converged, residual_history: history })
}
// ANCHOR_END: gmres

fn h_norm(w: &[f64]) -> f64 {
    norm(w)
}

/// MINRES for symmetric (possibly indefinite) `A` (Paige and Saunders; SciPy's `minres`):
/// the Lanczos process with Givens rotations, minimising `‖b − A x‖` over the Krylov subspace with
/// short recurrences.
pub fn minres(a: &dyn LinearOperator, b: &[f64], x0: Option<&[f64]>, options: IterativeOptions) -> Result<IterativeResult> {
    let mut x = setup(a, b, x0)?;
    let n = b.len();
    let tol = (options.rtol * norm(b)).max(options.atol);
    let max_it = options.max_iterations.unwrap_or(5 * n);
    let r0 = residual(a, b, &x);
    let beta1 = norm(&r0);
    let mut history = vec![beta1];
    if beta1 <= tol {
        return Ok(IterativeResult { x, iterations: 0, converged: true, residual_history: history });
    }
    // Paige and Saunders' recurrences, as in SciPy's `minres` without a preconditioner.
    let (mut r1, mut r2) = (r0.clone(), r0);
    let (mut oldb, mut beta) = (0.0, beta1);
    let (mut dbar, mut epsln, mut phibar) = (0.0, 0.0, beta1);
    let (mut cs, mut sn) = (-1.0, 0.0);
    let (mut w, mut w2) = (vec![0.0; n], vec![0.0; n]);
    let mut it = 0;
    while phibar > tol && it < max_it {
        it += 1;
        let v: Vec<f64> = r2.iter().map(|t| t / beta).collect();
        let mut y = a.matvec(&v);
        if it >= 2 {
            axpy(&mut y, -beta / oldb, &r1);
        }
        let alfa = dot(&v, &y);
        axpy(&mut y, -alfa / beta, &r2);
        r1 = r2;
        r2 = y;
        oldb = beta;
        beta = norm(&r2);
        let oldeps = epsln;
        let delta = cs * dbar + sn * alfa;
        let gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;
        let gamma = gbar.hypot(beta).max(f64::MIN_POSITIVE);
        cs = gbar / gamma;
        sn = beta / gamma;
        let phi = cs * phibar;
        phibar *= sn;
        let w1 = std::mem::replace(&mut w2, w.clone());
        w = (0..n).map(|i| (v[i] - oldeps * w1[i] - delta * w2[i]) / gamma).collect();
        axpy(&mut x, phi, &w);
        history.push(phibar);
        if beta == 0.0 {
            break;
        }
    }
    let true_res = norm(&residual(a, b, &x));
    let last = history.len() - 1;
    history[last] = true_res;
    Ok(IterativeResult { x, iterations: it, converged: true_res <= tol * (1.0 + 1e-6), residual_history: history })
}

// ANCHOR: lsqr
/// LSQR for `min ‖A x − b‖` with any rectangular `A` (Paige and Saunders; SciPy's `lsqr`):
/// Golub-Kahan bidiagonalisation of `A`, mathematically CG on the normal equations but stable. It
/// stops when `‖r‖ ≤ btol ‖b‖ + atol ‖A‖ ‖x‖` (a consistent system) or
/// `‖Aᵀr‖ ≤ atol ‖A‖ ‖r‖` (a least-squares solution), with SciPy's `atol = btol = 10⁻⁶` by
/// default (`options.rtol` is used for both).
pub fn lsqr(a: &dyn LinearOperator, b: &[f64], options: IterativeOptions) -> Result<IterativeResult> {
    let (m, n) = a.shape();
    if b.len() != m {
        return Err(LinalgError::Shape(format!("operator {m} × {n} with right-hand side of length {}", b.len())));
    }
    let tol = options.rtol;
    let max_it = options.max_iterations.unwrap_or(2 * n);
    let mut x = vec![0.0; n];
    let mut u = b.to_vec();
    let mut beta = norm(&u);
    let bnorm = beta;
    let mut history = vec![beta];
    if beta == 0.0 {
        return Ok(IterativeResult { x, iterations: 0, converged: true, residual_history: history });
    }
    u.iter_mut().for_each(|v| *v /= beta);
    let mut v = a.rmatvec(&u);
    let mut alpha = norm(&v);
    v.iter_mut().for_each(|t| *t /= alpha);
    let mut w = v.clone();
    let (mut phibar, mut rhobar) = (beta, alpha);
    let mut anorm2 = 0.0;
    let mut it = 0;
    let mut converged = false;
    while it < max_it {
        it += 1;
        let av = a.matvec(&v);
        u = av.iter().zip(&u).map(|(p, q)| p - alpha * q).collect();
        beta = norm(&u);
        if beta > 0.0 {
            u.iter_mut().for_each(|t| *t /= beta);
            let atu = a.rmatvec(&u);
            v = atu.iter().zip(&v).map(|(p, q)| p - beta * q).collect();
            alpha = norm(&v);
            if alpha > 0.0 {
                v.iter_mut().for_each(|t| *t /= alpha);
            }
        }
        anorm2 += alpha * alpha + beta * beta;
        let rho = rhobar.hypot(beta);
        let (c, s) = (rhobar / rho, beta / rho);
        let theta = s * alpha;
        rhobar = -c * alpha;
        let phi = c * phibar;
        phibar *= s;
        axpy(&mut x, phi / rho, &w);
        w = v.iter().zip(&w).map(|(vi, wi)| vi - theta / rho * wi).collect();
        let rnorm = phibar;
        history.push(rnorm);
        let arnorm = phibar * alpha * c.abs();
        let anorm = anorm2.sqrt();
        let xnorm = norm(&x);
        if rnorm <= tol * bnorm + tol * anorm * xnorm || arnorm <= tol * anorm * rnorm || rnorm == 0.0 {
            converged = true;
            break;
        }
    }
    Ok(IterativeResult { x, iterations: it, converged, residual_history: history })
}
// ANCHOR_END: lsqr
