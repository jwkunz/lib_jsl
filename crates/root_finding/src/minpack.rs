//! MINPACK's Levenberg-Marquardt method (`lmder` and `lmdif`, Moré 1978), as SciPy's
//! `optimize.leastsq` and `optimize.root(method="lm")` call it: nonlinear least squares
//! `min ‖F(x)‖²`, `F: ℝⁿ → ℝᵐ`, `m ≥ n`, and so also the roots of square systems.
//!
//! See the book entry [Levenberg-Marquardt for systems, mixed-integer programming and truncated Newton](https://jwkunz.github.io/lib_jsl/book/optimization/milp-tnc.html#minpacks-levenberg-marquardt).

// MINPACK's loops over columns and rows read best as index loops.
#![allow(clippy::needless_range_loop)]

use crate::nonlinear_systems::SystemRootResult;
use lib_jsl_core::ErrorsJSL;

/// An analytic Jacobian for [`leastsq`] and [`root_lm`]: the `m × n` matrix `∂Fᵢ/∂xⱼ` by rows.
pub type JacobianFn<'a> = &'a mut dyn FnMut(&[f64]) -> Vec<Vec<f64>>;

/// Options for [`leastsq`] and [`root_lm`], with SciPy's `leastsq` defaults.
#[derive(Clone, Copy, Debug)]
pub struct LmOptions {
    /// Stop when the relative reduction in the sum of squares is at most this (MINPACK info 1).
    pub ftol: f64,
    /// Stop when the relative change in `x` is at most this (info 2).
    pub xtol: f64,
    /// Stop when the cosine of the angle between `F` and any column of the Jacobian is at most
    /// this (info 4).
    pub gtol: f64,
    /// The largest number of evaluations of `F`; `None` for SciPy's `100(n + 1)` with a
    /// Jacobian and `200(n + 1)` without.
    pub maxfev: Option<usize>,
    /// The initial step bound is `factor ‖D x‖`.
    pub factor: f64,
    /// The relative step for forward-difference Jacobians, `√max(epsfcn, ε)`.
    pub epsfcn: f64,
}

impl Default for LmOptions {
    fn default() -> Self {
        LmOptions { ftol: 1.49012e-8, xtol: 1.49012e-8, gtol: 0.0, maxfev: None, factor: 100.0, epsfcn: 0.0 }
    }
}

/// The result of [`leastsq`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LmResult {
    pub x: Vec<f64>,
    /// `F(x)` at the solution.
    pub fvec: Vec<f64>,
    /// Evaluations of `F`, including those for finite-difference Jacobians.
    pub nfev: usize,
    /// Evaluations of the Jacobian (0 with finite differences).
    pub njev: usize,
    /// MINPACK's `info`: 1 ftol, 2 xtol, 3 both, 4 gtol, 5 the evaluation limit, 6–8 a tolerance
    /// too small to improve on.
    pub info: i32,
}

fn enorm(v: &[f64]) -> f64 {
    // MINPACK guards against overflow; a scaled sum of squares does the same.
    let s = v.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    if s == 0.0 || !s.is_finite() {
        return s;
    }
    s * v.iter().map(|x| (x / s) * (x / s)).sum::<f64>().sqrt()
}

/// Householder QR with column pivoting (`qrfac`): `a` (m × n, rows) is overwritten by the
/// Householder vectors below the diagonal and `R` above it; returns the permutation, `R`'s
/// diagonal and the column norms of the original matrix.
fn qrfac(a: &mut [Vec<f64>], m: usize, n: usize) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    let eps = f64::EPSILON;
    let col = |a: &[Vec<f64>], j: usize, from: usize| -> Vec<f64> { (from..m).map(|i| a[i][j]).collect() };
    let acnorm: Vec<f64> = (0..n).map(|j| enorm(&col(a, j, 0))).collect();
    let mut rdiag = acnorm.clone();
    let mut wa = acnorm.clone();
    let mut ipvt: Vec<usize> = (0..n).collect();
    for j in 0..m.min(n) {
        let kmax = (j..n).fold(j, |b, k| if rdiag[k] > rdiag[b] { k } else { b });
        if kmax != j {
            for row in a.iter_mut() {
                row.swap(j, kmax);
            }
            rdiag[kmax] = rdiag[j];
            wa[kmax] = wa[j];
            ipvt.swap(j, kmax);
        }
        let mut ajnorm = enorm(&col(a, j, j));
        if ajnorm != 0.0 {
            if a[j][j] < 0.0 {
                ajnorm = -ajnorm;
            }
            for i in j..m {
                a[i][j] /= ajnorm;
            }
            a[j][j] += 1.0;
            for k in j + 1..n {
                let sum: f64 = (j..m).map(|i| a[i][j] * a[i][k]).sum();
                let temp = sum / a[j][j];
                for i in j..m {
                    a[i][k] -= temp * a[i][j];
                }
                if rdiag[k] != 0.0 {
                    let t = a[j][k] / rdiag[k];
                    rdiag[k] *= (1.0 - t * t).max(0.0).sqrt();
                    if 0.05 * (rdiag[k] / wa[k]).powi(2) <= eps {
                        rdiag[k] = enorm(&col(a, k, j + 1));
                        wa[k] = rdiag[k];
                    }
                }
            }
        }
        rdiag[j] = -ajnorm;
    }
    (ipvt, rdiag, acnorm)
}

/// `qrsolv`: given `R` (upper triangle of `r`) from `A P = Q R`, solve
/// `min ‖[A; D] x − [b; 0]‖` with `qtb = Qᵀb`; leaves `S` (with `PᵀAᵀAP + PᵀD²P = SᵀS`) in the
/// strict lower triangle of `r` and its diagonal in `sdiag`.
fn qrsolv(r: &mut [Vec<f64>], n: usize, ipvt: &[usize], diag: &[f64], qtb: &[f64], x: &mut [f64], sdiag: &mut [f64]) {
    let mut wa = vec![0.0; n];
    for j in 0..n {
        for i in j..n {
            r[i][j] = r[j][i];
        }
        x[j] = r[j][j];
        wa[j] = qtb[j];
    }
    for j in 0..n {
        let l = ipvt[j];
        if diag[l] != 0.0 {
            for v in sdiag.iter_mut().skip(j) {
                *v = 0.0;
            }
            sdiag[j] = diag[l];
            let mut qtbpj = 0.0;
            for k in j..n {
                if sdiag[k] == 0.0 {
                    continue;
                }
                let (cos, sin) = if r[k][k].abs() < sdiag[k].abs() {
                    let cotan = r[k][k] / sdiag[k];
                    let sin = 0.5 / (0.25 + 0.25 * cotan * cotan).sqrt();
                    (sin * cotan, sin)
                } else {
                    let tan = sdiag[k] / r[k][k];
                    let cos = 0.5 / (0.25 + 0.25 * tan * tan).sqrt();
                    (cos, cos * tan)
                };
                r[k][k] = cos * r[k][k] + sin * sdiag[k];
                let temp = cos * wa[k] + sin * qtbpj;
                qtbpj = -sin * wa[k] + cos * qtbpj;
                wa[k] = temp;
                for i in k + 1..n {
                    let temp = cos * r[i][k] + sin * sdiag[i];
                    sdiag[i] = -sin * r[i][k] + cos * sdiag[i];
                    r[i][k] = temp;
                }
            }
        }
        sdiag[j] = r[j][j];
        r[j][j] = x[j];
    }
    let mut nsing = n;
    for j in 0..n {
        if sdiag[j] == 0.0 && nsing == n {
            nsing = j;
        }
        if nsing < n {
            wa[j] = 0.0;
        }
    }
    for j in (0..nsing).rev() {
        let sum: f64 = (j + 1..nsing).map(|i| r[i][j] * wa[i]).sum();
        wa[j] = (wa[j] - sum) / sdiag[j];
    }
    for j in 0..n {
        x[ipvt[j]] = wa[j];
    }
}

/// `lmpar`: the Levenberg-Marquardt parameter `λ` for which the step `x` solving
/// `min ‖[J; √λ D] x − [−F; 0]‖` has `‖D x‖` within 10 % of `Δ` (or `λ = 0` if the Gauss-Newton
/// step is shorter), by Moré's safeguarded Newton iteration on `‖D x(λ)‖ − Δ`.
fn lmpar(r: &mut [Vec<f64>], n: usize, ipvt: &[usize], diag: &[f64], qtb: &[f64], delta: f64, par: f64) -> (f64, Vec<f64>) {
    let dwarf = f64::MIN_POSITIVE;
    let mut x = vec![0.0; n];
    let mut sdiag = vec![0.0; n];
    let mut wa1 = qtb.to_vec();
    let mut nsing = n;
    for j in 0..n {
        if r[j][j] == 0.0 && nsing == n {
            nsing = j;
        }
        if nsing < n {
            wa1[j] = 0.0;
        }
    }
    for j in (0..nsing).rev() {
        wa1[j] /= r[j][j];
        let temp = wa1[j];
        for i in 0..j {
            wa1[i] -= r[i][j] * temp;
        }
    }
    for j in 0..n {
        x[ipvt[j]] = wa1[j];
    }
    let mut wa2: Vec<f64> = (0..n).map(|j| diag[j] * x[j]).collect();
    let mut dxnorm = enorm(&wa2);
    let mut fp = dxnorm - delta;
    if fp <= 0.1 * delta {
        return (0.0, x);
    }
    let mut parl = 0.0;
    if nsing >= n {
        for j in 0..n {
            let l = ipvt[j];
            wa1[j] = diag[l] * (wa2[l] / dxnorm);
        }
        for j in 0..n {
            let sum: f64 = (0..j).map(|i| r[i][j] * wa1[i]).sum();
            wa1[j] = (wa1[j] - sum) / r[j][j];
        }
        let temp = enorm(&wa1);
        parl = ((fp / delta) / temp) / temp;
    }
    for j in 0..n {
        let sum: f64 = (0..=j).map(|i| r[i][j] * qtb[i]).sum();
        wa1[j] = sum / diag[ipvt[j]];
    }
    let gnorm = enorm(&wa1);
    let mut paru = gnorm / delta;
    if paru == 0.0 {
        paru = dwarf / delta.min(0.1);
    }
    let mut par = par.max(parl).min(paru);
    if par == 0.0 {
        par = gnorm / dxnorm;
    }
    let mut iter = 0;
    loop {
        iter += 1;
        if par == 0.0 {
            par = dwarf.max(0.001 * paru);
        }
        let sp = par.sqrt();
        let d: Vec<f64> = diag.iter().map(|v| sp * v).collect();
        qrsolv(r, n, ipvt, &d, qtb, &mut x, &mut sdiag);
        wa2 = (0..n).map(|j| diag[j] * x[j]).collect();
        dxnorm = enorm(&wa2);
        let temp = fp;
        fp = dxnorm - delta;
        if fp.abs() <= 0.1 * delta || (parl == 0.0 && fp <= temp && temp < 0.0) || iter == 10 {
            break;
        }
        for j in 0..n {
            let l = ipvt[j];
            wa1[j] = diag[l] * (wa2[l] / dxnorm);
        }
        for j in 0..n {
            wa1[j] /= sdiag[j];
            let temp = wa1[j];
            for i in j + 1..n {
                wa1[i] -= r[i][j] * temp;
            }
        }
        let temp = enorm(&wa1);
        let parc = ((fp / delta) / temp) / temp;
        if fp > 0.0 {
            parl = parl.max(par);
        }
        if fp < 0.0 {
            paru = paru.min(par);
        }
        par = parl.max(par + parc);
    }
    (par, x)
}

// ANCHOR: leastsq
/// Minimises `‖F(x)‖²` by MINPACK's Levenberg-Marquardt method (SciPy's `leastsq`; `lmder` with
/// an analytic Jacobian, `lmdif` with forward differences). Each iteration factorises the
/// Jacobian `J P = Q R` once, then finds by `lmpar` the damping `λ` whose step
/// `min ‖J p + F‖² + λ‖D p‖²` fits the trust region `‖D p‖ ≤ Δ`, with `D` the running maximum of
/// the column norms of `J` (MINPACK's `mode = 1`); `Δ` grows or shrinks with the ratio of actual
/// to predicted reduction. `jac` returns the `m × n` Jacobian by rows.
pub fn leastsq(
    mut fun: impl FnMut(&[f64]) -> Vec<f64>,
    mut jac: Option<JacobianFn<'_>>,
    x0: &[f64],
    options: LmOptions,
) -> Result<LmResult, ErrorsJSL> {
    let n = x0.len();
    let eps = f64::EPSILON;
    let mut x = x0.to_vec();
    let mut fvec = fun(&x);
    let m = fvec.len();
    if n == 0 || m < n {
        return Err(ErrorsJSL::InvalidInputRange("leastsq: need at least as many residuals as unknowns"));
    }
    let maxfev = options.maxfev.unwrap_or(if jac.is_some() { 100 * (n + 1) } else { 200 * (n + 1) });
    let (mut nfev, mut njev) = (1usize, 0usize);
    let mut fnorm = enorm(&fvec);
    let mut par = 0.0;
    let mut iter = 1;
    let mut diag = vec![0.0; n];
    let (mut delta, mut xnorm) = (0.0, 0.0);
    let mut info = 0;
    let fd_eps = options.epsfcn.max(eps).sqrt();
    'outer: loop {
        // The Jacobian, analytic or by forward differences (MINPACK's fdjac2).
        let mut fjac: Vec<Vec<f64>> = match jac.as_mut() {
            Some(j) => {
                njev += 1;
                j(&x)
            }
            None => {
                let mut cols = vec![vec![0.0; n]; m];
                for jj in 0..n {
                    let temp = x[jj];
                    let mut h = fd_eps * temp.abs();
                    if h == 0.0 {
                        h = fd_eps;
                    }
                    x[jj] = temp + h;
                    let wa = fun(&x);
                    x[jj] = temp;
                    for i in 0..m {
                        cols[i][jj] = (wa[i] - fvec[i]) / h;
                    }
                }
                nfev += n;
                cols
            }
        };
        if fjac.len() != m || fjac.iter().any(|r| r.len() != n) {
            return Err(ErrorsJSL::IncompatibleArraySizes((fjac.len(), m)));
        }
        let (ipvt, rdiag, acnorm) = qrfac(&mut fjac, m, n);
        if iter == 1 {
            for j in 0..n {
                diag[j] = if acnorm[j] == 0.0 { 1.0 } else { acnorm[j] };
            }
            xnorm = enorm(&(0..n).map(|j| diag[j] * x[j]).collect::<Vec<_>>());
            delta = options.factor * xnorm;
            if delta == 0.0 {
                delta = options.factor;
            }
        }
        // Qᵀ F, keeping the first n components.
        let mut wa4 = fvec.clone();
        let mut qtf = vec![0.0; n];
        for j in 0..n {
            if fjac[j][j] != 0.0 {
                let sum: f64 = (j..m).map(|i| fjac[i][j] * wa4[i]).sum();
                let temp = -sum / fjac[j][j];
                for i in j..m {
                    wa4[i] += fjac[i][j] * temp;
                }
            }
            fjac[j][j] = rdiag[j];
            qtf[j] = wa4[j];
        }
        let mut gnorm = 0.0f64;
        if fnorm != 0.0 {
            for j in 0..n {
                let l = ipvt[j];
                if acnorm[l] != 0.0 {
                    let sum: f64 = (0..=j).map(|i| fjac[i][j] * (qtf[i] / fnorm)).sum();
                    gnorm = gnorm.max((sum / acnorm[l]).abs());
                }
            }
        }
        if gnorm <= options.gtol {
            info = 4;
            break;
        }
        for j in 0..n {
            diag[j] = diag[j].max(acnorm[j]);
        }
        let mut r: Vec<Vec<f64>> = (0..n).map(|i| fjac[i][..n].to_vec()).collect();
        loop {
            let (p, step) = lmpar(&mut r, n, &ipvt, &diag, &qtf, delta, par);
            par = p;
            let wa1: Vec<f64> = step.iter().map(|v| -v).collect();
            let wa2: Vec<f64> = (0..n).map(|j| x[j] + wa1[j]).collect();
            let pnorm = enorm(&(0..n).map(|j| diag[j] * wa1[j]).collect::<Vec<_>>());
            if iter == 1 {
                delta = delta.min(pnorm);
            }
            let wa4 = fun(&wa2);
            nfev += 1;
            let fnorm1 = enorm(&wa4);
            let actred = if 0.1 * fnorm1 < fnorm { 1.0 - (fnorm1 / fnorm).powi(2) } else { -1.0 };
            let mut wa3 = vec![0.0; n];
            for j in 0..n {
                let temp = wa1[ipvt[j]];
                for i in 0..=j {
                    wa3[i] += r[i][j] * temp;
                }
            }
            let temp1 = enorm(&wa3) / fnorm;
            let temp2 = par.sqrt() * pnorm / fnorm;
            let prered = temp1 * temp1 + temp2 * temp2 / 0.5;
            let dirder = -(temp1 * temp1 + temp2 * temp2);
            let ratio = if prered != 0.0 { actred / prered } else { 0.0 };
            if ratio <= 0.25 {
                let mut temp = if actred >= 0.0 { 0.5 } else { 0.5 * dirder / (dirder + 0.5 * actred) };
                if 0.1 * fnorm1 >= fnorm || temp < 0.1 {
                    temp = 0.1;
                }
                delta = temp * delta.min(pnorm / 0.1);
                par /= temp;
            } else if par == 0.0 || ratio >= 0.75 {
                delta = pnorm / 0.5;
                par *= 0.5;
            }
            if ratio >= 1e-4 {
                x = wa2;
                xnorm = enorm(&(0..n).map(|j| diag[j] * x[j]).collect::<Vec<_>>());
                fvec = wa4;
                fnorm = fnorm1;
                iter += 1;
            }
            if actred.abs() <= options.ftol && prered <= options.ftol && 0.5 * ratio <= 1.0 {
                info = 1;
            }
            if delta <= options.xtol * xnorm {
                info = if info == 1 { 3 } else { 2 };
            }
            if info != 0 {
                break 'outer;
            }
            if nfev >= maxfev {
                info = 5;
            }
            if actred.abs() <= eps && prered <= eps && 0.5 * ratio <= 1.0 {
                info = 6;
            }
            if delta <= eps * xnorm {
                info = 7;
            }
            if gnorm <= eps {
                info = 8;
            }
            if info != 0 {
                break 'outer;
            }
            if ratio >= 1e-4 {
                break;
            }
        }
    }
    Ok(LmResult { x, fvec, nfev, njev, info })
}
// ANCHOR_END: leastsq

/// A root of the square system `F(x) = 0` by MINPACK's Levenberg-Marquardt method (SciPy's
/// `root(method="lm")`): [`leastsq`] on `‖F‖²`. Converged when MINPACK stops on its ftol or xtol
/// test (info 1–4) with `‖F‖∞` at most `1e-8`; a local minimum of `‖F‖` that is not a root is
/// reported as not converged.
pub fn root_lm(fun: impl FnMut(&[f64]) -> Vec<f64>, jac: Option<JacobianFn<'_>>, x0: &[f64], options: LmOptions) -> Result<SystemRootResult, ErrorsJSL> {
    let r = leastsq(fun, jac, x0, options)?;
    if r.fvec.len() != r.x.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((r.fvec.len(), r.x.len())));
    }
    let residual = r.fvec.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let converged = (1..=4).contains(&r.info) && residual <= 1e-8;
    Ok(SystemRootResult::new(r.x, residual, r.nfev, r.nfev, converged))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosenbrock_and_a_system() {
        // Rosenbrock residuals: the minimum (1, 1) has zero residual.
        let f = |x: &[f64]| vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]];
        let mut j = |x: &[f64]| vec![vec![-20.0 * x[0], 10.0], vec![-1.0, 0.0]];
        let r = leastsq(f, Some(&mut j), &[-1.2, 1.0], LmOptions::default()).unwrap();
        assert!((r.x[0] - 1.0).abs() < 1e-10 && (r.x[1] - 1.0).abs() < 1e-10, "{:?}", r.x);
        let r2 = leastsq(f, None, &[-1.2, 1.0], LmOptions::default()).unwrap();
        assert!((r2.x[0] - 1.0).abs() < 1e-8);
        // x² + y² = 4, eˣ + y = 1.
        let g = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 4.0, x[0].exp() + x[1] - 1.0];
        let s = root_lm(g, None, &[1.0, -1.0], LmOptions::default()).unwrap();
        assert!(s.converged && s.residual < 1e-10, "{s:?}");
    }
}
