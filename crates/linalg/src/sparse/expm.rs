//! The action of the matrix exponential on a vector, `e^{tA} b`, without forming `e^{tA}`
//! (SciPy's `expm_multiply`): Al-Mohy and Higham's truncated Taylor series with scaling, after
//! shifting `A` by the mean of its eigenvalues.

use super::formats::CsrMatrix;
use crate::error::{LinalgError, Result};

/// `θ_m`: the largest `‖tA‖₁` for which `m` Taylor terms reach double precision (Al-Mohy and
/// Higham 2011, table 3.1, as in SciPy).
const THETA: [(usize, f64); 35] = [
    (1, 2.29e-16), (2, 2.58e-8), (3, 1.39e-5), (4, 3.40e-4), (5, 2.40e-3), (6, 9.07e-3), (7, 2.38e-2), (8, 5.00e-2), (9, 8.96e-2), (10, 1.44e-1),
    (11, 2.14e-1), (12, 3.00e-1), (13, 4.00e-1), (14, 5.14e-1), (15, 6.41e-1), (16, 7.81e-1), (17, 9.31e-1), (18, 1.09), (19, 1.26), (20, 1.44),
    (21, 1.62), (22, 1.82), (23, 2.01), (24, 2.22), (25, 2.43), (26, 2.64), (27, 2.86), (28, 3.08), (29, 3.31), (30, 3.54),
    (35, 4.7), (40, 6.0), (45, 7.2), (50, 8.5), (55, 9.9),
];
const M_MAX: usize = 55;

fn theta(m: usize) -> Option<f64> {
    THETA.iter().find(|(k, _)| *k == m).map(|(_, t)| *t)
}

fn one_norm(a: &CsrMatrix) -> f64 {
    let mut col = vec![0.0; a.ncols];
    for (j, v) in a.indices.iter().zip(&a.data) {
        col[*j] += v.abs();
    }
    col.into_iter().fold(0.0, f64::max)
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// The largest `p` with `p(p − 1) ≤ m_max + 1`.
fn p_max() -> usize {
    (1..).take_while(|p| p * (p - 1) <= M_MAX + 1).last().expect("p = 1 qualifies")
}

// ANCHOR: parameters
/// The number of Taylor terms `m` and scaling steps `s` that minimise the work `m s` (code
/// fragment 3.1 of Al-Mohy and Higham). For moderate norms (condition 3.13) `s = ⌈‖tA‖₁/θ_m⌉`;
/// otherwise the bound uses `α_p = max(‖(tA)^p‖^{1/p}, ‖(tA)^{p+1}‖^{1/(p+1)})`, which can be much
/// smaller than `‖tA‖₁` for non-normal matrices. SciPy estimates those norms; here they are
/// computed exactly from sparse powers.
fn parameters(a: &CsrMatrix, t: f64, norm: f64) -> (usize, usize) {
    let pm = p_max();
    let ell = 2.0;
    let condition_3_13 = norm <= 2.0 * ell * pm as f64 * (pm as f64 + 3.0) * theta(M_MAX).unwrap() / M_MAX as f64;
    let (mut best_m, mut best_s) = (0usize, usize::MAX);
    if condition_3_13 {
        for &(m, th) in &THETA {
            let s = (norm / th).ceil() as usize;
            if best_s == usize::MAX || m * s < best_m * best_s {
                best_m = m;
                best_s = s;
            }
        }
    } else {
        // d_p = ‖(tA)^p‖₁^{1/p}, from exact sparse powers.
        let mut d = vec![0.0; pm + 2];
        let mut power = a.clone();
        for (p, slot) in d.iter_mut().enumerate().skip(1) {
            if p > 1 {
                power = power.matmul(a).expect("square");
            }
            *slot = t.abs() * one_norm(&power).powf(1.0 / p as f64);
        }
        for p in 2..=pm {
            let alpha = d[p].max(d[p + 1]);
            for m in p * (p - 1) - 1..=M_MAX {
                if let Some(th) = theta(m) {
                    let s = (alpha / th).ceil() as usize;
                    if best_s == usize::MAX || m * s < best_m * best_s {
                        best_m = m;
                        best_s = s;
                    }
                }
            }
        }
        best_s = best_s.max(1);
    }
    (best_m, best_s)
}
// ANCHOR_END: parameters

// ANCHOR: expm_multiply
/// `e^{tA} b` (SciPy's `expm_multiply` at one time point). With `μ = trace(A)/n` and `B = A − μI`,
/// `e^{tA} b = e^{tμ} (e^{tB/s})^s b`, and each factor is applied by the Taylor series
/// `Σ_{j≤m} (tB/s)^j b / j!`, stopping early once two consecutive terms are negligible.
pub fn expm_multiply(a: &CsrMatrix, b: &[f64], t: f64) -> Result<Vec<f64>> {
    let n = a.nrows;
    if a.ncols != n || b.len() != n {
        return Err(LinalgError::Shape(format!("{} × {} operator with a vector of length {}", a.nrows, a.ncols, b.len())));
    }
    let mu = a.diagonal().iter().sum::<f64>() / n as f64;
    let shifted = a.add(1.0, &CsrMatrix::identity(n), -mu)?;
    let norm = t.abs() * one_norm(&shifted);
    let (m, s) = if norm == 0.0 { (0, 1) } else { parameters(&shifted, t, norm) };
    let tol = 2f64.powi(-53);
    let eta = (t * mu / s as f64).exp();
    let mut f = b.to_vec();
    let mut v = b.to_vec();
    for _ in 0..s {
        let mut c1 = inf_norm(&v);
        for j in 0..m {
            let coeff = t / (s * (j + 1)) as f64;
            v = shifted.matvec(&v).into_iter().map(|x| coeff * x).collect();
            let c2 = inf_norm(&v);
            f.iter_mut().zip(&v).for_each(|(fi, vi)| *fi += vi);
            if c1 + c2 <= tol * inf_norm(&f) {
                break;
            }
            c1 = c2;
        }
        f.iter_mut().for_each(|x| *x *= eta);
        v = f.clone();
    }
    Ok(f)
}
// ANCHOR_END: expm_multiply

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matfun::expm;
    use ndarray::Array1;

    #[test]
    fn matches_the_dense_exponential() {
        // A nonsymmetric tridiagonal matrix, at small and large norms.
        let n = 30;
        let a = CsrMatrix::diags(&[vec![-2.0; n], vec![1.3; n - 1], vec![0.7; n - 1]], &[0, 1, -1], n, n).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        for t in [0.1, 1.0, 25.0] {
            let got = expm_multiply(&a, &b, t).unwrap();
            let dense = expm(&a.to_dense().mapv(|v| v * t)).unwrap();
            let want = dense.dot(&Array1::from_vec(b.clone()));
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(got.iter().zip(want.iter()).all(|(g, w)| (g - w).abs() < 1e-12 * scale), "t = {t}");
        }
    }
}
