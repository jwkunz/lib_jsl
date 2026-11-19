//! The singular value decomposition of a small square matrix by one-sided Jacobi rotations, for
//! the orthogonal fitting problems (Kabsch, Procrustes, rotation means) that need one without a
//! dependency on the linear-algebra crate.

// Dense matrices read best as index loops.
#![allow(clippy::needless_range_loop)]

/// `A = U diag(s) Vᵀ` for a square `n × n` matrix (rows as vectors), singular values descending;
/// `U` and `V` are complete orthogonal matrices even when `A` is singular.
pub(crate) fn svd_square(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    // Work on the columns of W = A V.
    let mut w: Vec<Vec<f64>> = (0..n).map(|j| (0..n).map(|i| a[i][j]).collect()).collect();
    let mut v: Vec<Vec<f64>> = (0..n).map(|j| (0..n).map(|i| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();
    for _ in 0..60 {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in p + 1..n {
                let (alpha, beta, gamma) = (dot(&w[p], &w[p]), dot(&w[q], &w[q]), dot(&w[p], &w[q]));
                if gamma == 0.0 {
                    continue;
                }
                off = off.max(gamma.abs() / (alpha * beta).sqrt().max(f64::MIN_POSITIVE));
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for cols in [&mut w, &mut v] {
                    for i in 0..n {
                        let (x, y) = (cols[p][i], cols[q][i]);
                        cols[p][i] = c * x - s * y;
                        cols[q][i] = s * x + c * y;
                    }
                }
            }
        }
        if off < 1e-15 {
            break;
        }
    }
    let mut sv: Vec<(f64, usize)> = (0..n).map(|j| (dot(&w[j], &w[j]).sqrt(), j)).collect();
    sv.sort_by(|a, b| b.0.total_cmp(&a.0));
    let s: Vec<f64> = sv.iter().map(|x| x.0).collect();
    let tiny = s.first().copied().unwrap_or(0.0) * 1e-14;
    // U's columns: the normalised columns of W, completed by Gram-Schmidt where s is ~0.
    let mut ucols: Vec<Vec<f64>> = Vec::with_capacity(n);
    for &(sigma, j) in &sv {
        if sigma > tiny && sigma > 0.0 {
            ucols.push(w[j].iter().map(|x| x / sigma).collect());
        } else {
            let mut best: Option<Vec<f64>> = None;
            for e in 0..n {
                let mut u: Vec<f64> = (0..n).map(|i| if i == e { 1.0 } else { 0.0 }).collect();
                for prev in &ucols {
                    let d = dot(&u, prev);
                    u.iter_mut().zip(prev).for_each(|(x, y)| *x -= d * y);
                }
                let nrm = dot(&u, &u).sqrt();
                if nrm > 0.5 {
                    best = Some(u.iter().map(|x| x / nrm).collect());
                    break;
                }
            }
            ucols.push(best.expect("a complementary direction exists"));
        }
    }
    let u: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|k| ucols[k][i]).collect()).collect();
    let vm: Vec<Vec<f64>> = (0..n).map(|i| sv.iter().map(|&(_, j)| v[j][i]).collect()).collect();
    (u, s, vm)
}

/// The determinant of a small square matrix by Gaussian elimination with partial pivoting.
pub(crate) fn determinant(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut m = a.to_vec();
    let mut det = 1.0;
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs())).expect("non-empty");
        if m[p][c] == 0.0 {
            return 0.0;
        }
        if p != c {
            m.swap(p, c);
            det = -det;
        }
        det *= m[c][c];
        for r in c + 1..n {
            let f = m[r][c] / m[c][c];
            for k in c..n {
                m[r][k] -= f * m[c][k];
            }
        }
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstructs_including_singular() {
        for a in [
            vec![vec![2.0, -1.0, 0.3], vec![0.5, 4.0, 1.0], vec![-1.0, 0.2, 3.0]],
            vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0], vec![1.0, 1.0, 1.0]],
            vec![vec![0.0; 3]; 3],
        ] {
            let (u, s, v) = svd_square(&a);
            for i in 0..3 {
                for j in 0..3 {
                    let r: f64 = (0..3).map(|k| u[i][k] * s[k] * v[j][k]).sum();
                    assert!((r - a[i][j]).abs() < 1e-13);
                    let uu: f64 = (0..3).map(|k| u[k][i] * u[k][j]).sum();
                    assert!((uu - if i == j { 1.0 } else { 0.0 }).abs() < 1e-13);
                }
            }
            assert!(s.windows(2).all(|w| w[0] >= w[1]));
        }
        assert!((determinant(&[vec![0.0, 2.0], vec![3.0, 1.0]]) + 6.0).abs() < 1e-15);
    }
}
