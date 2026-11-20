//! The quadratic assignment problem (SciPy's `quadratic_assignment`): find the permutation `p`
//! minimising `Σᵢⱼ A[i][j] B[p(i)][p(j)]`, by Vogelstein et al.'s Fast Approximate QAP or by 2-opt
//! local search.
//!
//! See the book entry [Global and combinatorial optimisation](https://jwkunz.github.io/lib_jsl/book/optimization/shgo-qap.html).

use crate::assignment::linear_sum_assignment;
use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

/// The method of [`quadratic_assignment`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum QapMethod {
    /// Fast Approximate QAP: Frank-Wolfe on the doubly stochastic relaxation from the
    /// barycentre, then projected to the nearest permutation (SciPy's default `faq`, with
    /// `maxiter` and `tol`).
    Faq { maxiter: usize, tol: f64 },
    /// 2-opt from the given starting permutation (identity if empty): repeatedly apply the first
    /// improving swap in the order `(0,0), (0,1), …, (n−1,n−1)` (SciPy's `2opt`).
    TwoOpt { initial: Vec<usize> },
}

impl Default for QapMethod {
    fn default() -> Self {
        QapMethod::Faq { maxiter: 30, tol: 0.03 }
    }
}

/// The result of [`quadratic_assignment`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct QapResult {
    /// `col_ind`: node `i` of `A` is matched to node `perm[i]` of `B`.
    pub perm: Vec<usize>,
    /// `Σᵢⱼ A[i][j] B[perm[i]][perm[j]]`.
    pub fun: f64,
    pub nit: usize,
}

/// `Σᵢⱼ A[i][j] B[p(i)][p(j)]`.
pub fn qap_score(a: &Array2<f64>, b: &Array2<f64>, perm: &[usize]) -> f64 {
    let n = perm.len();
    (0..n).map(|i| (0..n).map(|j| a[(i, j)] * b[(perm[i], perm[j])]).sum::<f64>()).sum()
}

// ANCHOR: qap
/// Solves (approximately: the problem is NP-hard) `min_p Σᵢⱼ A[i][j] B[p(i)][p(j)]` over
/// permutations, or the maximum with `maximize` (SciPy's `quadratic_assignment`). Both methods
/// are deterministic ports of SciPy's and give its permutations.
pub fn quadratic_assignment(a: &Array2<f64>, b: &Array2<f64>, method: QapMethod, maximize: bool) -> Result<QapResult, ErrorsJSL> {
    let n = a.nrows();
    if a.dim() != (n, n) || b.dim() != (n, n) {
        return Err(ErrorsJSL::InvalidInputRange("A and B must be square and of the same size"));
    }
    if n == 0 {
        return Ok(QapResult { perm: Vec::new(), fun: 0.0, nit: 0 });
    }
    let sign = if maximize { -1.0 } else { 1.0 };
    match method {
        QapMethod::Faq { maxiter, tol } => {
            let mut p = Array2::from_elem((n, n), 1.0 / n as f64);
            let mut nit = 0;
            for _ in 0..maxiter {
                nit += 1;
                let grad = a.dot(&p).dot(&b.t()) + a.t().dot(&p).dot(b);
                let (_, cols) = linear_sum_assignment(&grad, maximize)?;
                let q = Array2::from_shape_fn((n, n), |(i, j)| if cols[i] == j { 1.0 } else { 0.0 });
                let r = &p - &q;
                // f(P + t R) is quadratic in t: a t² + b t + const.
                let ar = a.t().dot(&r);
                let br = b.dot(&r.t());
                let b22a: f64 = (0..n).map(|i| (0..n).map(|j| ar[(i, j)] * b[(j, cols[i])]).sum::<f64>()).sum();
                let b22b: f64 = (0..n).map(|i| (0..n).map(|j| a[(i, j)] * br[(cols[i], j)]).sum::<f64>()).sum();
                let qa: f64 = (0..n).map(|i| (0..n).map(|j| ar[(j, i)] * br[(i, j)]).sum::<f64>()).sum();
                let qb = b22a + b22b;
                let alpha = if qa * sign > 0.0 && (0.0..=1.0).contains(&(-qb / (2.0 * qa))) {
                    -qb / (2.0 * qa)
                } else if 0.0 <= (qb + qa) * sign {
                    0.0
                } else {
                    1.0
                };
                let next = &p * alpha + &q * (1.0 - alpha);
                let change = (&p - &next).iter().map(|v| v * v).sum::<f64>().sqrt() / (n as f64).sqrt();
                p = next;
                if change < tol {
                    break;
                }
            }
            let (_, perm) = linear_sum_assignment(&p, true)?;
            Ok(QapResult { fun: qap_score(a, b, &perm), perm, nit })
        }
        QapMethod::TwoOpt { initial } => {
            let mut perm: Vec<usize> = if initial.is_empty() { (0..n).collect() } else { initial };
            let mut seen = perm.clone();
            seen.sort_unstable();
            if seen != (0..n).collect::<Vec<_>>() {
                return Err(ErrorsJSL::InvalidInputRange("the initial guess must be a permutation of 0..n"));
            }
            let mut best = qap_score(a, b, &perm);
            let mut nit = 0;
            'outer: loop {
                for i in 0..n {
                    for j in i..n {
                        nit += 1;
                        perm.swap(i, j);
                        let s = qap_score(a, b, &perm);
                        if s * sign < best * sign {
                            best = s;
                            continue 'outer;
                        }
                        perm.swap(i, j);
                    }
                }
                break;
            }
            Ok(QapResult { perm, fun: best, nit })
        }
    }
}
// ANCHOR_END: qap

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_hidden_permutation() {
        // B is A relabelled by a permutation: the optimum of the maximisation is that
        // permutation, and 2-opt and FAQ both find it on this small, well-separated instance.
        let n = 8;
        let a = Array2::from_shape_fn((n, n), |(i, j)| if i == j { 0.0 } else { ((i * 7 + j * 3) % 11) as f64 + (i as f64 - j as f64).abs() });
        let hidden = [3usize, 0, 6, 1, 7, 2, 5, 4];
        let mut b = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                b[(hidden[i], hidden[j])] = a[(i, j)];
            }
        }
        let best = qap_score(&a, &b, &hidden);
        // SciPy: FAQ finds the hidden permutation in 2 iterations; 2-opt from the identity
        // (SciPy's partial_guess) finds it after trying 144 swaps.
        let f = quadratic_assignment(&a, &b, QapMethod::default(), true).unwrap();
        assert_eq!((f.perm.as_slice(), f.fun, f.nit), (&hidden[..], best, 2));
        let r = quadratic_assignment(&a, &b, QapMethod::TwoOpt { initial: Vec::new() }, true).unwrap();
        assert_eq!((r.perm, r.fun, r.nit), (hidden.to_vec(), best, 144));
        assert_eq!(qap_score(&a, &b, &f.perm), f.fun);
    }
}
