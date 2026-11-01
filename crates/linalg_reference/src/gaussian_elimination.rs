//! Gaussian elimination: the algorithm every linear solver descends from.
//!
//! To solve `A x = b`, subtract multiples of each row from the rows below it until `A` is upper
//! triangular, then solve the triangular system from the bottom up (back substitution). The
//! multiplier for row `i` at step `k` is `a_ik / a_kk`, so everything depends on the **pivot**
//! `a_kk`. A zero pivot stops the naive algorithm; a tiny one lets rounding errors grow without
//! bound. **Partial pivoting** swaps the row with the largest `|a_ik|` into the pivot position,
//! keeping every multiplier at most one in magnitude.
//!
//! See the book entry [Gaussian elimination and LU factorisation](https://jwkunz.github.io/lib_jsl/book/linear_algebra/lu.html#gaussian-elimination-and-pivoting).

use crate::Error;
use ndarray::{Array1, Array2};

/// Solves `A x = b` by elimination **without** pivoting. Fails on a zero pivot and is unstable
/// when a pivot is small; see [`solve_with_partial_pivoting`].
pub fn solve_without_pivoting(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, Error> {
    let n = a.nrows();
    let (mut a, mut b) = (a.clone(), b.clone());
    for k in 0..n {
        if a[[k, k]] == 0.0 {
            return Err("zero pivot: elimination without pivoting cannot continue");
        }
        for i in k + 1..n {
            let multiplier = a[[i, k]] / a[[k, k]];
            for j in k..n {
                a[[i, j]] -= multiplier * a[[k, j]];
            }
            b[i] -= multiplier * b[k];
        }
    }
    back_substitute(&a, &b)
}

// ANCHOR: partial_pivoting
/// Solves `A x = b` by elimination with **partial pivoting**: at step `k` the row with the
/// largest `|a_ik|` (`i ≥ k`) is swapped into the pivot position.
pub fn solve_with_partial_pivoting(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, Error> {
    let n = a.nrows();
    let (mut a, mut b) = (a.clone(), b.clone());
    for k in 0..n {
        let pivot_row = (k..n).max_by(|&i, &j| a[[i, k]].abs().total_cmp(&a[[j, k]].abs())).ok_or("empty matrix")?;
        if a[[pivot_row, k]] == 0.0 {
            return Err("the matrix is singular");
        }
        if pivot_row != k {
            for j in 0..n {
                a.swap([k, j], [pivot_row, j]);
            }
            b.swap(k, pivot_row);
        }
        for i in k + 1..n {
            let multiplier = a[[i, k]] / a[[k, k]];
            for j in k..n {
                a[[i, j]] -= multiplier * a[[k, j]];
            }
            b[i] -= multiplier * b[k];
        }
    }
    back_substitute(&a, &b)
}
// ANCHOR_END: partial_pivoting

/// Solves the upper triangular system `U x = c`, last unknown first.
pub fn back_substitute(u: &Array2<f64>, c: &Array1<f64>) -> Result<Array1<f64>, Error> {
    let n = u.nrows();
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let sum: f64 = (i + 1..n).map(|j| u[[i, j]] * x[j]).sum();
        if u[[i, i]] == 0.0 {
            return Err("zero on the diagonal of the triangular factor");
        }
        x[i] = (c[i] - sum) / u[[i, i]];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_matrix;
    use ndarray::array;

    #[test]
    fn both_variants_solve_well_conditioned_systems() {
        let a = random_matrix(20, 20, 1) + crate::identity(20) * 5.0;
        let b = Array1::from_iter((0..20).map(|i| i as f64));
        for x in [solve_without_pivoting(&a, &b).unwrap(), solve_with_partial_pivoting(&a, &b).unwrap()] {
            assert!((a.dot(&x) - &b).iter().all(|r| r.abs() < 1e-10));
        }
    }

    #[test]
    fn pivoting_rescues_small_pivots() {
        // The classic example: a tiny pivot ε makes naive elimination lose every digit of x₁.
        let eps = 1e-17;
        let a = array![[eps, 1.0], [1.0, 1.0]];
        let b = array![1.0, 2.0]; // x ≈ (1, 1)
        let naive = solve_without_pivoting(&a, &b).unwrap();
        let pivoted = solve_with_partial_pivoting(&a, &b).unwrap();
        assert!((naive[0] - 1.0).abs() > 0.5, "naive elimination should fail: {naive}");
        assert!((pivoted[0] - 1.0).abs() < 1e-15 && (pivoted[1] - 1.0).abs() < 1e-15);
        // A zero pivot stops the naive algorithm outright.
        assert!(solve_without_pivoting(&array![[0.0, 1.0], [1.0, 0.0]], &array![1.0, 1.0]).is_err());
        assert!(solve_with_partial_pivoting(&array![[1.0, 2.0], [2.0, 4.0]], &array![1.0, 1.0]).is_err());
    }
}
