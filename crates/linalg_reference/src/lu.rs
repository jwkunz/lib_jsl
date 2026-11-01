//! The LU factorisation `P A = L U` (Doolittle form, unit lower triangular `L`).
//!
//! Gaussian elimination secretly computes `L` (the multipliers) and `U` (the reduced matrix).
//! Keeping them lets you solve `A x = b` for any number of right-hand sides at `O(n²)` each after
//! one `O(n³)` factorisation, and gives the determinant as the product of the pivots.

use crate::Error;
use ndarray::{Array1, Array2};

/// `P A = L U` with the permutation stored as a row order.
#[derive(Debug, Clone)]
pub struct Lu {
    /// Unit lower triangular factor.
    pub l: Array2<f64>,
    /// Upper triangular factor.
    pub u: Array2<f64>,
    /// Row `i` of `P A` is row `rows[i]` of `A`.
    pub rows: Vec<usize>,
    /// The number of row swaps (for the determinant's sign).
    pub swaps: usize,
}

/// Factorises a square matrix with partial pivoting.
pub fn lu(a: &Array2<f64>) -> Result<Lu, Error> {
    let n = a.nrows();
    let mut u = a.clone();
    let mut l = crate::identity(n);
    let mut rows: Vec<usize> = (0..n).collect();
    let mut swaps = 0;
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| u[[i, k]].abs().total_cmp(&u[[j, k]].abs())).ok_or("empty matrix")?;
        if u[[p, k]] == 0.0 {
            return Err("the matrix is singular");
        }
        if p != k {
            for j in 0..n {
                u.swap([k, j], [p, j]);
            }
            // Multipliers already computed move with their rows.
            for j in 0..k {
                l.swap([k, j], [p, j]);
            }
            rows.swap(k, p);
            swaps += 1;
        }
        for i in k + 1..n {
            l[[i, k]] = u[[i, k]] / u[[k, k]];
            for j in k..n {
                u[[i, j]] -= l[[i, k]] * u[[k, j]];
            }
        }
    }
    Ok(Lu { l, u, rows, swaps })
}

impl Lu {
    /// Solves `A x = b`: permute, then `L y = P b` forward and `U x = y` backward.
    pub fn solve(&self, b: &Array1<f64>) -> Result<Array1<f64>, Error> {
        let n = self.l.nrows();
        let mut y = Array1::zeros(n);
        for i in 0..n {
            y[i] = b[self.rows[i]] - (0..i).map(|j| self.l[[i, j]] * y[j]).sum::<f64>();
        }
        crate::gaussian_elimination::back_substitute(&self.u, &y)
    }

    /// `det A = (−1)^{swaps} Π u_ii`.
    pub fn determinant(&self) -> f64 {
        let sign = if self.swaps.is_multiple_of(2) { 1.0 } else { -1.0 };
        sign * (0..self.u.nrows()).map(|i| self.u[[i, i]]).product::<f64>()
    }

    /// The inverse, one column `A x = e_j` at a time.
    pub fn inverse(&self) -> Result<Array2<f64>, Error> {
        let n = self.l.nrows();
        let mut inv = Array2::zeros((n, n));
        for j in 0..n {
            let e = Array1::from_shape_fn(n, |i| if i == j { 1.0 } else { 0.0 });
            inv.column_mut(j).assign(&self.solve(&e)?);
        }
        Ok(inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random_matrix;
    use ndarray::array;

    #[test]
    fn factors_reconstruct_and_solve() {
        let a = random_matrix(12, 12, 2);
        let f = lu(&a).unwrap();
        let pa = Array2::from_shape_fn((12, 12), |(i, j)| a[[f.rows[i], j]]);
        assert!((f.l.dot(&f.u) - pa).iter().all(|v| v.abs() < 1e-12));
        let b = Array1::from_iter((0..12).map(|i| (i as f64).sin()));
        assert!((a.dot(&f.solve(&b).unwrap()) - &b).iter().all(|v| v.abs() < 1e-10));
        assert!((a.dot(&f.inverse().unwrap()) - crate::identity(12)).iter().all(|v| v.abs() < 1e-10));
        assert!((lu(&array![[2.0, 1.0], [7.0, 4.0]]).unwrap().determinant() - 1.0).abs() < 1e-14);
        assert!((lu(&array![[0.0, 1.0], [1.0, 0.0]]).unwrap().determinant() + 1.0).abs() < 1e-15);
    }
}
