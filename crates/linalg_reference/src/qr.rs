//! The QR factorisation `A = Q R`: orthonormal columns times an upper triangular matrix.
//!
//! **Gram-Schmidt** builds `Q` one column at a time by subtracting from each column of `A` its
//! projections onto the columns already found. The *classical* form projects the original column;
//! the *modified* form projects the partially orthogonalised one, which is algebraically identical
//! but loses orthogonality much more slowly in floating point (roughly `ε κ(A)` instead of
//! `ε κ(A)²`, where `κ` is the condition number).
//!
//! **Householder** QR instead multiplies `A` by reflections `H = I − 2 v vᵀ`, each zeroing the
//! entries below one diagonal position. Products of reflections are exactly orthogonal, so `Q` is
//! orthonormal to working precision whatever the conditioning of `A`. This is what production
//! libraries use.
//!
//! All three return the thin factorisation of an `m × n` matrix with `m ≥ n`: `Q` is `m × n` and
//! `R` is `n × n`.

use ndarray::{Array1, Array2, s};

/// Classical Gram-Schmidt: `r_ij = q_i · a_j` with the *original* column `a_j`.
pub fn classical_gram_schmidt(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (m, n) = a.dim();
    let mut q = Array2::zeros((m, n));
    let mut r = Array2::zeros((n, n));
    for j in 0..n {
        let mut v = a.column(j).to_owned();
        for i in 0..j {
            r[[i, j]] = q.column(i).dot(&a.column(j));
            v.scaled_add(-r[[i, j]], &q.column(i));
        }
        r[[j, j]] = v.dot(&v).sqrt();
        q.column_mut(j).assign(&(v / r[[j, j]]));
    }
    (q, r)
}

/// Modified Gram-Schmidt: `r_ij = q_i · v` with the column `v` as orthogonalised so far.
pub fn modified_gram_schmidt(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (m, n) = a.dim();
    let mut q = Array2::zeros((m, n));
    let mut r = Array2::zeros((n, n));
    for j in 0..n {
        let mut v = a.column(j).to_owned();
        for i in 0..j {
            r[[i, j]] = q.column(i).dot(&v);
            v.scaled_add(-r[[i, j]], &q.column(i));
        }
        r[[j, j]] = v.dot(&v).sqrt();
        q.column_mut(j).assign(&(v / r[[j, j]]));
    }
    (q, r)
}

/// Householder QR. Reflection `k` maps column `k` (from row `k` down) onto a multiple of `e_1`;
/// `Q` is then accumulated by applying the reflections to the first `n` columns of the identity.
pub fn householder(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (m, n) = a.dim();
    let mut r = a.clone();
    let mut reflectors: Vec<Array1<f64>> = Vec::with_capacity(n);
    for k in 0..n {
        let mut v = r.slice(s![k.., k]).to_owned();
        let norm = v.dot(&v).sqrt();
        // Reflect onto −sign(x₀)‖x‖ e₁ so that v₀ = x₀ + sign(x₀)‖x‖ involves no cancellation.
        v[0] += if v[0] >= 0.0 { norm } else { -norm };
        let v_norm = v.dot(&v).sqrt();
        if v_norm > 0.0 {
            v /= v_norm;
        }
        apply_reflector(&mut r, &v, k);
        reflectors.push(v);
    }
    let mut q = Array2::from_shape_fn((m, n), |(i, j)| if i == j { 1.0 } else { 0.0 });
    for (k, v) in reflectors.iter().enumerate().rev() {
        apply_reflector(&mut q, v, k);
    }
    let r = Array2::from_shape_fn((n, n), |(i, j)| if j >= i { r[[i, j]] } else { 0.0 });
    (q, r)
}

/// Overwrites rows `k..` of `x` with `(I − 2 v vᵀ)` applied to them.
fn apply_reflector(x: &mut Array2<f64>, v: &Array1<f64>, k: usize) {
    for j in 0..x.ncols() {
        let d: f64 = v.iter().enumerate().map(|(i, vi)| vi * x[[k + i, j]]).sum();
        for (i, vi) in v.iter().enumerate() {
            x[[k + i, j]] -= 2.0 * d * vi;
        }
    }
}

/// `‖QᵀQ − I‖_F`, the loss of orthogonality.
pub fn orthogonality_error(q: &Array2<f64>) -> f64 {
    let e = q.t().dot(q) - crate::identity(q.ncols());
    e.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hilbert(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(i, j)| 1.0 / (i + j + 1) as f64)
    }

    #[test]
    fn all_three_factorise_a_well_conditioned_matrix() {
        let a = crate::random_matrix(9, 6, 4);
        for (q, r) in [classical_gram_schmidt(&a), modified_gram_schmidt(&a), householder(&a)] {
            assert!((q.dot(&r) - &a).iter().all(|v| v.abs() < 1e-12));
            assert!(orthogonality_error(&q) < 1e-12);
        }
    }

    #[test]
    fn orthogonality_degrades_in_the_textbook_order() {
        // The 10 × 10 Hilbert matrix has condition number ≈ 1.6e13.
        let a = hilbert(10);
        let cgs = orthogonality_error(&classical_gram_schmidt(&a).0);
        let mgs = orthogonality_error(&modified_gram_schmidt(&a).0);
        let (q, r) = householder(&a);
        let house = orthogonality_error(&q);
        assert!(cgs > 1.0, "classical: {cgs:e}");
        assert!(mgs > 1e-6 && mgs < 1e-1, "modified: {mgs:e}");
        assert!(house < 1e-14, "Householder: {house:e}");
        assert!((q.dot(&r) - &a).iter().all(|v| v.abs() < 1e-14));
    }
}
