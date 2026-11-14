//! The principal matrix square root and logarithm, and fractional powers, through the complex
//! Schur form (SciPy's `sqrtm`, `logm` and `fractional_matrix_power`).
//!
//! See the book entry [Matrix functions, subspaces and norms](https://jwkunz.github.io/lib_jsl/book/linear_algebra/subspaces-norms.html).

use crate::dense::{Mat, lincomb, matmul, norm1};
use crate::eig::schur_unbalanced;
use crate::error::{LinalgError, Result};
use crate::lu::LuFactorization;
use crate::scalar::Scalar;
use ndarray::{Array2, ArrayBase, Data, Ix2};
use num_complex::Complex;

type C = Complex<f64>;

// ANCHOR: sqrtm_triangular
/// The principal square root of an upper triangular matrix, column by column (Björck and
/// Hammarling): `rᵢᵢ = √tᵢᵢ` and `rᵢⱼ = (tᵢⱼ − Σ_{i<k<j} rᵢₖ rₖⱼ) / (rᵢᵢ + rⱼⱼ)`.
fn sqrt_triangular(t: &Mat<C>) -> Result<Mat<C>> {
    let n = t.rows;
    let mut r = Mat::<C>::zeros(n, n);
    for j in 0..n {
        *r.at_mut(j, j) = t.at(j, j).sqrt();
        for i in (0..j).rev() {
            let s = (i + 1..j).fold(C::new(0.0, 0.0), |acc, k| acc + r.at(i, k) * r.at(k, j));
            let d = r.at(i, i) + r.at(j, j);
            if d.norm() == 0.0 {
                return Err(LinalgError::Singular(i));
            }
            *r.at_mut(i, j) = (t.at(i, j) - s) / d;
        }
    }
    Ok(r)
}
// ANCHOR_END: sqrtm_triangular

fn schur<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>) -> Result<(Mat<C>, Mat<C>)> {
    schur_unbalanced(&Mat::from_array(a)?)
}

fn back(z: &Mat<C>, f: &Mat<C>) -> Array2<C> {
    matmul(&matmul(z, f), &z.adjoint()).into_array()
}

/// The principal square root `X` with `X² = A` and eigenvalues in the right half-plane (SciPy's
/// `sqrtm`), by the Schur method: `A = Z T Zᴴ`, `X = Z R Zᴴ` with `R² = T` triangular. Matrices
/// with eigenvalues on the closed negative real axis have no principal square root; zero
/// eigenvalues of multiplicity above one make it fail.
pub fn sqrtm<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>) -> Result<Array2<C>> {
    let (t, z) = schur(a)?;
    Ok(back(&z, &sqrt_triangular(&t)?))
}

/// Nodes and weights of `m`-point Gauss-Legendre quadrature on [0, 1].
fn gauss_legendre_01(m: usize) -> Vec<(f64, f64)> {
    (0..m)
        .map(|i| {
            let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (m as f64 + 0.5)).cos();
            let mut dp = 0.0;
            for _ in 0..100 {
                let (mut p0, mut p1) = (1.0, z);
                for k in 2..=m {
                    let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                    p0 = p1;
                    p1 = p2;
                }
                dp = m as f64 * (z * p1 - p0) / (z * z - 1.0);
                let dz = p1 / dp;
                z -= dz;
                if dz.abs() < 1e-16 {
                    break;
                }
            }
            (0.5 * (z + 1.0), 1.0 / ((1.0 - z * z) * dp * dp))
        })
        .collect()
}

// ANCHOR: logm
/// The principal logarithm `L` with `e^L = A` and eigenvalues with imaginary parts in `(−π, π)`
/// (SciPy's `logm`), by the inverse scaling and squaring method on the Schur form: take square
/// roots of `T` until `‖T − I‖₁ ≤ 0.25` (`k` times), evaluate `log(I + X)` by the `[8/8]` Padé
/// approximant written as Gauss-Legendre quadrature, `Σ wⱼ X (I + xⱼ X)⁻¹`, and multiply by `2ᵏ`.
pub fn logm<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>) -> Result<Array2<C>> {
    let (mut t, z) = schur(a)?;
    let n = t.rows;
    if (0..n).any(|i| t.at(i, i).norm() == 0.0) {
        return Err(LinalgError::Singular(0));
    }
    let eye = Mat::<C>::identity(n);
    let one = C::new(1.0, 0.0);
    let mut k = 0;
    while norm1(&lincomb(one, &t, -one, &eye)) > 0.25 {
        t = sqrt_triangular(&t)?;
        k += 1;
        if k > 100 {
            return Err(LinalgError::NoConvergence("logm square roots"));
        }
    }
    let x = lincomb(one, &t, -one, &eye);
    let mut log = Mat::<C>::zeros(n, n);
    for (node, weight) in gauss_legendre_01(8) {
        // X (I + x_j X)⁻¹ = (I + x_j X)⁻¹ X, since the two commute.
        let m = lincomb(one, &eye, C::new(node, 0.0), &x);
        let lu = LuFactorization::new(m)?;
        for col in 0..n {
            let rhs: Vec<C> = (0..n).map(|r| x.at(r, col)).collect();
            for (r, v) in lu.solve_vec(&rhs).into_iter().enumerate() {
                *log.at_mut(r, col) += v * weight;
            }
        }
    }
    let scale = C::new(2f64.powi(k), 0.0);
    log.data.iter_mut().for_each(|v| *v *= scale);
    Ok(back(&z, &log))
}
// ANCHOR_END: logm

/// `A^p = exp(p log A)` for real `p` (SciPy's `fractional_matrix_power`, by this route rather
/// than its Schur-Padé algorithm).
pub fn fractional_matrix_power<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>, p: f64) -> Result<Array2<C>> {
    let l = logm(a)?;
    crate::expm(&l.mapv(|v| v * p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn max_diff(a: &Array2<C>, b: &Array2<C>) -> f64 {
        a.iter().zip(b.iter()).fold(0.0, |m, (x, y)| m.max((x - y).norm()))
    }

    #[test]
    fn sqrtm_squares_back_and_logm_inverts_expm() {
        let a = array![[4.0, 1.0, 0.5], [0.2, 3.0, -0.4], [0.1, 0.3, 2.0]];
        let s = sqrtm(&a).unwrap();
        let ac = a.mapv(|v| C::new(v, 0.0));
        assert!(max_diff(&s.dot(&s), &ac) < 1e-13);
        let l = logm(&a).unwrap();
        assert!(max_diff(&crate::expm(&l).unwrap(), &ac) < 1e-12);
        // A rotation by θ: log is the generator.
        let th = 0.7f64;
        let r = array![[th.cos(), -th.sin()], [th.sin(), th.cos()]];
        let l = logm(&r).unwrap();
        assert!((l[(1, 0)].re - th).abs() < 1e-14 && (l[(0, 1)].re + th).abs() < 1e-14 && l[(0, 0)].norm() < 1e-14);
        let half = fractional_matrix_power(&a, 0.5).unwrap();
        assert!(max_diff(&half, &s) < 1e-12);
    }
}
