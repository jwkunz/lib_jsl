//! Linear and quadratic matrix equations: Sylvester, continuous and discrete Lyapunov, and the
//! continuous and discrete algebraic Riccati equations, with SciPy's names and conventions.
//!
//! The linear equations are solved by the Bartels-Stewart method: reduce the coefficient matrices
//! to complex Schur form, so that the transformed equation is triangular and can be solved column
//! by column. The Riccati equations are solved by a structured iteration (the matrix sign function
//! for the continuous equation, the doubling algorithm for the discrete one) followed by Newton
//! refinement, each step of which is a Lyapunov equation.
//!
//! See the book entry [Matrix functions and matrix equations](https://jwkunz.github.io/lib_jsl/book/linear_algebra/matrix-equations.html).

use crate::dense::{Mat, lincomb, matmul, norm1, require_square};
use crate::eig::schur_unbalanced;
use crate::error::{LinalgError, Result};
use crate::lu::LuFactorization;
use crate::scalar::Scalar;
use crate::LeastSquaresSvd;
use ndarray::{Array2, ArrayBase, Data, Ix2};
use num_complex::Complex;

type C = Complex<f64>;
/// The four coefficient matrices of a Riccati equation, `(A, B, Q, R)`.
type Coefficients = (Mat<C>, Mat<C>, Mat<C>, Mat<C>);

fn complex<T: Scalar, S: Data<Elem = T>>(a: &ArrayBase<S, Ix2>) -> Result<Mat<C>> {
    let m = Mat::from_array(a)?;
    Ok(Mat { rows: m.rows, cols: m.cols, data: m.data.iter().map(|v| v.to_complex()).collect() })
}

fn back<T: Scalar>(m: Mat<C>) -> Array2<T> {
    Mat { rows: m.rows, cols: m.cols, data: m.data.into_iter().map(T::from_complex).collect() }.into_array()
}

fn shape(what: &str, got: (usize, usize), want: (usize, usize)) -> Result<()> {
    if got == want { Ok(()) } else { Err(LinalgError::Shape(format!("{what} is {} × {}, expected {} × {}", got.0, got.1, want.0, want.1))) }
}

fn hermitian_part(x: &Mat<C>) -> Mat<C> {
    lincomb(C::new(0.5, 0.0), x, C::new(0.5, 0.0), &x.adjoint())
}

fn max_abs(m: &Mat<C>) -> f64 {
    m.data.iter().fold(0.0, |a, v| a.max(v.norm()))
}

/// Solves `(R + σ I) y = f` for upper triangular `R`, in place.
fn shifted_back_substitution(r: &Mat<C>, sigma: C, y: &mut [C], tiny: f64) -> Result<()> {
    let n = r.rows;
    for i in (0..n).rev() {
        let s = (i + 1..n).fold(C::new(0.0, 0.0), |acc, k| acc + r.at(i, k) * y[k]);
        let d = r.at(i, i) + sigma;
        if d.norm() <= tiny {
            return Err(LinalgError::Singular(i));
        }
        y[i] = (y[i] - s) / d;
    }
    Ok(())
}

fn column(m: &Mat<C>, j: usize) -> Vec<C> {
    (0..m.rows).map(|i| m.at(i, j)).collect()
}

fn set_column(m: &mut Mat<C>, j: usize, v: &[C]) {
    for (i, &x) in v.iter().enumerate() {
        *m.at_mut(i, j) = x;
    }
}

// ANCHOR: sylvester
/// Bartels-Stewart for `A X + X B = Q` on complex matrices.
fn sylvester(a: &Mat<C>, b: &Mat<C>, q: &Mat<C>) -> Result<Mat<C>> {
    let (r, u) = schur_unbalanced(a)?;
    let (s, v) = schur_unbalanced(b)?;
    // With A = U R Uᴴ and B = V S Vᴴ, Y = Uᴴ X V solves R Y + Y S = F = Uᴴ Q V.
    let f = matmul(&matmul(&u.adjoint(), q), &v);
    let (n, m) = (a.rows, b.rows);
    let tiny = f64::EPSILON * (max_abs(&r) + max_abs(&s)).max(f64::MIN_POSITIVE);
    let mut y = Mat::<C>::zeros(n, m);
    // S is upper triangular, so column k of Y involves only columns j ≤ k:
    // (R + s_kk I) y_k = f_k − Σ_{j<k} s_jk y_j.
    for k in 0..m {
        let mut rhs = column(&f, k);
        for j in 0..k {
            let sjk = s.at(j, k);
            for i in 0..n {
                rhs[i] -= sjk * y.at(i, j);
            }
        }
        shifted_back_substitution(&r, s.at(k, k), &mut rhs, tiny)?;
        set_column(&mut y, k, &rhs);
    }
    Ok(matmul(&matmul(&u, &y), &v.adjoint()))
}
// ANCHOR_END: sylvester

// ANCHOR: stein
/// The discrete Lyapunov (Stein) equation `A X Aᴴ − X + Q = 0` by the same Schur reduction.
fn stein(a: &Mat<C>, q: &Mat<C>) -> Result<Mat<C>> {
    let (r, u) = schur_unbalanced(a)?;
    let f = matmul(&matmul(&u.adjoint(), q), &u);
    let n = a.rows;
    let tiny = f64::EPSILON * (1.0 + max_abs(&r) * max_abs(&r));
    let mut y = Mat::<C>::zeros(n, n);
    // Column j of R Y Rᴴ is R (conj(r_jj) y_j + w_j) with w_j = Σ_{l>j} conj(r_jl) y_l, so
    // (conj(r_jj) R − I) y_j = −f_j − R w_j, solved from the last column to the first.
    for j in (0..n).rev() {
        let mut w = vec![C::new(0.0, 0.0); n];
        for l in j + 1..n {
            let c = r.at(j, l).conj();
            for i in 0..n {
                w[i] += c * y.at(i, l);
            }
        }
        let mut rhs: Vec<C> = (0..n).map(|i| -f.at(i, j) - (i..n).fold(C::new(0.0, 0.0), |acc, k| acc + r.at(i, k) * w[k])).collect();
        // (c R − I) y = rhs  ⇔  (R − I/c) y = rhs / c, for c = conj(r_jj) ≠ 0.
        let c = r.at(j, j).conj();
        if c.norm() <= tiny {
            // c = 0: −y = rhs.
            rhs.iter_mut().for_each(|v| *v = -*v);
        } else {
            rhs.iter_mut().for_each(|v| *v /= c);
            shifted_back_substitution(&r, -c.inv(), &mut rhs, tiny / c.norm())?;
        }
        set_column(&mut y, j, &rhs);
    }
    Ok(matmul(&matmul(&u, &y), &u.adjoint()))
}
// ANCHOR_END: stein

/// Solves the Sylvester equation `A X + X B = Q` for `X` (SciPy's `linalg.solve_sylvester`), with
/// `A` `n × n`, `B` `m × m` and `Q` `n × m`. The solution is unique when no eigenvalue of `A` is
/// the negative of an eigenvalue of `B`; otherwise [`LinalgError::Singular`] is returned.
pub fn solve_sylvester<T: Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>, S3: Data<Elem = T>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    q: &ArrayBase<S3, Ix2>,
) -> Result<Array2<T>> {
    let (a, b, q) = (complex(a)?, complex(b)?, complex(q)?);
    require_square(a.rows, a.cols)?;
    require_square(b.rows, b.cols)?;
    shape("Q", (q.rows, q.cols), (a.rows, b.rows))?;
    Ok(back(sylvester(&a, &b, &q)?))
}

/// Solves the continuous Lyapunov equation `A X + X Aᴴ = Q` (SciPy's
/// `linalg.solve_continuous_lyapunov`). For a stable `A` and `Q = −C Cᴴ`, `X` is the
/// controllability Gramian.
pub fn solve_continuous_lyapunov<T: Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>>(a: &ArrayBase<S1, Ix2>, q: &ArrayBase<S2, Ix2>) -> Result<Array2<T>> {
    let (a, q) = (complex(a)?, complex(q)?);
    require_square(a.rows, a.cols)?;
    shape("Q", (q.rows, q.cols), (a.rows, a.rows))?;
    Ok(back(sylvester(&a, &a.adjoint(), &q)?))
}

/// Solves the discrete Lyapunov equation `A X Aᴴ − X + Q = 0` (SciPy's
/// `linalg.solve_discrete_lyapunov`). The solution is unique when no product of two eigenvalues
/// of `A` (one conjugated) equals 1; for `A` with spectral radius below 1 and `Q = B Bᴴ`, `X` is the
/// controllability Gramian `Σ Aᵏ B Bᴴ (Aᴴ)ᵏ`.
pub fn solve_discrete_lyapunov<T: Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>>(a: &ArrayBase<S1, Ix2>, q: &ArrayBase<S2, Ix2>) -> Result<Array2<T>> {
    let (a, q) = (complex(a)?, complex(q)?);
    require_square(a.rows, a.cols)?;
    shape("Q", (q.rows, q.cols), (a.rows, a.rows))?;
    Ok(back(stein(&a, &q)?))
}

fn riccati_inputs<T: Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>, S3: Data<Elem = T>, S4: Data<Elem = T>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    q: &ArrayBase<S3, Ix2>,
    r: &ArrayBase<S4, Ix2>,
) -> Result<Coefficients> {
    let (a, b, q, r) = (complex(a)?, complex(b)?, complex(q)?, complex(r)?);
    require_square(a.rows, a.cols)?;
    let (n, m) = (a.rows, b.cols);
    shape("B", (b.rows, b.cols), (n, m))?;
    shape("Q", (q.rows, q.cols), (n, n))?;
    shape("R", (r.rows, r.cols), (m, m))?;
    Ok((a, b, q, r))
}

/// `R⁻¹ M` by an LU factorisation of `R`.
fn solve_left(r: &Mat<C>, m: &Mat<C>) -> Result<Mat<C>> {
    let lu = LuFactorization::new(r.clone())?;
    let mut out = Mat::zeros(m.rows, m.cols);
    for j in 0..m.cols {
        set_column(&mut out, j, &lu.solve_vec(&column(m, j)));
    }
    Ok(out)
}

fn sub(a: &Mat<C>, b: &Mat<C>) -> Mat<C> {
    lincomb(C::new(1.0, 0.0), a, C::new(-1.0, 0.0), b)
}

fn add(a: &Mat<C>, b: &Mat<C>) -> Mat<C> {
    lincomb(C::new(1.0, 0.0), a, C::new(1.0, 0.0), b)
}

/// Newton steps on a Riccati residual until it stops decreasing (at most `steps`).
fn refine(mut x: Mat<C>, steps: usize, residual: impl Fn(&Mat<C>) -> Mat<C>, correction: impl Fn(&Mat<C>, &Mat<C>) -> Result<Mat<C>>) -> Mat<C> {
    let mut res = residual(&x);
    let mut size = norm1(&res);
    for _ in 0..steps {
        let Ok(delta) = correction(&x, &res) else { break };
        let candidate = hermitian_part(&add(&x, &delta));
        let res_c = residual(&candidate);
        let size_c = norm1(&res_c);
        // Stop unless the residual decreased (a NaN residual also stops).
        if size_c.partial_cmp(&size) != Some(std::cmp::Ordering::Less) {
            break;
        }
        (x, res, size) = (candidate, res_c, size_c);
    }
    x
}

// ANCHOR: care
/// Solves the continuous algebraic Riccati equation `Aᴴ X + X A − X B R⁻¹ Bᴴ X + Q = 0` for the
/// stabilising solution `X` (SciPy's `linalg.solve_continuous_are`, without its `e` and `s`
/// options), for which `A − B R⁻¹ Bᴴ X` has all its eigenvalues in the left half-plane.
///
/// The Hamiltonian `H = [[A, −G], [−Q, −Aᴴ]]` with `G = B R⁻¹ Bᴴ` satisfies
/// `H [I; X] = [I; X] (A − G X)`, so `[I; X]` spans its stable invariant subspace, the null space
/// of `sign(H) + I`. The sign function is computed by Newton's iteration
/// `Z ← (Z / c + c Z⁻¹) / 2` with determinant scaling `c = |det Z|^{1/2n}`; `X` follows by least
/// squares and is polished by Newton (Kleinman) steps, each a continuous Lyapunov equation.
/// Requires `(A, B)` stabilisable and no eigenvalues of `H` on the imaginary axis.
pub fn solve_continuous_are<T: Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>, S3: Data<Elem = T>, S4: Data<Elem = T>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    q: &ArrayBase<S3, Ix2>,
    r: &ArrayBase<S4, Ix2>,
) -> Result<Array2<T>> {
    let (a, b, q, r) = riccati_inputs(a, b, q, r)?;
    let n = a.rows;
    let g = matmul(&b, &solve_left(&r, &b.adjoint())?);
    let ah = a.adjoint();
    // The Hamiltonian.
    let mut z = Mat::<C>::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            *z.at_mut(i, j) = a.at(i, j);
            *z.at_mut(i, j + n) = -g.at(i, j);
            *z.at_mut(i + n, j) = -q.at(i, j);
            *z.at_mut(i + n, j + n) = -ah.at(i, j);
        }
    }
    let mut converged = false;
    for _ in 0..100 {
        let lu = LuFactorization::new(z.clone()).map_err(|_| LinalgError::NoConvergence("the Hamiltonian has eigenvalues on the imaginary axis; sign iteration"))?;
        let log_det = (0..2 * n).map(|i| lu.lu.at(i, i).norm().ln()).sum::<f64>();
        let c = (log_det / (2 * n) as f64).exp();
        let next = lincomb(C::new(0.5 / c, 0.0), &z, C::new(0.5 * c, 0.0), &lu.inverse());
        let change = norm1(&sub(&next, &z));
        z = next;
        if change <= 1e-13 * norm1(&z) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(LinalgError::NoConvergence("the matrix sign iteration"));
    }
    // [W12; W22 + I] X = −[W11 + I; W21], solved column by column in the least-squares sense.
    let mut lhs = Array2::<C>::zeros((2 * n, n));
    let mut rhs = Array2::<C>::zeros((2 * n, n));
    let one = |yes: bool| C::new(if yes { 1.0 } else { 0.0 }, 0.0);
    for i in 0..2 * n {
        for j in 0..n {
            lhs[(i, j)] = z.at(i, j + n) + one(i == j + n);
            rhs[(i, j)] = -(z.at(i, j) + one(i == j));
        }
    }
    let mut x = Mat::<C>::zeros(n, n);
    for j in 0..n {
        let sol = lhs.least_squares(&rhs.column(j))?.solution;
        set_column(&mut x, j, sol.as_slice().expect("contiguous"));
    }
    let residual = |x: &Mat<C>| {
        let xa = matmul(x, &a);
        hermitian_part(&add(&sub(&add(&xa.adjoint(), &xa), &matmul(&matmul(x, &g), x)), &q))
    };
    let correction = |x: &Mat<C>, res: &Mat<C>| {
        // (A − G X)ᴴ Δ + Δ (A − G X) = −Res.
        let ac = sub(&a, &matmul(&g, x));
        sylvester(&ac.adjoint(), &ac, &lincomb(C::new(-1.0, 0.0), res, C::new(0.0, 0.0), &Mat::zeros(n, n)))
    };
    Ok(back(refine(hermitian_part(&x), 10, residual, correction)))
}
// ANCHOR_END: care

// ANCHOR: dare
/// Solves the discrete algebraic Riccati equation
/// `Aᴴ X A − X − Aᴴ X B (R + Bᴴ X B)⁻¹ Bᴴ X A + Q = 0` for the stabilising solution (SciPy's
/// `linalg.solve_discrete_are`, without its `e` and `s` options), for which `A − B K` with
/// `K = (R + Bᴴ X B)⁻¹ Bᴴ X A` has all its eigenvalues inside the unit circle.
///
/// The structure-preserving doubling algorithm (Chu, Fan, Lin and Wang 2004) starts from
/// `A₀ = A`, `G₀ = B R⁻¹ Bᴴ`, `H₀ = Q` and iterates
/// `A ← A (I + G H)⁻¹ A`, `G ← G + A (I + G H)⁻¹ G Aᴴ`, `H ← H + Aᴴ H (I + G H)⁻¹ A`;
/// `H` converges quadratically to `X`, which is then polished by Newton (Hewer) steps, each a
/// discrete Lyapunov equation. Requires `(A, B)` stabilisable and no eigenvalues of the
/// symplectic pencil on the unit circle; `A` need not be invertible.
pub fn solve_discrete_are<T: Scalar, S1: Data<Elem = T>, S2: Data<Elem = T>, S3: Data<Elem = T>, S4: Data<Elem = T>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
    q: &ArrayBase<S3, Ix2>,
    r: &ArrayBase<S4, Ix2>,
) -> Result<Array2<T>> {
    let (a, b, q, r) = riccati_inputs(a, b, q, r)?;
    let n = a.rows;
    let eye = Mat::<C>::identity(n);
    let (mut ak, mut gk, mut hk) = (a.clone(), matmul(&b, &solve_left(&r, &b.adjoint())?), q.clone());
    let mut converged = false;
    for _ in 0..100 {
        let w = add(&eye, &matmul(&gk, &hk));
        let t1 = solve_left(&w, &ak).map_err(|_| LinalgError::NoConvergence("the doubling algorithm (I + G H singular)"))?;
        let t2 = solve_left(&w, &gk)?;
        let h_next = add(&hk, &matmul(&matmul(&ak.adjoint(), &hk), &t1));
        let g_next = add(&gk, &matmul(&matmul(&ak, &t2), &ak.adjoint()));
        ak = matmul(&ak, &t1);
        let change = norm1(&sub(&h_next, &hk));
        (gk, hk) = (g_next, h_next);
        if change <= 1e-14 * norm1(&hk) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(LinalgError::NoConvergence("the doubling algorithm"));
    }
    let bh = b.adjoint();
    let gain = |x: &Mat<C>| -> Result<Mat<C>> {
        let bhx = matmul(&bh, x);
        solve_left(&add(&r, &matmul(&bhx, &b)), &matmul(&bhx, &a))
    };
    let residual = |x: &Mat<C>| {
        let ah_x = matmul(&a.adjoint(), x);
        let k = gain(x).unwrap_or_else(|_| Mat::zeros(b.cols, n));
        hermitian_part(&add(&sub(&sub(&matmul(&ah_x, &a), x), &matmul(&matmul(&ah_x, &b), &k)), &q))
    };
    let correction = |x: &Mat<C>, res: &Mat<C>| {
        // (A − B K)ᴴ Δ (A − B K) − Δ + Res = 0.
        let ac = sub(&a, &matmul(&b, &gain(x)?));
        stein(&ac.adjoint(), res)
    };
    Ok(back(refine(hermitian_part(&hk), 10, residual, correction)))
}
// ANCHOR_END: dare

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{expm, schur_complex};
    use ndarray::array;

    fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        (a - b).iter().fold(0.0, |m, v| m.max(v.abs()))
    }

    fn sample() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let a = array![[0.9, 0.2, 0.0, -0.3], [-0.1, 0.8, 0.3, 0.1], [0.0, -0.2, 0.7, 0.4], [0.5, 0.0, -0.6, 1.1]];
        let b = array![[1.0, 0.0], [0.5, 0.2], [0.0, 1.0], [0.3, -0.4]];
        let q = array![[2.0, 0.1, 0.0, 0.0], [0.1, 1.0, 0.0, 0.2], [0.0, 0.0, 1.5, 0.0], [0.0, 0.2, 0.0, 0.5]];
        let r = array![[1.0, 0.2], [0.2, 0.5]];
        (a, b, q, r)
    }

    #[test]
    fn schur_reconstructs_the_matrix() {
        let (a, ..) = sample();
        let (t, z) = schur_complex(&a).unwrap();
        let back = z.dot(&t).dot(&z.t().mapv(|v| v.conj()));
        assert!(back.iter().zip(a.iter()).all(|(x, y)| (x - C::new(*y, 0.0)).norm() < 1e-13));
        for i in 0..4 {
            for j in 0..i {
                assert!(t[(i, j)].norm() < 1e-14);
            }
        }
    }

    #[test]
    fn sylvester_and_lyapunov_residuals() {
        let (a, _, q, _) = sample();
        let bm = array![[-0.5, 1.0], [0.0, 2.0]];
        let qs = array![[1.0, 2.0], [0.0, -1.0], [3.0, 1.0], [0.5, 0.5]];
        let x = solve_sylvester(&a, &bm, &qs).unwrap();
        assert!(max_diff(&(a.dot(&x) + x.dot(&bm)), &qs) < 1e-12);
        let x = solve_continuous_lyapunov(&a, &q).unwrap();
        assert!(max_diff(&(a.dot(&x) + x.dot(&a.t())), &q) < 1e-12);
        let x = solve_discrete_lyapunov(&a, &q).unwrap();
        assert!(max_diff(&(a.dot(&x).dot(&a.t()) - &x + &q), &Array2::zeros((4, 4))) < 1e-11);
    }

    #[test]
    fn riccati_solutions_are_stabilising() {
        let (a, b, q, r) = sample();
        let rinv = array![[0.5 / 0.46, -0.2 / 0.46], [-0.2 / 0.46, 1.0 / 0.46]];
        let x = solve_continuous_are(&a, &b, &q, &r).unwrap();
        let res = a.t().dot(&x) + x.dot(&a) - x.dot(&b).dot(&rinv).dot(&b.t()).dot(&x) + &q;
        assert!(res.iter().all(|v| v.abs() < 1e-11), "{res}");
        let closed = &a - &b.dot(&rinv).dot(&b.t()).dot(&x);
        assert!(crate::EigVals::eigvals(&closed).unwrap().iter().all(|l| l.re < 0.0));

        let x = solve_discrete_are(&a, &b, &q, &r).unwrap();
        let s = &r + &b.t().dot(&x).dot(&b);
        let sinv = crate::Inverse::inv(&s).unwrap();
        let k = sinv.dot(&b.t()).dot(&x).dot(&a);
        let res = a.t().dot(&x).dot(&a) - &x - a.t().dot(&x).dot(&b).dot(&k) + &q;
        assert!(res.iter().all(|v| v.abs() < 1e-11), "{res}");
        let closed = &a - &b.dot(&k);
        assert!(crate::EigVals::eigvals(&closed).unwrap().iter().all(|l| l.norm() < 1.0));
    }

    #[test]
    fn expm_of_a_rotation_generator() {
        for theta in [0.001, 0.3, 2.0, 40.0] {
            let a = array![[0.0, -theta], [theta, 0.0]];
            let e = expm(&a).unwrap();
            let want = array![[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]];
            assert!(max_diff(&e, &want) < 1e-12 * theta.max(1.0), "θ = {theta}: {e}");
        }
        // A nilpotent matrix: e^N = I + N + N²/2.
        let nmat = array![[0.0, 1.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]];
        let want = array![[1.0, 1.0, 3.5], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]];
        assert!(max_diff(&expm(&nmat).unwrap(), &want) < 1e-14);
    }
}
