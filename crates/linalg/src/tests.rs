//! Tests of the decompositions through the identities they must satisfy.

use super::*;
use ndarray::{Array2, array};
use num_complex::Complex;

type C = Complex<f64>;

/// A deterministic pseudo-random generator for test matrices.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    fn real(&mut self, m: usize, n: usize) -> Array2<f64> {
        Array2::from_shape_fn((m, n), |_| self.next())
    }
    fn complex(&mut self, m: usize, n: usize) -> Array2<C> {
        Array2::from_shape_fn((m, n), |_| C::new(self.next(), self.next()))
    }
}

fn adjoint<T: Scalar>(a: &Array2<T>) -> Array2<T> {
    a.t().mapv(|v| v.conj())
}

fn max_abs<T: Scalar>(a: &Array2<T>) -> f64 {
    a.iter().fold(0.0, |m, v| m.max(v.abs()))
}

fn matmul<T: Scalar>(a: &Array2<T>, b: &Array2<T>) -> Array2<T> {
    let (m, k, n) = (a.nrows(), a.ncols(), b.ncols());
    Array2::from_shape_fn((m, n), |(i, j)| (0..k).fold(T::zero(), |s, l| s + a[[i, l]] * b[[l, j]]))
}

fn identity_error<T: Scalar>(q: &Array2<T>) -> f64 {
    let g = matmul(&adjoint(q), q);
    max_abs(&(g - Array2::from_shape_fn((q.ncols(), q.ncols()), |(i, j)| if i == j { T::one() } else { T::zero() })))
}

#[test]
fn lu_solves_inverts_and_computes_determinants() {
    let mut rng = Lcg(1);
    for n in [1, 2, 7, 40, 150] {
        let a = rng.real(n, n);
        let b = Array1::from_shape_fn(n, |_| rng.next());
        let x = a.solve(&b).unwrap();
        assert!((a.dot(&x) - &b).iter().all(|v| v.abs() < 1e-10), "n={n}");
        let inv = a.inv().unwrap();
        assert!(max_abs(&(a.dot(&inv) - Array2::<f64>::eye(n))) < 1e-9, "n={n}");
        let ac = rng.complex(n, n);
        let bc = Array1::from_shape_fn(n, |_| C::new(rng.next(), rng.next()));
        let xc = ac.solve(&bc).unwrap();
        assert!((ac.dot(&xc) - &bc).iter().all(|v| v.norm() < 1e-10));
        // A factorisation is reusable.
        let xt = ac.solve_t(&bc).unwrap();
        assert!((ac.t().dot(&xt) - &bc).iter().all(|v| v.norm() < 1e-10));
        let xh = ac.solve_h_into(bc.clone()).unwrap();
        assert!((ac.t().mapv(|v| v.conj()).dot(&xh) - &bc).iter().all(|v| v.norm() < 1e-10));
        let lu = ac.factorize().unwrap();
        assert!((lu.solve(&bc).unwrap() - &xc).iter().all(|v| v.norm() < 1e-12));
    }
    assert!((array![[2.0, 1.0], [7.0, 4.0]].det().unwrap() - 1.0).abs() < 1e-14);
    assert!((array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 3.0]].det().unwrap() + 3.0).abs() < 1e-14);
    assert_eq!(array![[1.0, 2.0], [2.0, 4.0]].det().unwrap(), 0.0);
    assert!(matches!(array![[1.0, 2.0], [2.0, 4.0]].solve(&array![1.0, 1.0]), Err(LinalgError::Singular(_))));
    assert!(matches!(Array2::<f64>::zeros((2, 3)).inv(), Err(LinalgError::NotSquare { .. })));
    assert!(matches!(array![[f64::NAN]].inv(), Err(LinalgError::NonFinite)));
    assert_eq!(array![[1.0, 0.0], [0.0, 2.0]].solve_into(array![3.0, 4.0]).unwrap(), array![3.0, 2.0]);
}

#[test]
fn cholesky_and_triangular_solves() {
    let mut rng = Lcg(2);
    for n in [1, 5, 30, 150] {
        let g = rng.complex(n, n);
        let a = matmul(&g, &adjoint(&g)) + Array2::from_diag(&Array1::from_elem(n, C::new(n as f64, 0.0)));
        let l = a.cholesky(UPLO::Lower).unwrap();
        assert!(max_abs(&(matmul(&l, &adjoint(&l)) - &a)) < 1e-10, "n={n}");
        let u = a.cholesky(UPLO::Upper).unwrap();
        assert!(max_abs(&(matmul(&adjoint(&u), &u) - &a)) < 1e-10);
        // Triangular solves with matrix and vector right-hand sides.
        let b = rng.complex(n, 3);
        let x = l.solve_triangular(UPLO::Lower, Diag::NonUnit, &b).unwrap();
        assert!(max_abs(&(matmul(&l, &x) - &b)) < 1e-10);
        let v = b.column(0).to_owned();
        let y = u.solve_triangular(UPLO::Upper, Diag::NonUnit, &v).unwrap();
        assert!((u.dot(&y) - &v).iter().all(|e| e.norm() < 1e-10));
    }
    // Only the requested triangle is read.
    let a = array![[4.0, 99.0], [2.0, 3.0]];
    let l = a.cholesky(UPLO::Lower).unwrap();
    assert!((l[[1, 0]] - 1.0).abs() < 1e-15 && l[[0, 1]] == 0.0);
    let unit = array![[7.0, 0.0], [2.0, 9.0]].solve_triangular(UPLO::Lower, Diag::Unit, &array![1.0, 4.0]).unwrap();
    assert_eq!(unit, array![1.0, 2.0]);
    assert!(matches!(array![[1.0, 2.0], [2.0, 1.0]].cholesky(UPLO::Lower), Err(LinalgError::NotPositiveDefinite(1))));
}

#[test]
fn hermitian_eigenproblems() {
    let mut rng = Lcg(3);
    for n in [1, 2, 6, 50] {
        let g = rng.real(n, n);
        let a = &g + &g.t();
        let (w, v) = a.eigh(UPLO::Lower).unwrap();
        assert!(w.windows(2).into_iter().all(|p| p[0] <= p[1]));
        assert!(max_abs(&(a.dot(&v) - &v * &w)) < 1e-10, "real n={n}");
        assert!(identity_error(&v) < 1e-12);
        let h = rng.complex(n, n);
        let ah = &h + &adjoint(&h);
        let (wc, vc) = ah.eigh(UPLO::Upper).unwrap();
        let scaled = &vc * &wc.mapv(|x| C::new(x, 0.0));
        assert!(max_abs(&(matmul(&ah, &vc) - scaled)) < 1e-10, "complex n={n}");
        assert!(identity_error(&vc) < 1e-12);
    }
    // The second-difference matrix has eigenvalues 2 − 2 cos(kπ/(n+1)).
    let n = 20;
    let t = Array2::from_shape_fn((n, n), |(i, j)| if i == j { 2.0 } else if i.abs_diff(j) == 1 { -1.0 } else { 0.0 });
    let w = t.eigvalsh(UPLO::Lower).unwrap();
    for (k, &x) in w.iter().enumerate() {
        assert!((x - (2.0 - 2.0 * (std::f64::consts::PI * (k + 1) as f64 / (n + 1) as f64).cos())).abs() < 1e-12);
    }
    // Repeated eigenvalues still give an orthonormal basis.
    let (w, v) = Array2::<f64>::eye(4).eigh(UPLO::Lower).unwrap();
    assert!(w.iter().all(|&x| (x - 1.0).abs() < 1e-15) && identity_error(&v) < 1e-14);
}

#[test]
fn general_eigenproblems() {
    let mut rng = Lcg(4);
    for n in [1, 2, 5, 30] {
        let a = rng.real(n, n);
        let (w, v) = a.eig().unwrap();
        let ac = a.mapv(|x| C::new(x, 0.0));
        let residual = matmul(&ac, &v) - &v * &w;
        assert!(max_abs(&residual) < 1e-9, "real n={n}: {}", max_abs(&residual));
        // Real matrices have eigenvalues in conjugate pairs: the sum is real.
        assert!(w.iter().map(|z| z.im).sum::<f64>().abs() < 1e-9);
        let b = rng.complex(n, n);
        let (wb, vb) = b.eig().unwrap();
        assert!(max_abs(&(matmul(&b, &vb) - &vb * &wb)) < 1e-9, "complex n={n}");
        assert!((wb.iter().copied().sum::<C>() - (0..n).map(|i| b[[i, i]]).sum::<C>()).norm() < 1e-9); // trace
    }
    // A rotation by 90° has eigenvalues ±i.
    let mut w: Vec<C> = array![[0.0, -1.0], [1.0, 0.0]].eigvals().unwrap().to_vec();
    w.sort_by(|a, b| a.im.total_cmp(&b.im));
    assert!((w[0] - C::new(0.0, -1.0)).norm() < 1e-14 && (w[1] - C::new(0.0, 1.0)).norm() < 1e-14);
    // The companion matrix of (x − 1)(x − 2)(x² + 1) = x⁴ − 3x³ + 3x² − 3x + 2.
    let companion = array![[3.0, -3.0, 3.0, -2.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
    let mut roots: Vec<C> = companion.eigvals().unwrap().to_vec();
    let key = |z: &C| ((z.re * 1e8).round(), (z.im * 1e8).round());
    roots.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
    let expected = [C::new(0.0, -1.0), C::new(0.0, 1.0), C::new(1.0, 0.0), C::new(2.0, 0.0)];
    assert!(roots.iter().zip(expected).all(|(r, e)| (r - e).norm() < 1e-10), "{roots:?}");
    // An upper triangular matrix is already in Schur form.
    let upper = array![[1.0, 5.0, -2.0], [0.0, 3.0, 4.0], [0.0, 0.0, -1.0]];
    let mut eigen: Vec<f64> = upper.eigvals().unwrap().iter().map(|z| z.re).collect();
    eigen.sort_by(f64::total_cmp);
    assert_eq!(eigen, vec![-1.0, 1.0, 3.0]);
}

#[test]
fn singular_value_decomposition() {
    let mut rng = Lcg(5);
    for (m, n) in [(1, 1), (6, 4), (4, 6), (30, 20)] {
        let a = rng.real(m, n);
        let (u, s, vt) = a.svd(true, true).unwrap();
        let (u, vt) = (u.unwrap(), vt.unwrap());
        assert_eq!((u.dim(), s.len(), vt.dim()), ((m, m), m.min(n), (n, n)));
        assert!(identity_error(&u) < 1e-12 && identity_error(&vt) < 1e-12, "{m}x{n}");
        let mut sigma = Array2::<f64>::zeros((m, n));
        s.iter().enumerate().for_each(|(i, &v)| sigma[[i, i]] = v);
        assert!(max_abs(&(u.dot(&sigma).dot(&vt) - &a)) < 1e-12, "{m}x{n}");
        assert!(s.windows(2).into_iter().all(|p| p[0] >= p[1]));
        // Singular values are the square roots of the eigenvalues of AᵀA.
        let mut w = a.t().dot(&a).eigvalsh(UPLO::Lower).unwrap().to_vec();
        w.reverse();
        assert!(s.iter().zip(&w).all(|(x, y)| (x * x - y).abs() < 1e-10));
        let c = rng.complex(m, n);
        let (uc, sc, vtc) = c.svd(true, true).unwrap();
        let mut sigma = Array2::<C>::zeros((m, n));
        sc.iter().enumerate().for_each(|(i, &v)| sigma[[i, i]] = C::new(v, 0.0));
        assert!(max_abs(&(matmul(&matmul(&uc.unwrap(), &sigma), &vtc.unwrap()) - &c)) < 1e-12);
    }
    // A rank-one matrix has one non-zero singular value.
    let r = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]];
    let (_, s, _) = r.svd(false, false).unwrap();
    assert!((s[0] - (70f64).sqrt()).abs() < 1e-12 && s[1] < 1e-14);
}

#[test]
fn least_squares_solutions() {
    // Fitting a line to exact data recovers it, with zero residual.
    let x = Array1::linspace(0.0, 1.0, 11);
    let a = Array2::from_shape_fn((11, 2), |(i, j)| if j == 0 { 1.0 } else { x[i] });
    let b = x.mapv(|t| 3.0 - 2.0 * t);
    let r = a.least_squares(&b).unwrap();
    assert!((r.solution[0] - 3.0).abs() < 1e-12 && (r.solution[1] + 2.0).abs() < 1e-12);
    assert_eq!(r.rank, 2);
    assert!(r.residual_sum_of_squares.unwrap() < 1e-24);
    // Noisy overdetermined systems agree with the normal equations.
    let mut rng = Lcg(6);
    let a = rng.real(40, 5);
    let b = Array1::from_shape_fn(40, |_| rng.next());
    let normal = a.t().dot(&a).solve(&a.t().dot(&b)).unwrap();
    let r = a.least_squares(&b).unwrap();
    assert!((r.solution - normal).iter().all(|v| v.abs() < 1e-10));
    // The fast QR path reports the same singular values as the SVD.
    let sigma = a.svd(false, false).unwrap().1;
    assert!((r.singular_values - sigma).iter().all(|v| v.abs() < 1e-10));
    let complex = Lcg(7).complex(30, 4);
    let cb = Array1::from_shape_fn(30, |i| C::new(i as f64, 1.0));
    let normal = matmul(&adjoint(&complex), &complex).solve(&adjoint(&complex).dot(&cb)).unwrap();
    assert!((complex.least_squares(&cb).unwrap().solution - normal).iter().all(|v| v.norm() < 1e-9));
    // Rank-deficient and underdetermined systems give the minimum-norm solution.
    let deficient = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
    let r = deficient.least_squares(&array![2.0, 2.0, 2.0]).unwrap();
    assert_eq!(r.rank, 1);
    assert!((r.solution[0] - 1.0).abs() < 1e-12 && (r.solution[1] - 1.0).abs() < 1e-12);
    let wide = array![[1.0, 2.0, 2.0]];
    let r = wide.least_squares(&array![9.0]).unwrap();
    assert!((r.solution - array![1.0, 2.0, 2.0]).iter().all(|v| v.abs() < 1e-12));
    assert!(wide.least_squares(&array![1.0, 2.0]).is_err());
}

#[test]
fn real_schur_path_stress() {
    let mut rng = Lcg(9);
    let check = |a: &Array2<f64>, label: &str| {
        let n = a.nrows();
        let (w, v) = a.eig().unwrap();
        let ac = a.mapv(|x| C::new(x, 0.0));
        let scale = max_abs(a).max(1.0);
        let residual = max_abs(&(matmul(&ac, &v) - &v * &w));
        assert!(residual < 1e-9 * scale, "{label} n={n}: residual {residual:e}");
        // The real path agrees with the complex path on the same matrix.
        let key = |z: &C| ((z.re * 1e6).round(), (z.im * 1e6).round());
        let mut real_path: Vec<C> = a.eigvals().unwrap().to_vec();
        let mut complex_path: Vec<C> = ac.eigvals().unwrap().to_vec();
        real_path.sort_by(|x, y| key(x).partial_cmp(&key(y)).unwrap());
        complex_path.sort_by(|x, y| key(x).partial_cmp(&key(y)).unwrap());
        let gap = real_path.iter().zip(&complex_path).fold(0.0f64, |m, (x, y)| m.max((x - y).norm()));
        assert!(gap < 1e-8 * scale, "{label}: real and complex paths differ by {gap:e}");
    };
    for n in [3, 4, 64, 150] {
        check(&rng.real(n, n), "random");
    }
    let s = rng.real(40, 40);
    check(&(&s + &s.t()), "symmetric");
    check(&Array2::<f64>::eye(7), "identity");
    check(&Array2::<f64>::zeros((5, 5)), "zero");
    // A cyclic permutation: eigenvalues are the 12th roots of unity, all of equal modulus.
    check(&Array2::from_shape_fn((12, 12), |(i, j)| if (i + 1) % 12 == j { 1.0 } else { 0.0 }), "cyclic");
    // Block diagonal rotations and a Jordan-like upper bidiagonal block.
    let mut blocks = Array2::<f64>::zeros((6, 6));
    for k in [0, 2] {
        blocks[[k, k]] = 0.5;
        blocks[[k + 1, k + 1]] = 0.5;
        blocks[[k, k + 1]] = -2.0;
        blocks[[k + 1, k]] = 2.0;
    }
    blocks[[4, 4]] = 3.0;
    blocks[[5, 5]] = 3.0;
    blocks[[4, 5]] = 1e-3;
    check(&blocks, "blocks");
}

#[test]
fn preconditioned_svd_edge_cases() {
    let mut rng = Lcg(11);
    let check = |a: &Array2<f64>, label: &str| -> Array1<f64> {
        let (m, n) = a.dim();
        let (u, s, vt) = a.svd(true, true).unwrap();
        let (u, vt) = (u.unwrap(), vt.unwrap());
        assert!(identity_error(&u) < 1e-11 && identity_error(&vt) < 1e-11, "{label}: orthogonality");
        let mut sigma = Array2::<f64>::zeros((m, n));
        s.iter().enumerate().for_each(|(i, &v)| sigma[[i, i]] = v);
        assert!(max_abs(&(u.dot(&sigma).dot(&vt) - a)) < 1e-11 * max_abs(a).max(1.0), "{label}: reconstruction");
        assert!(s.windows(2).into_iter().all(|p| p[0] >= p[1]), "{label}: order");
        s
    };
    check(&rng.real(150, 150), "square");
    check(&rng.real(20, 45), "wide");
    // Rank 10 in a 40 × 30 matrix: twenty singular values vanish.
    let low = rng.real(40, 10).dot(&rng.real(10, 30));
    let s = check(&low, "rank-deficient");
    assert!(s[9] > 1e-3 && s[10] < 1e-12 * s[0], "{s}");
    // Graded singular values 10⁰ … 10⁻¹² are found to high relative accuracy.
    let (q1, _, q2) = rng.real(24, 24).svd(true, true).unwrap();
    let graded: Vec<f64> = (0..24).map(|i| 10f64.powf(-12.0 * i as f64 / 23.0)).collect();
    let a = q1.unwrap().dot(&Array2::from_diag(&Array1::from(graded.clone()))).dot(&q2.unwrap());
    let s = check(&a, "graded");
    let worst = s.iter().zip(&graded).fold(0.0f64, |m, (x, y)| m.max((x - y).abs() / y));
    assert!(worst < 1e-3, "relative error {worst:e}");
}
