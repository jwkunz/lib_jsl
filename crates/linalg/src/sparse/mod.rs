//! Sparse matrices and solvers for large systems, following `scipy.sparse` and
//! `scipy.sparse.linalg`: COO, CSR and CSC storage; a sparse LU factorisation with a
//! fill-reducing ordering (minimum degree or reverse Cuthill-McKee; `spsolve`, `SparseLu`); Krylov
//! solvers (CG, MINRES, GMRES, LGMRES, BiCGSTAB, LSQR) for any [`LinearOperator`], with Jacobi and
//! incomplete-LU preconditioners; a few eigenvalues or singular values by Lanczos methods
//! (`eigsh`, its shift-invert mode, `svds`) and of nonsymmetric matrices by Arnoldi's (`eigs`,
//! `eigs_shift_invert`); and `expm_multiply`. Real (`f64`) matrices only.
//!
//! See the book entry [Sparse matrices and iterative solvers](https://jwkunz.github.io/lib_jsl/book/linear_algebra/sparse.html).

mod arnoldi;
mod direct;
mod eigen;
mod expm;
mod formats;
mod ilu;
mod iterative;
mod lgmres;

pub use arnoldi::{EigsOptions, EigsResult, EigsWhich, eigs, eigs_shift_invert};
pub use direct::{Ordering, SparseLu, bandwidth, minimum_degree, reverse_cuthill_mckee, spsolve};
pub use eigen::{EigenPairs, Which, eigsh, eigsh_shift_invert, svds};
pub use expm::expm_multiply;
pub use ilu::Ilu;
pub use formats::{CooMatrix, CscMatrix, CsrMatrix};
pub use lgmres::{LgmresSolver, lgmres};
pub use iterative::{FnOperator, IterativeOptions, IterativeResult, Jacobi, LinearOperator, bicgstab, cg, gmres, lsqr, minres};

#[cfg(test)]
mod tests {
    use super::*;

    /// The 5-point Laplacian on an `m × m` grid (SPD, n = m²).
    fn poisson(m: usize) -> CsrMatrix {
        let mut coo = CooMatrix::new(m * m, m * m);
        for i in 0..m {
            for j in 0..m {
                let k = i * m + j;
                coo.push(k, k, 4.0);
                if i > 0 {
                    coo.push(k, k - m, -1.0);
                }
                if i + 1 < m {
                    coo.push(k, k + m, -1.0);
                }
                if j > 0 {
                    coo.push(k, k - 1, -1.0);
                }
                if j + 1 < m {
                    coo.push(k, k + 1, -1.0);
                }
            }
        }
        coo.to_csr().unwrap()
    }

    fn residual(a: &CsrMatrix, x: &[f64], b: &[f64]) -> f64 {
        a.matvec(x).iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f64>().sqrt()
    }

    #[test]
    fn formats_round_trip_and_multiply() {
        let a = CsrMatrix::from_triplets(3, 4, &[0, 2, 1, 0, 2], &[1, 3, 0, 1, 0], &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(a.get(0, 1), 5.0);
        assert_eq!(a.nnz(), 4);
        assert_eq!(a.to_csc().to_csr(), a);
        let d = a.to_dense();
        let p = a.matmul(&a.transpose()).unwrap().to_dense();
        assert!((p - d.dot(&d.t())).iter().all(|v| v.abs() < 1e-14));
    }

    #[test]
    fn direct_and_iterative_solvers_agree() {
        let a = poisson(12);
        let n = a.nrows;
        let b: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
        let x = spsolve(&a, &b).unwrap();
        assert!(residual(&a, &x, &b) < 1e-11);
        let o = IterativeOptions { rtol: 1e-10, ..Default::default() };
        for r in [cg(&a, &b, None, None, o).unwrap(), bicgstab(&a, &b, None, None, o).unwrap(), gmres(&a, &b, None, None, 30, o).unwrap(), minres(&a, &b, None, o).unwrap()] {
            assert!(r.converged, "{} iterations", r.iterations);
            assert!(x.iter().zip(&r.x).all(|(p, q)| (p - q).abs() < 1e-8));
        }
        let pre = Jacobi::new(&a).unwrap();
        assert!(cg(&a, &b, None, Some(&pre), o).unwrap().converged);
    }

    #[test]
    fn lanczos_finds_the_extreme_eigenvalues_of_the_laplacian() {
        let m = 10;
        let a = poisson(m);
        let (vals, vecs) = eigsh(&a, 3, Which::LargestAlgebraic, 0.0).unwrap();
        // Eigenvalues 4 − 2cos(iπ/(m+1)) − 2cos(jπ/(m+1)).
        let mut exact: Vec<f64> = (1..=m).flat_map(|i| (1..=m).map(move |j| 4.0 - 2.0 * (i as f64 * std::f64::consts::PI / (m + 1) as f64).cos() - 2.0 * (j as f64 * std::f64::consts::PI / (m + 1) as f64).cos())).collect();
        exact.sort_by(f64::total_cmp);
        let top = &exact[exact.len() - 3..];
        for (v, e) in vals.iter().zip(top) {
            assert!((v - e).abs() < 1e-8, "{vals:?} vs {top:?}");
        }
        let av = a.matvec(&vecs.column(2).to_vec());
        assert!(av.iter().zip(vecs.column(2)).all(|(p, q)| (p - vals[2] * q).abs() < 1e-6));
        let (s, _, _) = svds(&a, 2, 0.0).unwrap();
        assert!((s[0] - top[2]).abs() < 1e-8);
    }

    #[test]
    fn deflation_finds_both_copies_of_a_repeated_eigenvalue() {
        // On a square grid λ(i, j) = λ(j, i): the second and third largest eigenvalues coincide.
        let m = 30;
        let (vals, vecs) = eigsh(&poisson(m), 4, Which::LargestAlgebraic, 0.0).unwrap();
        let h = std::f64::consts::PI / (m + 1) as f64;
        let lam = |i: usize, j: usize| 4.0 - 2.0 * (i as f64 * h).cos() - 2.0 * (j as f64 * h).cos();
        let mut exact: Vec<f64> = (1..=m).flat_map(|i| (1..=m).map(move |j| (i, j))).map(|(i, j)| lam(i, j)).collect();
        exact.sort_by(f64::total_cmp);
        let top = &exact[exact.len() - 4..];
        for (v, e) in vals.iter().zip(top) {
            assert!((v - e).abs() < 1e-10, "{vals:?} vs {top:?}");
        }
        let gram = vecs.t().dot(&vecs);
        assert!(gram.indexed_iter().all(|((i, j), g)| (g - if i == j { 1.0 } else { 0.0 }).abs() < 1e-10));
    }

    #[test]
    fn lsqr_matches_the_normal_equations() {
        // A 30 × 8 matrix with a few entries per row.
        let mut coo = CooMatrix::new(30, 8);
        for i in 0..30 {
            coo.push(i, i % 8, 1.0 + 0.1 * i as f64);
            coo.push(i, (3 * i + 1) % 8, -0.5 + 0.03 * i as f64);
        }
        let a = coo.to_csr().unwrap();
        let b: Vec<f64> = (0..30).map(|i| (0.3 * i as f64).sin()).collect();
        let r = lsqr(&a, &b, IterativeOptions { rtol: 1e-12, ..Default::default() }).unwrap();
        assert!(r.converged);
        // Aᵀ(b − A x) = 0 at the least-squares solution.
        let res: Vec<f64> = a.matvec(&r.x).iter().zip(&b).map(|(p, q)| q - p).collect();
        assert!(a.rmatvec(&res).iter().all(|g| g.abs() < 1e-9));
        let (s, u, v) = svds(&a, 3, 0.0).unwrap();
        let dense = a.to_dense();
        for c in 0..3 {
            let av = dense.dot(&v.column(c));
            assert!(av.iter().zip(u.column(c)).all(|(p, q)| (p - s[c] * q).abs() < 1e-9));
        }
    }

    #[test]
    fn minimum_degree_reduces_fill_and_ilu_preconditions() {
        let a = poisson(20);
        let natural = SparseLu::factor(&a, Ordering::Natural).unwrap().nnz();
        let md = SparseLu::factor(&a, Ordering::MinimumDegree).unwrap().nnz();
        let rcm = SparseLu::factor(&a, Ordering::ReverseCuthillMcKee).unwrap().nnz();
        assert!(md < rcm && rcm <= natural, "md {md} rcm {rcm} natural {natural}");
        let mut perm = minimum_degree(&a);
        perm.sort_unstable();
        assert_eq!(perm, (0..a.nrows).collect::<Vec<_>>());
        let b: Vec<f64> = (0..a.nrows).map(|i| (i as f64).cos()).collect();
        let o = IterativeOptions { rtol: 1e-10, ..Default::default() };
        let plain = cg(&a, &b, None, None, o).unwrap().iterations;
        let ilu0 = Ilu::ilu0(&a).unwrap();
        assert_eq!(ilu0.nnz(), a.nnz());
        let with = gmres(&a, &b, None, Some(&ilu0), 50, o).unwrap();
        assert!(with.converged && with.iterations < plain / 2, "{} vs {plain}", with.iterations);
        let ilut = Ilu::ilut(&a, 1e-12, 1000).unwrap();
        // With no dropping ILUT is the exact LU, so one iteration suffices.
        assert!(gmres(&a, &b, None, Some(&ilut), 50, o).unwrap().iterations <= 2);
    }

    #[test]
    fn shift_invert_finds_interior_eigenvalues() {
        let m = 12;
        let a = poisson(m);
        let h = std::f64::consts::PI / (m + 1) as f64;
        let mut exact: Vec<f64> = (1..=m).flat_map(|i| (1..=m).map(move |j| 4.0 - 2.0 * (i as f64 * h).cos() - 2.0 * (j as f64 * h).cos())).collect();
        exact.sort_by(f64::total_cmp);
        let sigma = 3.3;
        let mut near: Vec<f64> = exact.clone();
        near.sort_by(|x, y| (x - sigma).abs().total_cmp(&(y - sigma).abs()));
        let mut want: Vec<f64> = near[..5].to_vec();
        want.sort_by(f64::total_cmp);
        let (vals, _) = eigsh_shift_invert(&a, 5, sigma, 0.0).unwrap();
        for (v, w) in vals.iter().zip(&want) {
            assert!((v - w).abs() < 1e-10, "{vals:?} vs {want:?}");
        }
    }
}
