//! Textbook algorithms against `lib_jsl_linalg`, on real matrices of order 16 and 64.

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use lib_jsl_linalg::{Cholesky, EigValsh, LeastSquaresSvd, SVD, Solve, UPLO};
use lib_jsl_linalg_reference::*;
use ndarray::Array1;
use std::hint::black_box;

const SIZES: [usize; 2] = [16, 64];

fn configure<'a>(c: &'a mut Criterion, name: &str) -> BenchmarkGroup<'a, WallTime> {
    let mut group = c.benchmark_group(name);
    group.sample_size(10);
    group
}

fn solve(c: &mut Criterion) {
    let mut g = configure(c, "solve");
    for n in SIZES {
        let a = random_matrix(n, n, 1);
        let b = Array1::from_elem(n, 1.0);
        g.bench_with_input(BenchmarkId::new("gaussian_elimination", n), &n, |x, _| {
            x.iter(|| gaussian_elimination::solve_with_partial_pivoting(black_box(&a), &b))
        });
        g.bench_with_input(BenchmarkId::new("doolittle_lu", n), &n, |x, _| x.iter(|| lu::lu(black_box(&a)).unwrap().solve(&b)));
        g.bench_with_input(BenchmarkId::new("lib_jsl_linalg", n), &n, |x, _| x.iter(|| black_box(&a).solve(&b)));
    }
}

fn cholesky(c: &mut Criterion) {
    let mut g = configure(c, "cholesky");
    for n in SIZES {
        let a = random_positive_definite(n, 2);
        g.bench_with_input(BenchmarkId::new("banachiewicz", n), &n, |x, _| x.iter(|| cholesky::cholesky(black_box(&a))));
        g.bench_with_input(BenchmarkId::new("lib_jsl_linalg", n), &n, |x, _| x.iter(|| black_box(&a).cholesky(UPLO::Lower)));
    }
}

fn qr(c: &mut Criterion) {
    let mut g = configure(c, "qr");
    for n in SIZES {
        let a = random_matrix(2 * n, n, 3);
        g.bench_with_input(BenchmarkId::new("classical_gram_schmidt", n), &n, |x, _| x.iter(|| qr::classical_gram_schmidt(black_box(&a))));
        g.bench_with_input(BenchmarkId::new("modified_gram_schmidt", n), &n, |x, _| x.iter(|| qr::modified_gram_schmidt(black_box(&a))));
        g.bench_with_input(BenchmarkId::new("householder", n), &n, |x, _| x.iter(|| qr::householder(black_box(&a))));
    }
}

fn eigenvalues(c: &mut Criterion) {
    let mut g = configure(c, "symmetric_eigenvalues");
    for n in SIZES {
        let a = random_symmetric(n, 4);
        g.bench_with_input(BenchmarkId::new("jacobi", n), &n, |x, _| x.iter(|| jacobi_eigenvalue::jacobi_eigen(black_box(&a), 1e-14)));
        g.bench_with_input(BenchmarkId::new("shifted_qr", n), &n, |x, _| x.iter(|| qr_algorithm::shifted_qr(black_box(&a), 1e-14, 10_000)));
        g.bench_with_input(BenchmarkId::new("lib_jsl_linalg", n), &n, |x, _| x.iter(|| black_box(&a).eigvalsh(UPLO::Lower)));
    }
}

fn svd(c: &mut Criterion) {
    let mut g = configure(c, "svd");
    for n in SIZES {
        let a = random_matrix(2 * n, n, 5);
        g.bench_with_input(BenchmarkId::new("via_normal_matrix", n), &n, |x, _| x.iter(|| naive_svd::svd_via_normal_matrix(black_box(&a))));
        g.bench_with_input(BenchmarkId::new("lib_jsl_linalg", n), &n, |x, _| x.iter(|| black_box(&a).svd(true, true)));
    }
}

fn least_squares(c: &mut Criterion) {
    let mut g = configure(c, "least_squares");
    for n in SIZES {
        let a = random_matrix(2 * n, n, 6);
        let b = Array1::from_elem(2 * n, 1.0);
        g.bench_with_input(BenchmarkId::new("normal_equations", n), &n, |x, _| x.iter(|| least_squares::normal_equations(black_box(&a), &b)));
        g.bench_with_input(BenchmarkId::new("householder_qr", n), &n, |x, _| x.iter(|| least_squares::qr_least_squares(black_box(&a), &b)));
        g.bench_with_input(BenchmarkId::new("lib_jsl_linalg", n), &n, |x, _| x.iter(|| black_box(&a).least_squares(&b)));
    }
}

criterion_group!(benches, solve, cholesky, qr, eigenvalues, svd, least_squares);
criterion_main!(benches);
