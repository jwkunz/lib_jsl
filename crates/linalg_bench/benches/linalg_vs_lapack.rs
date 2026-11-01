//! Compares `lib_jsl_linalg` (pure Rust) with `ndarray-linalg` (LAPACK via OpenBLAS) on the
//! operations the library uses, across matrix sizes.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lib_jsl_linalg_bench::{lapack, ours, positive_definite, random_matrix, random_vector, symmetric};
use std::hint::black_box;
use std::time::Duration;

const SIZES: [usize; 3] = [16, 64, 256];

fn configure<'a>(c: &'a mut Criterion, name: &str) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group(name);
    group.sample_size(10).warm_up_time(Duration::from_millis(300)).measurement_time(Duration::from_secs(2));
    group
}

fn solve(c: &mut Criterion) {
    let mut group = configure(c, "solve (LU)");
    for n in SIZES {
        let (a, b) = (random_matrix(n, n, 1), random_vector(n, 2));
        group.bench_with_input(BenchmarkId::new("lib_jsl_linalg", n), &n, |bench, _| bench.iter(|| ours::solve(black_box(&a), black_box(&b))));
        group.bench_with_input(BenchmarkId::new("ndarray-linalg", n), &n, |bench, _| bench.iter(|| lapack::solve(black_box(&a), black_box(&b))));
    }
    group.finish();
}

fn inverse(c: &mut Criterion) {
    let mut group = configure(c, "inverse");
    for n in SIZES {
        let a = random_matrix(n, n, 3);
        group.bench_with_input(BenchmarkId::new("lib_jsl_linalg", n), &n, |bench, _| bench.iter(|| ours::inverse(black_box(&a))));
        group.bench_with_input(BenchmarkId::new("ndarray-linalg", n), &n, |bench, _| bench.iter(|| lapack::inverse(black_box(&a))));
    }
    group.finish();
}

fn cholesky(c: &mut Criterion) {
    let mut group = configure(c, "cholesky");
    for n in SIZES {
        let a = positive_definite(n, 4);
        group.bench_with_input(BenchmarkId::new("lib_jsl_linalg", n), &n, |bench, _| bench.iter(|| ours::cholesky(black_box(&a))));
        group.bench_with_input(BenchmarkId::new("ndarray-linalg", n), &n, |bench, _| bench.iter(|| lapack::cholesky(black_box(&a))));
    }
    group.finish();
}

fn eigh(c: &mut Criterion) {
    let mut group = configure(c, "eigh (symmetric)");
    for n in SIZES {
        let a = symmetric(n, 5);
        group.bench_with_input(BenchmarkId::new("lib_jsl_linalg", n), &n, |bench, _| bench.iter(|| ours::eigh(black_box(&a))));
        group.bench_with_input(BenchmarkId::new("ndarray-linalg", n), &n, |bench, _| bench.iter(|| lapack::eigh(black_box(&a))));
    }
    group.finish();
}

fn eig(c: &mut Criterion) {
    let mut group = configure(c, "eig (general)");
    for n in SIZES {
        let a = random_matrix(n, n, 6);
        group.bench_with_input(BenchmarkId::new("lib_jsl_linalg", n), &n, |bench, _| bench.iter(|| ours::eig(black_box(&a))));
        group.bench_with_input(BenchmarkId::new("ndarray-linalg", n), &n, |bench, _| bench.iter(|| lapack::eig(black_box(&a))));
    }
    group.finish();
}

fn svd(c: &mut Criterion) {
    let mut group = configure(c, "svd (values and vectors)");
    for n in SIZES {
        let a = random_matrix(n, n, 7);
        group.bench_with_input(BenchmarkId::new("lib_jsl_linalg", n), &n, |bench, _| bench.iter(|| ours::svd(black_box(&a))));
        group.bench_with_input(BenchmarkId::new("ndarray-linalg", n), &n, |bench, _| bench.iter(|| lapack::svd(black_box(&a))));
    }
    group.finish();
}

fn least_squares(c: &mut Criterion) {
    let mut group = configure(c, "least squares (2n × n)");
    for n in SIZES {
        let (a, b) = (random_matrix(2 * n, n, 8), random_vector(2 * n, 9));
        group.bench_with_input(BenchmarkId::new("lib_jsl_linalg", n), &n, |bench, _| bench.iter(|| ours::least_squares(black_box(&a), black_box(&b))));
        group.bench_with_input(BenchmarkId::new("ndarray-linalg", n), &n, |bench, _| bench.iter(|| lapack::least_squares(black_box(&a), black_box(&b))));
    }
    group.finish();
}

criterion_group!(benches, solve, inverse, cholesky, eigh, eig, svd, least_squares);
criterion_main!(benches);
